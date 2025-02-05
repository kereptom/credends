#!/usr/bin/env python
import os
import sys
import time
import cv2
import csv
import tempfile
import shutil
import argparse
import logging

import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    Part,
    HarmCategory,
    HarmBlockThreshold,
    SafetySetting,
)

# ---------------------------------------------------------------------------
# Global configuration variables (to be set after reading the setup file)
PROJECT_ID = None
REGION = None
CREDENTIALS_PATH = None
BUCKET_NAME = None
working_folder_gcp = None
model = None
credentials = None
storage_client = None
safety_settings = None

# Configuration parameters (loaded from setup, with default values)
ANALYSIS_START_PERCENT = None         # Analyze from 90% of the video duration by default
SEARCH_TOLERANCE = None               # Tolerance in seconds used in the search algorithm
CANDIDATE_FRACTION = None             # Fraction used to compute candidate frame positions (default 1/3)
BACKWARD_SEARCH_SECONDS = None        # Maximum seconds to search backward for boundary verification
CORRECTION_THRESHOLD = None           # Maximum allowed corrections before aborting

# ------------------------- Helper Functions ----------------------------------

def delete_frame_from_gcs(bucket_name, file_path):
    """Deletes a file from GCS if it exists."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    if blob.exists():
        blob.delete()

def contains_final_subtitles(description):
    """Returns True if the provided text description contains the keyword."""
    keywords = ["final subtitles"]
    return any(keyword in description.lower() for keyword in keywords)

def section_search(video_path, fps, start, end, tolerance):
    """
    Given a video file and a time window (start to end in seconds),
    performs an iterative search to locate the frame that shows final subtitles.
    
    Implements adaptive shifting with a correction counter.
    
    If the total number of corrections exceeds CORRECTION_THRESHOLD (loaded from setup),
    the function logs an error and returns 0.
    """
    global CORRECTION_THRESHOLD, CANDIDATE_FRACTION, BACKWARD_SEARCH_SECONDS
    correction_count = 0

    # Convert tolerance (seconds) to frames.
    frame_tolerance = tolerance * fps

    start_frame = int(start * fps)
    end_frame = int(end * fps)
    shift_multiplier = 1  # For adaptive shifting

    # Initial candidate points within the window.
    c = start_frame + (end_frame - start_frame) * CANDIDATE_FRACTION
    d = end_frame - (end_frame - start_frame) * CANDIDATE_FRACTION

    while (end_frame - start_frame) > frame_tolerance:
        if correction_count >= CORRECTION_THRESHOLD:
            logging.error("Exceeded correction threshold of %d corrections. Aborting search.", CORRECTION_THRESHOLD)
            return 0

        logging.info("Search window: start_frame=%d, end_frame=%d, shift_multiplier=%d", start_frame, end_frame, shift_multiplier)

        # --- Double-Check Before Window ---
        double_check_frame = int(start_frame - (end_frame - start_frame) * CANDIDATE_FRACTION)
        if double_check_frame >= 0:
            frame_check = extract_frame(video_path, double_check_frame, fps)
            description_check = describe_frame(frame_check)
            if contains_final_subtitles(description_check):
                logging.info("Double-check: frame %d contains final subtitles. Adjusting start_frame to %d.",
                                double_check_frame, double_check_frame)
                start_frame = double_check_frame
                correction_count += 1
                shift_multiplier = 1
                c = start_frame + (end_frame - start_frame) * CANDIDATE_FRACTION
                d = end_frame - (end_frame - start_frame) * CANDIDATE_FRACTION
                continue
        # --- End Double-Check ---

        # Check frame at candidate point c.
        frame_c = extract_frame(video_path, int(c), fps)
        description_c = describe_frame(frame_c)
        if contains_final_subtitles(description_c):
            # Further sample two candidate points (e and f) within the region.
            e = c + (d - c) / 2
            f = d + (end_frame - d) / 2
            frame_e = extract_frame(video_path, int(e), fps)
            frame_f = extract_frame(video_path, int(f), fps)
            description_e = describe_frame(frame_e)
            description_f = describe_frame(frame_f)
            if contains_final_subtitles(description_e) and contains_final_subtitles(description_f):
                logging.info("Found strong evidence of final subtitles at candidate region; narrowing window.")
                end_frame = int(c)
                shift_multiplier = 1  # Reset adaptive shift
            else:
                old_start = start_frame
                shift_amount = int((2 ** shift_multiplier) * frame_tolerance)
                start_frame = int(start_frame + shift_amount)
                logging.info("Insufficient confirmation in branch A; shifting start_frame from %d to %d (2**%d * frame_tolerance = %d)",
                             old_start, start_frame, shift_multiplier, shift_amount)
                shift_multiplier += 1
                correction_count += 1
        else:
            # Check frame at candidate point d.
            frame_d = extract_frame(video_path, int(d), fps)
            description_d = describe_frame(frame_d)
            if contains_final_subtitles(description_d):
                f = d + (end_frame - d) / 2
                frame_f = extract_frame(video_path, int(f), fps)
                description_f = describe_frame(frame_f)
                if contains_final_subtitles(description_f):
                    logging.info("Found valid credits in branch B; adjusting window to between c and d.")
                    start_frame = int(c)
                    end_frame = int(d)
                    shift_multiplier = 1
                else:
                    old_start = start_frame
                    shift_amount = int((2 ** shift_multiplier) * frame_tolerance)
                    start_frame = int(start_frame + shift_amount)
                    logging.info("Insufficient confirmation in branch B; shifting start_frame from %d to %d (2**%d * frame_tolerance = %d)",
                                 old_start, start_frame, shift_multiplier, shift_amount)
                    shift_multiplier += 1
                    correction_count += 1
            else:
                # When neither candidate point c nor d shows final subtitles.  
                logging.info("No evidence at c or d; setting start_frame to d (%d).", int(d))
                old_start = start_frame
                shift_amount = int((2 ** shift_multiplier) * frame_tolerance)
                start_frame = int(d)
                logging.info("Shifting start_frame from %d to %d (resetting window to d, multiplier=%d, shift=%d)",
                             old_start, start_frame, shift_multiplier, shift_amount)
                shift_multiplier = 1  # Reset after a full window reset

        # Recalculate candidate points for the next iteration.
        c = start_frame + (end_frame - start_frame) * CANDIDATE_FRACTION
        d = end_frame - (end_frame - start_frame) * CANDIDATE_FRACTION

    # Candidate found. Now perform backward double-check to ensure that the frame just before does not contain credits.
    final_frame = int(end_frame)
    logging.info("Iterative search converged with candidate final_frame=%d", final_frame)
    max_back_steps = int(BACKWARD_SEARCH_SECONDS * fps)  # use configurable backward search seconds
    back_steps = 0
    candidate_frame = final_frame
    while back_steps < max_back_steps:
        prev_frame_num = candidate_frame - fps
        frame_prev = extract_frame(video_path, prev_frame_num, fps)
        description_prev = describe_frame(frame_prev)
        if contains_final_subtitles(description_prev):
            logging.info("Backward check: frame %d also contains final subtitles. Shifting candidate backward.", prev_frame_num)
            candidate_frame = prev_frame_num
            back_steps += fps
        else:
            logging.info("Backward check: frame %d does not contain final subtitles. Boundary confirmed.", prev_frame_num)
            break
    logging.info("Final detected start of credits at frame %d", candidate_frame)
    return candidate_frame

def extract_frame(video_path, frame_number, fps):
    """
    Extracts a frame from the video at a given frame number.
    Saves the frame as "frame_<frame_number>.png" in the current directory.
    Returns the filename.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()
    if success:
        frame_path = f"frame_{frame_number}.png"
        cv2.imwrite(frame_path, frame)
        cap.release()
    else:
        cap.release()
        raise Exception(f"Failed to extract frame {frame_number}")
    return frame_path

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """
    Uploads a file to GCS.
    Returns the GCS URI.
    """
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    return f"gs://{bucket_name}/{destination_blob_name}"

def describe_frame(frame_path):
    """
    Uses Vertex AI to analyze the frame image.
    Returns the text response.
    """
    gcs_uri = upload_to_gcs(BUCKET_NAME, frame_path, working_folder_gcp + os.path.basename(frame_path))
    prompt = f"""
    Analyze the frame. Determine if the frame contains **final subtitles** that indicate the ending phase of a show or movie.
    Indicators include:
    1. Text suggesting "watch next" (e.g., 'Uvidíte příště') or similar phrases in Czech.
    2. Names and titles (e.g., director, producer, cast, crew) or phrases like 'dramatický tým.'
    3. Text overlays on black or semi-transparent backgrounds, commonly used for end credits.
    4. Contextual clues of ending transitions, such as a fade to black (with appearing credits), or previews of the next episode.

    **Warning**: Not all text overlays indicate the end. Unrelated overlays may include in-movie dialogue, captions, or action-related text that is part of the story.

    Use content, context, and style to determine if the frame represents end credits or related transitions.
    If yes, respond with 'Final Subtitles.' If no, respond with 'Continue.'
    """
    response = model.generate_content(
        [Part.from_uri(gcs_uri, mime_type="image/png"), prompt],
        safety_settings=safety_settings
    )
    delete_frame_from_gcs(BUCKET_NAME, working_folder_gcp + os.path.basename(frame_path))
    return response.text

def process_video(video_path, ground_truth):
    """
    Processes a single video file.
    Returns the detected end-credit time as a string in HH:MM:SS format.
    If any error occurs, returns "0".
    Also logs the status for this video including:
      - Video length (HH:MM:SS)
      - End credits time (HH:MM:SS)
      - Full processing time (in seconds)
    """
    detected_time = "0"
    orig_dir = os.getcwd()
    temp_dir = None
    processing_start_time = time.perf_counter()
    try:
        logging.info("Processing video path: %s", video_path)
        temp_dir = tempfile.mkdtemp()
        os.chdir(temp_dir)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Failed to open video")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_duration = total_frames / fps

        # Format video length as HH:MM:SS.
        v_hours = int(total_duration // 3600)
        v_minutes = int((total_duration % 3600) // 60)
        v_seconds = int(total_duration % 60)
        video_length_formatted = f"{v_hours:02d}:{v_minutes:02d}:{v_seconds:02d}"
        cap.release()

        # Analyze the last portion of the video, starting from ANALYSIS_START_PERCENT of its duration.
        analysis_start_time = ANALYSIS_START_PERCENT * total_duration
        analysis_end_time = total_duration

        subtitle_frame = section_search(video_path, fps, analysis_start_time, analysis_end_time, SEARCH_TOLERANCE)
        frame_time = subtitle_frame / fps
        hours = int(frame_time // 3600)
        minutes = int((frame_time % 3600) // 60)
        seconds = int(frame_time % 60)
        detected_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        processing_time = time.perf_counter() - processing_start_time

        logging.info("Processing video path: %s: successful", video_path)
        logging.info("Video length: %s, End credits time: %s, Full processing time: %.2f seconds",
                     video_length_formatted, detected_time, processing_time)
    except Exception as e:
        processing_time = time.perf_counter() - processing_start_time
        logging.error("Processing video path: %s: error: %s", video_path, str(e))
        logging.error("Processing time for video %s: %.2f seconds", video_path, processing_time)
        detected_time = "0"
    finally:
        try:
            os.chdir(orig_dir)
        except Exception:
            pass
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    return detected_time

# ----------------------------- Main ------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect end credits in videos using Google Cloud Vertex AI."
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Input CSV file path containing video paths and optional ground truth times.")
    parser.add_argument("-o", "--output", required=True,
                        help="Output folder where results will be saved.")
    parser.add_argument("-s", "--setup", required=True,
                        help="Setup configuration file path.")
    args = parser.parse_args()

    # Ensure output folder exists.
    output_folder = os.path.abspath(args.output)
    if os.path.exists(output_folder):
        # Create (or ensure) an "archive" subfolder inside the output folder.
        archive_dir = os.path.join(output_folder, "archive")
        if not os.path.exists(archive_dir):
            os.makedirs(archive_dir)

        # Determine the next available version folder name (e.g. archive_v1, archive_v2, etc.)
        version = 1
        archive_version_folder = os.path.join(archive_dir, f"archive_v{version}")
        while os.path.exists(archive_version_folder):
            version += 1
            archive_version_folder = os.path.join(archive_dir, f"archive_v{version}")
        os.makedirs(archive_version_folder)

        # Move all files/subfolders (except the archive folder) into the new versioned archive folder.
        for item in os.listdir(output_folder):
            if item == "archive":
                continue  # Don't move the archive folder itself
            src_path = os.path.join(output_folder, item)
            dst_path = os.path.join(archive_version_folder, item)
            shutil.move(src_path, dst_path)
    else:
        os.makedirs(output_folder)

    # Set up logging to both a file and the terminal.
    log_path = os.path.join(output_folder, "log.txt")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

    # Console handler (prints to terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (writes to log file)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logging.info("Starting processing. Logs will be output to both terminal and %s", log_path)

    # Load configuration from the setup file.
    config = {}
    with open(args.setup, "r", encoding="utf-8") as f:
        exec(f.read(), {}, config)

    # Set global configuration variables.
    global PROJECT_ID, REGION, CREDENTIALS_PATH, BUCKET_NAME, working_folder_gcp, model, credentials, storage_client, safety_settings
    global ANALYSIS_START_PERCENT, SEARCH_TOLERANCE, CANDIDATE_FRACTION, BACKWARD_SEARCH_SECONDS, CORRECTION_THRESHOLD

    PROJECT_ID = config.get("PROJECT_ID")
    REGION = config.get("REGION")
    CREDENTIALS_PATH = config.get("CREDENTIALS_PATH")
    BUCKET_NAME = config.get("BUCKET_NAME")
    working_folder_gcp = config.get("working_folder_gcp")
    model_name = config.get("GENERATIVE_MODEL", "gemini-1.5-flash-001")

    # New parameters loaded from setup.
    ANALYSIS_START_PERCENT = config.get("ANALYSIS_START_PERCENT", 0.90)
    SEARCH_TOLERANCE = config.get("SEARCH_TOLERANCE", 0.2)
    CANDIDATE_FRACTION = config.get("CANDIDATE_FRACTION", 1/3)
    BACKWARD_SEARCH_SECONDS = config.get("BACKWARD_SEARCH_SECONDS", 5)
    CORRECTION_THRESHOLD = config.get("CORRECTION_THRESHOLD", 30)

    from google.oauth2 import service_account
    credentials = service_account.Credentials.from_service_account_file(
        os.path.join(os.getcwd(), CREDENTIALS_PATH)
    )
    vertexai.init(project=PROJECT_ID, location=REGION, credentials=credentials)
    from google.cloud import storage
    storage_client = storage.Client(credentials=credentials)
    model = GenerativeModel(model_name)
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    results = []  # Will hold rows: [video_path, detected_time, ground_truth_time]

    # Read the input CSV.
    with open(args.input, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row or len(row) == 0:
                continue
            video_path = row[0].strip()
            ground_truth = row[1].strip() if len(row) > 1 else ""
            detected_time = process_video(video_path, ground_truth)
            results.append([video_path, detected_time, ground_truth])

    # Write the results to output.csv in the output folder (unchanged).
    output_csv_path = os.path.join(output_folder, "output.csv")
    with open(output_csv_path, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["video_path", "detected_time", "ground_truth_time"])
        writer.writerows(results)

    logging.info("Processing completed. Results saved to %s", output_csv_path)

if __name__ == "__main__":
    main()
