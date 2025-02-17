🍽️ Credends 🍽️ - Final Credits Detector
=========================================

Description:
------------
This script detects the starting time of the final credits in a video using an iterative search
algorithm and Google Cloud Vertex AI. It examines the final portion of the video to pinpoint the
exact frame where “final credits” appear, thereby indicating the beginning of the end credits.

Algorithm Overview:
-------------------
The algorithm examines only the last portion of the video (starting at 90% of its duration by default)
and progressively narrows the search window to identify the exact frame where the final credits begin.
The search window is defined as:

    [ start_frame --- candidate c --- candidate d --- end_frame ]

Iterative Search Process:
-------------------------
1. Pre-Candidate Double-Check:
   - Before examining candidate frames, the algorithm computes a "double-check" frame:
         double_check_frame = start_frame - (end_frame - start_frame) * CANDIDATE_FRACTION
   - If this frame exists and shows “final credits” (via Vertex AI), the algorithm immediately adjusts 
     the start of the search window to that frame, increments the correction counter, resets the shift multiplier,
     and recalculates candidate positions.

2. Candidate "c" Check:
   - The algorithm extracts candidate frame "c" and analyzes it via Vertex AI.
   - If "c" shows “final credits”, two additional frames are sampled:
       * Frame "e": Located between candidate "c" and candidate "d".
       * Frame "f": Located between candidate "d" and end_frame.
   - If both "e" and "f" confirm final credits, there is strong evidence that the credits have begun;
     the window is narrowed by setting end_frame to candidate "c".
   - Otherwise, if confirmation fails, the window (start_frame) is shifted slightly forward using an adaptive shift multiplier.

3. Candidate "d" Check:
   - If candidate "c" does not show final credits, candidate frame "d" is checked.
   - If "d" shows “final credits”, one additional frame is sampled:
       * Frame "f": Located between candidate "d" and end_frame.
   - If "f" confirms final credits, the window is adjusted by setting start_frame to candidate "c"
     and end_frame to candidate "d".
   - Otherwise, the window (start_frame) is shifted forward (with an adaptive shift multiplier).

4. No Clear Evidence:
   - When neither candidate "c" nor candidate "d" shows final credits, the search window is redefined 
     as [candidate d --- end_frame].

5. Backward Check:
   - Once the iterative narrowing reduces the window below a specified tolerance (in frames),
     the candidate final frame is set to end_frame.
   - A backward check then ensues: the algorithm moves one second at a time backward from the candidate,
     verifying that preceding frames also contain final credits.
   - This continues until a frame is found that does not contain final credits, confirming the precise boundary
     where the final credits start.

Throughout this process, an adaptive shift multiplier and a correction counter are used to control the search
progress and prevent endless loops.

Cloud Integration & Logging:
----------------------------
- The script retrieves video records from a BigQuery queue table.
- It downloads each video, processes it to detect the start of the final credits, and logs detailed processing information.
- A timestamped local log file (e.g., log_YYYYMMDD_HHMMSS.txt) is generated.
- Processing results—including detected final credits time, video ID, processing duration, and job status—are exported
  to a designated BigQuery output table.

Environment Setup:
------------------
1. Install Conda (if not already installed)
   - See: https://docs.conda.io/en/latest/miniconda.html

2. Create and Activate the Conda Environment:
   - conda create -n final_credits python=3.8 -y
   - conda activate final_credits

3. Install Required Packages:
   pip install opencv-python google-cloud-storage google-auth google-cloud-bigquery vertexai requests

4. Install Google Cloud SDK (gcloud):
   Download and install the Google Cloud SDK from:
   https://cloud.google.com/sdk/docs/install

5. Set up Application Default Credentials:
   Run the following command to authenticate your environment with Google Cloud:
       gcloud auth application-default login

Running the Script:
-------------------
To run the detector, simply execute:
   python credends.py

The script will:
   - Retrieve video items from the BigQuery queue.
   - Download and process each video to detect the final credits.
   - Log processing details in a timestamped local log file.
   - Export processing results to the designated BigQuery output table.

Configuration:
--------------
Key parameters are defined as global variables within the script and can be adjusted as needed:

   - Google Cloud Settings:
       - PROJECT_ID, REGION, BUCKET_NAME
       
   - BigQuery Tables:
       - Input Queue: BQ_QUEUE (e.g., vdm_end_credits_detection.end_credits_queue)
       - Output Table: BQ_OUTPUT (e.g., vdm_end_credits_detection.end_credits_output)
       
   - Detection Algorithm:
       - ANALYSIS_START_PERCENT (default: 0.90)
       - SEARCH_TOLERANCE (in seconds)
       - CANDIDATE_FRACTION (default: 1/3)
       - CORRECTION_THRESHOLD and BACKWARD_SEARCH_SECONDS

Troubleshooting:
----------------
- Download Failures:
  If a video fails to download, an error is logged and processing for that video is skipped.

- Processing Errors:
  Any errors during frame extraction or analysis will be logged, and the video will be marked as failed.

- Cloud Permissions:
  Verify that your service account has the necessary permissions for accessing BigQuery, Cloud Storage, and Vertex AI.
