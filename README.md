🍽️ Credends 🍽️ - Final Credits Detector
=======================

Description:
------------
This script detects the starting time of final credits in a video using an iterative search
algorithm and Google Cloud Vertex AI.

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
   - If this frame exists  and shows “final subtitles” (via Vertex AI),
     the algorithm immediately adjusts the start of the search window to that frame,
     increments the correction counter, resets the shift multiplier, and recalculates candidate positions.

2. Candidate "c" Check:
   - The algorithm extracts candidate frame "c" and analyzes it via Vertex AI.
   - If "c" shows “final subtitles”, two additional frames are sampled:
       * Frame "e": Located between candidate "c" and candidate "d".
       * Frame "f": Located between candidate "d" and end_frame.
   - If both "e" and "f" confirm final subtitles, there is strong evidence that the credits have begun;
     the window is narrowed by setting end_frame to candidate "c".
   - Otherwise, if confirmation fails, the window (start_frame) is shifted slightly forward using an adaptive shift multiplier.

3. Candidate "d" Check:
   - If candidate "c" does not show final subtitles, candidate frame "d" is checked.
   - If "d" shows “final subtitles”, one additional frame is sampled:
       * Frame "f": Located between candidate "d" and end_frame.
   - If "f" confirms final subtitles, the window is adjusted by setting start_frame to candidate "c"
     and end_frame to candidate "d".
   - Otherwise, the window (start_frame) is shifted forward (with an adaptive shift multiplier).

4. No Clear Evidence:
   - When neither candidate "c" nor candidate "d" shows final subtitles,  new region is set to be
     
     [candidate d --- end_frame]
     

5. Backward Check:
   - Once the iterative narrowing reduces the window below a specified tolerance (in frames),
     the candidate final frame is set to end_frame.
   - A backward check then ensues: the algorithm moves one second at a time backward from the candidate,
     verifying that  preceding frames also contains final subtitles.
   - This continues until a frame is found that does not contain final subtitles,
     confirming the precise boundary where the final credits start.

Throughout this process, an adaptive shift multiplier and a correction counter are used to control the search
progress and prevent endless loops.

Environment Setup:
------------------
1. Install Conda (if not already installed)
   - See: https://docs.conda.io/en/latest/miniconda.html

2. Create and Activate the Conda Environment:
   ```bash
   conda create -n final_credits python=3.8 -y
   conda activate final_credits
   ```

3. Install Required Packages:
   ```bash
   pip install opencv-python google-cloud-storage google-auth vertexai
   ```

Running the Script:
-------------------
Usage:

Download Service account key "prima-video-intelligence...json" and modify the file in the setup.txt

   ```bash
   python credends.py -i input/input1.csv -o output/output1 -s setup.txt
   ```

Parameters:
```bash
   -i, --input    : CSV file with video paths and (optional) ground truth times.
   -o, --output   : Folder where results (and logs) will be saved.
   -s, --setup    : Setup configuration file with Google Cloud and analysis settings.
```

Input File Format (input.csv):
------------------------------
Each line should have:
   <video_path>,<ground_truth_time>
Example:
```bash
   C:\path\to\video1.mp4,44:15
   C:\path\to\video2.mp4,
   C:\path\to\video3.mp4,42:15
```
<ground_truth_time> is NOT mandatory

Output File (output.csv):
-------------------------
After processing, an output.csv is created in the specified output folder.

Columns:
   video_path, detected_time, ground_truth_time (or empty)

   - detected_time is in HH:MM:SS format (or "0" if detection fails).
