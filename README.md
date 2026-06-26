# 🔬 EE200: Robust Audio Fingerprinting System

### Signals, Systems & Networks — Course Project Report
**Indian Institute of Technology Kanpur (IITK)**

A high-fidelity audio fingerprinting system implemented via a Streamlit web application. Based on the landmark-pairing algorithm framework (the core concept behind Shazam), this application extracts time-frequency spectrogram features, constructs robust constellation maps, generates combinatorial forward-hashes, and identifies short, noisy, or time-shifted audio clips against an indexed library in milliseconds.

---

## 📋 Theoretical Framework & Core Concept

The core objective of an audio fingerprinting system is to identify an unknown audio query by matching it against a pre-indexed database of reference tracks. This implementation relies on the fact that while noise, compression, and channel distortion alter the absolute amplitude of an audio signal, the local maxima (the highest energy peaks) in the time-frequency spectrogram remain highly stable.

### 1. Spectrogram Peak Extraction
The incoming continuous audio signal is converted into a digital representation and evaluated via a Short-Time Fourier Transform (STFT). To find robust landmark points, a 2D neighborhood maximum filter scans the log-spectrogram space:

    S_dB(f, t) = 10 * log10(|STFT{y}|^2 + epsilon)

A coordinate (f, t) is selected as a peak if its magnitude is the absolute highest within a localized neighborhood window (15 x 15) and its energy passes a baseline silence suppression threshold of -70 dB. Subsonic filtering ignores all frequencies below 90 Hz to bypass low-frequency industrial hums, ground loops, and non-lethal noise floor baselines.

### 2. Combinatorial Hashing
To make the search invariant to global time shifts, individual peaks are not indexed alone. Instead, an anchor peak (t1, f1) is combinatorially paired with a target peak (t2, f2) inside a forward-looking target zone bounded by a maximum time gap (max_time_gap = 35). This generates a translation-invariant hash key:

    Hash Key = (f1, f2, t2 - t1)

This key is stored in a hash table dictionary linked to the absolute time offset of the anchor from the start of the track: Delta_t_anchor = t1.

### 3. Time-Coherence Alignment & Scoring
When an unidentified audio clip is analyzed, its query hashes are generated using the exact same processing pipeline. For every match found in the database, the relative time difference is computed:

    Offset = t_song_anchor - t_query_anchor

If the query is a true match for a specific database track, thousands of matching hashes will share the exact same structural time offset, forming a massive, sharp spike in the delta histogram. Random noise, cross-talk, or incorrect songs will have scattered, uniformly distributed offsets.

---

## 🚀 Key System Features

* **Robust Feature Extraction Engine**: Computes high-fidelity STFT arrays coupled with a duration-proportional peak collection mechanism to maintain matching integrity across varying clip lengths.
* **Combinatorial Hashing Matrix**: Implements a dedicated forward-looking fan-out pairing routine to build high-entropy keys that resist intense environmental distortion.
* **Noise Floor & False-Positive Guard**: Evaluates candidate matches using relative time-offset histograms. Implements a rigorous **safety matching threshold of 30 hashes** to confidently filter out random noise or stress-test anomalies during evaluation.
* **Theme-Agnostic UI Optimization**: Injects custom high-contrast CSS to guarantee crisp text legibility and bright teal/white contrast rows regardless of whether the browser is running native Light or Dark mode settings.
* **Granular Latency Profiling**: Real-time analytical dashboard tracks runtime execution metrics down to the millisecond across Spectrogram creation, Constellation mapping, Hashing, Database search, and Histogram scoring.
* **Batch Processing Platform**: Supports multi-file sequential query processing that automatically aggregates batch test data into a downloadable, standardized results.csv sheet.

---

## 📂 Project Architecture

    ├── .gitignore                   # Prevents tracking local metadata and temporary cache files
    ├── README.md                    # Project documentation and user guide
    ├── app.py                       # Main Streamlit web application layout and pipeline
    ├── requirements.txt             # Required Python libraries (librosa, scipy, streamlit, etc.)
    ├── EE200 Q3.ipynb               # Core assignment Jupyter Notebook containing calculations
    ├── database_cache.pkl           # Pre-computed binary hash fingerprint database loaded by app.py
    ├── songs/                       # Directory containing full-length reference audio dataset (.mp3)
    └── demo_queries/                # Standard and distorted audio samples used for UI stress tests (.wav)

---

## 💻 Local Setup & Execution

Follow these step-by-step instructions to configure the environment and execute the system locally on your workstation.

### 1. Clone the Workspace
Open your terminal window and clone the project repository from GitHub:

    git clone https://github.com/Ayush-Mishra6343/EE200-Audio-Fingerprinting.git
    cd EE200-Audio-Fingerprinting

### 2. Activate the Virtual Environment
Ensure your local Python environment is isolated. If you are running on Windows using PowerShell, execute:

    .venv\Scripts\Activate.ps1

If you are operating on a macOS or Linux platform, run:

    source .venv/bin/activate

### 3. Install Required Dependencies
Deploy all required mathematical, signal processing, and user interface libraries using pip:

    pip install -r requirements.txt

### 4. Launch the Streamlit Dashboard
Execute the server script to boot up the interactive interface:

    python -m streamlit run app.py

Once initialized, the local deployment server will automatically spin up and launch a fresh viewport tab inside your default web browser at http://localhost:8501.

---

## 📊 Streamlit User Interface Guide

The application features a structured three-tab navigation matrix designed for comprehensive system evaluation.

### Tab 1: Library Catalog Showcase
* Displays an overview of all reference tracks currently indexed inside the system database.
* Renders localized time-frequency constellation point maps for each track.
* Displays the total number of pre-computed hash tokens stored per song.

### Tab 2: Single Clip Forensic Identifier
* Allows users to upload custom query audio assets (.wav, .mp3, .m4a, .flac) via a drag-and-drop panel.
* Features a quick-test dock providing instant selection between Standard queries and Distorted stress-test samples.
* Renders side-by-side graphical extractions (Log-Spectrogram view vs. Clean Constellation Map).
* Generates an interactive delta-time alignment histogram displaying the exact statistical match spike.

### Tab 3: Mass Batch Evaluator Engine
* Allows for the concurrent upload of multiple query tracks to test bulk processing throughput.
* Automatically processes files sequentially, applying the matching safety threshold to filter false matches.
* Generates an on-screen tracking dataframe showing file names alongside corresponding predictions.
* Provides a one-click download button for a standardized results.csv tracking file matching the requested template format.

---

## 🛠️ Troubleshooting & Configuration

### 1. Light Mode Text Visibility
If your native browser configuration is set to Light Mode, Streamlit may attempt to override text parameters to dark gray. The application includes a custom CSS injection block targeting data-testid="stTable" to hard-force cells to high-contrast white #ffffff and headers to teal #00ffcc, rendering text perfectly across all machine configurations.

### 2. Threshold Constraints
The matching limit is bounded at 30 hashes. If a query yields peak alignments below this index line, the dashboard triggers an intentional IDENTIFICATION FAILED block, reporting the trace as None or Signal Confidence Too Low to prevent noise floor artifacts from causing false classifications.

---

## 🎓 Academic Submission Notice
This system has been developed as an official final project package submission for EE200 (Signals, Systems & Networks) at the Indian Institute of Technology Kanpur (IITK). All core digital signal processing functions, algorithmic mathematical derivations, and noise-floor validation thresholds were fully sandboxed, tested, and validated inside the workspace notebook (EE200 Q3.ipynb) prior to UI compilation and deployment.
