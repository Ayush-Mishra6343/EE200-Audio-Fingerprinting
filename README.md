# 🔬 EE200: Audio Fingerprinting Dashboard

### Signals, Systems & Networks — Course Project
**Indian Institute of Technology Kanpur (IITK)**

A high-fidelity audio fingerprinting system implemented via a Streamlit web interface. Based on the landmark-pairing algorithm framework (the core concept behind Shazam), this application extracts time-frequency spectrogram features, builds constellation maps, generates robust forward-hashes, and identifies short or distorted audio clips against an indexed library in milliseconds.

---

## 🚀 Key Features

* **Robust Feature Extraction Engine**: Computes Short-Time Fourier Transforms (STFT) coupled with 2D neighborhood local maximum filters for duration-proportional peak collection.
* **Combinatorial Hashing**: Links peak pairs into translation-invariant hash keys `(f1, f2, delta_t)` anchored to absolute time offsets, making lookups immune to environmental noise and clipping.
* **Noise Floor & False-Positive Guard**: Evaluates candidate matches using relative time-offset histograms. Implements a rigorous **safety matching threshold of 30 hashes** to confidently filter out random noise or stress-test anomalies.
* **Theme-Agnostic UI Optimization**: Injects custom high-contrast CSS to guarantee crisp text legibility and bright teal/white contrast rows regardless of whether the browser is running native Light or Dark mode.
* **Granular Latency Profiling**: Real-time analytical dashboard tracks runtime execution metrics across Spectrogram creation, Constellation mapping, Hashing, Database search, and Histogram scoring.
* **Batch Processing Platform**: Supports multi-file sequential query processing that automatically aggregates batch test data into a downloadable, standardized `results.csv` sheet.

---

## 📂 Project Architecture

```text
├── .gitignore                   # Prevents tracking local metadata and temporary cache files
├── README.md                    # Project documentation and user guide
├── app.py                       # Main Streamlit web application layout and pipeline
├── requirements.txt             # Required Python libraries (librosa, scipy, streamlit, etc.)
├── EE200 Q3.ipynb               # Core assignment Jupyter Notebook containing calculations
├── database_cache.pkl           # Pre-computed binary hash fingerprint database loaded by app.py
├── songs/                       # Directory containing full-length reference audio dataset (.mp3)
└── demo_queries/                # Standard and distorted audio samples used for UI stress tests (.wav)
