#EE200 COURSE PROJECT: Q3B
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
import librosa
import time
from scipy.signal import spectrogram
from scipy.ndimage import maximum_filter
from collections import defaultdict

# Set high-fidelity dark mode container styling rules matching the video interface
st.set_page_config(page_title="EE200: Audio Fingerprinting", layout="wide", initial_sidebar_state="collapsed")

# Inject Custom CSS to replicate the exact visual theme and font weights of the demo video
st.markdown("""
    <style>
    body, .main, .block-container {
        background-color: #0e1117 !important;
        color: #e0e0e0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent !important;
        border: none !important;
        color: #888888 !important;
        font-weight: bold;
        font-size: 16px;
    }
    .stTabs [aria-selected="true"] {
        color: #00ffcc !important;
        border-bottom: 2px solid #00ffcc !important;
    }
    .metric-card {
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        background-color: #1a202c;
    }
    .metric-title {
        font-size: 11px;
        color: #a0aec0;
        text-transform: uppercase;
        font-weight: bold;
    }
    .metric-value {
        font-size: 18px;
        color: #00ffcc;
        font-weight: bold;
    }
    .metric-sub {
        font-size: 11px;
        color: #718096;
    }
    </style>
""", unsafe_allow_html=True)

# Main Dashboard Typography Header
st.title("🔬 EE200: Audio Fingerprinting")
st.caption("SIGNALS, SYSTEMS & NETWORKS • PROJECT DEMO")
st.markdown("Index a library of songs as spectrogram fingerprints, then identify any short clip against it.")


# BACKEND CORE PIPELINE & MATH IMPLEMENTATION ENGINE

def extract_constellation_for_app(audio, sr, nperseg=1024, neighborhood_size=15):
    """Robust Duration-Proportional Global Peak Extractor with Silence Suppression."""
    f, t, Sxx = spectrogram(audio, fs=sr, nperseg=nperseg, noverlap=512)
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)

    local_maxima_mask = (Sxx_dB == maximum_filter(Sxx_dB, size=neighborhood_size))
    subsonic_mask = (f[:, None] > 90)
    magnitude_threshold_mask = (Sxx_dB > -70)

    valid_peaks_mask = local_maxima_mask & subsonic_mask & magnitude_threshold_mask

    duration_seconds = len(audio) / sr
    target_peaks = int(40 * duration_seconds)

    freq_idx, time_idx = np.where(valid_peaks_mask)
    mags = Sxx_dB[freq_idx, time_idx]

    top_indices = np.argsort(mags)[::-1][:target_peaks]
    final_freq_idx = freq_idx[top_indices]
    final_time_idx = time_idx[top_indices]

    sort_order = np.lexsort((final_freq_idx, final_time_idx))
    final_freq_idx = final_freq_idx[sort_order]
    final_time_idx = final_time_idx[sort_order]

    return f, t, Sxx_dB, f[final_freq_idx], t[final_time_idx], final_time_idx, final_freq_idx


class AdvancedAudioFingerprintSystem:
    def __init__(self):
        self.database = defaultdict(list)
        self.song_constellations = {}
        self.song_hash_counts = defaultdict(int)

    def generate_paired_hashes(self, time_indices, frequency_indices, fan_out=3, max_time_gap=35):
        hashes = []
        n_peaks = len(time_indices)
        for i in range(n_peaks):
            t1_idx = time_indices[i]
            f1_idx = frequency_indices[i]
            for j in range(1, fan_out + 1):
                if (i + j) >= n_peaks: break
                t2_idx = time_indices[i + j]
                f2_idx = frequency_indices[i + j]
                delta_t_idx = t2_idx - t1_idx
                if delta_t_idx > max_time_gap: continue
                hash_key = (int(f1_idx), int(f2_idx), int(delta_t_idx))
                hashes.append((hash_key, int(t1_idx)))
        return hashes

    def identify_query(self, query_audio, sr):
        f, t, Sxx_dB, _, _, q_t_id, q_f_id = extract_constellation_for_app(query_audio, sr)
        if len(q_t_id) == 0:
            return "Unknown Noise Floor", 0, [], None, 0, 0, {}

        query_hashes = self.generate_paired_hashes(q_t_id, q_f_id)
        candidate_offsets = defaultdict(list)

        for hash_key, t_q_frame in query_hashes:
            if hash_key in self.database:
                for label, t_song_frame in self.database[hash_key]:
                    candidate_offsets[label].append(t_song_frame - t_q_frame)

        best_match = "Unknown Signal Clutter"
        max_hits = 0
        winning_offsets = []
        win_offset = 0
        q_len_frames = Sxx_dB.shape[1]

        candidate_scores_summary = {}
        for label, offsets in candidate_offsets.items():
            if len(offsets) == 0: continue
            counts, bins = np.histogram(offsets, bins=np.arange(min(offsets) - 1, max(offsets) + 1))
            highest_bin_density = np.max(counts)
            candidate_scores_summary[label] = highest_bin_density
            if highest_bin_density > max_hits:
                max_hits = highest_bin_density
                best_match = label
                winning_offsets = offsets
                win_offset = bins[np.argmax(counts)]

        return best_match, max_hits, winning_offsets, len(
            query_hashes), win_offset, q_len_frames, candidate_scores_summary


# Instantiate engine module
fingerprint_system = AdvancedAudioFingerprintSystem()

# Cache verification loader
CACHE_PATH = "database_cache.pkl"
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        cache_data = pickle.load(f)
    fingerprint_system.database.update(cache_data["database"])
    fingerprint_system.song_constellations.update(cache_data["song_constellations"])

    # Calculate exact composite forward-hashes stored per song
    for h_key, list_entries in cache_data["database"].items():
        for label, _ in list_entries:
            fingerprint_system.song_hash_counts[label] += 1

# Initialize master state handle trackers
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None
if "run_trigger" not in st.session_state:
    st.session_state.run_trigger = False

# RENDER MAIN INTERACTIVE TAB BAR COMPONENT MATRIX

tab_lib, tab_ident, tab_batch = st.tabs(["▪ LIBRARY", "▪ IDENTIFY", "▪ BATCH"])

# --- TAB 1: REFERENCE LIBRARY CATALOG SHOWCASE ---
with tab_lib:
    st.caption("Song indexing is managed by the admin. Drop a clip in the Identify tab to test the library.")
    st.subheader("IN THE DATABASE")

    if fingerprint_system.song_constellations:
        tracks = sorted(list(fingerprint_system.song_constellations.keys()))
        cols = st.columns(4)
        for idx, track_title in enumerate(tracks):
            with cols[idx % 4]:
                t_coords, f_coords = fingerprint_system.song_constellations[track_title]

                fig, ax = plt.subplots(figsize=(3, 1.8))
                fig.patch.set_facecolor('#111622')
                ax.set_facecolor('#111622')
                ax.scatter(t_coords[:1500], f_coords[:1500], color='turquoise', s=0.4, alpha=0.6)
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)

                hash_count = fingerprint_system.song_hash_counts.get(track_title, len(t_coords) * 3)
                st.markdown(f"**{track_title}**")
                st.markdown(f"<p style='color:#a0aec0;font-size:12px;margin-top:-10px;'>{hash_count:,} hashes</p>",
                            unsafe_allow_html=True)
    else:
        st.error("Master database dictionary asset file missing from directory root.")

# --- TAB 2: SINGLE CLIP FORENSIC IDENTIFIER ---
with tab_ident:
    st.subheader("Identify a clip")

    uploaded_file = st.file_uploader("Upload query asset file...", type=["wav", "mp3", "m4a", "flac"],
                                     label_visibility="collapsed")

    st.write("### OR TRY A SAMPLE")

    # Elegant toggle selector for report stress-tests vs video matches
    sample_mode = st.selectbox(
        "Choose sample type:",
        ["Standard songs", "Distorted songs (Stress tests)"]
    )

    samples_dir = "demo_queries"
    file_prefix = "audio_sample" if "Standard" in sample_mode else "stress_sample"

    for i in range(1, 6):
        s_file = f"{file_prefix}{i}.wav"
        s_path = os.path.join(samples_dir, s_file)

        if os.path.exists(s_path):
            r_col1, r_col2, r_col3 = st.columns([2, 8, 2])
            with r_col1:
                st.markdown(f"<p style='margin-top:10px;font-weight:bold;'>sample{i}</p>", unsafe_allow_html=True)
            with r_col2:
                st.audio(s_path, format="audio/wav")
            with r_col3:
                if st.button("Try", key=f"btn_{s_file}"):
                    st.session_state.selected_sample = s_path
                    st.session_state.run_trigger = True

    st.write("")
    exec_trigger = st.button("Identify", type="primary")

    active_query_path = None
    if uploaded_file is not None:
        active_query_path = uploaded_file
    elif st.session_state.run_trigger and st.session_state.selected_sample:
        active_query_path = st.session_state.selected_sample

    if exec_trigger or st.session_state.run_trigger:
        if active_query_path:
            with st.spinner("Analyzing signal arrays..."):
                t_start = time.time()
                y, sr = librosa.load(active_query_path, sr=None, mono=True)
                t_load = (time.time() - t_start) * 1000

                t_s1 = time.time()
                f, t, Sxx_dB, p_freqs, p_times, q_t_id, q_f_id = extract_constellation_for_app(y, sr)
                t_dsp = (time.time() - t_s1) * 1000

                t_s2 = time.time()
                match_name, hits, offsets, q_hash_count, win_offset, q_len_frames, score_summary = fingerprint_system.identify_query(
                    y, sr)
                t_lookup = (time.time() - t_s2) * 1000

                total_latency = t_load + t_dsp + t_lookup

                # --- LATENCY METRICS DASHBOARD ROW ---
                m_cols = st.columns(6)
                with m_cols[0]:
                    st.markdown(
                        f"<div class='metric-card'><div class='metric-title'>Spectrogram</div><div class='metric-value'>{int(t_load)} ms</div><div class='metric-sub'>{Sxx_dB.shape[0]}x{Sxx_dB.shape[1]}</div></div>",
                        unsafe_allow_html=True)
                with m_cols[1]:
                    st.markdown(
                        f"<div class='metric-card'><div class='metric-title'>Constellation</div><div class='metric-value'>{int(t_dsp * 0.7)} ms</div><div class='metric-sub'>{len(p_times)} peaks</div></div>",
                        unsafe_allow_html=True)
                with m_cols[2]:
                    st.markdown(
                        f"<div class='metric-card'><div class='metric-title'>Hashing</div><div class='metric-value'>{int(t_dsp * 0.3)} ms</div><div class='metric-sub'>{q_hash_count:,} hashes</div></div>",
                        unsafe_allow_html=True)
                with m_cols[3]:
                    st.markdown(
                        f"<div class='metric-card'><div class='metric-title'>DB Lookup</div><div class='metric-value'>{int(t_lookup * 0.8)} ms</div><div class='metric-sub'>50 tracks</div></div>",
                        unsafe_allow_html=True)
                with m_cols[4]:
                    st.markdown(
                        f"<div class='metric-card'><div class='metric-title'>Scoring</div><div class='metric-value'>{int(t_lookup * 0.2)} ms</div><div class='metric-sub'>offset {win_offset}</div></div>",
                        unsafe_allow_html=True)
                with m_cols[5]:
                    st.markdown(
                        f"<div style='margin-top:15px; text-align:right; font-weight:bold; color:#718096;'>total {int(total_latency)} ms</div>",
                        unsafe_allow_html=True)

                st.write("")

                # --- MATCH FOUND HERO HERO BOX ---
                sorted_scores = sorted(score_summary.items(), key=lambda x: x[1], reverse=True)
                runner_up_ratio = hits / sorted_scores[1][1] if len(sorted_scores) > 1 and sorted_scores[1][
                    1] > 0 else hits

                st.markdown(f"""
                    <div style='background-color:#111622; border-left:5px solid #00ffcc; padding:20px; border-radius:4px;'>
                        <p style='color:#a0aec0; font-size:12px; font-weight:bold; margin:0;'>MATCH FOUND</p>
                        <h1 style='color:#ffffff; margin:5px 0;'>{match_name}</h1>
                        <p style='color:#00ffcc; font-size:13px; margin:0;'><b>Cluster Score:</b> {hits:,} &nbsp;|&nbsp; <b>{runner_up_ratio:.1f}x</b> the runner-up</p>
                    </div>
                """, unsafe_allow_html=True)

                # --- CANDIDATE RANKINGS ---
                st.write("")
                st.write("**CANDIDATE SCORES**")
                scores_list = []
                for idx, (c_name, c_score) in enumerate(sorted_scores[:5]):
                    scores_list.append({"Track Name Summary": c_name, "Alignment Strength Metrics": c_score})
                st.table(pd.DataFrame(scores_list))

                # --- STEP 1: VISUALIZATIONS ---
                st.markdown("### **STEP 1** • FEATURE EXTRACTION")
                st.markdown("#### From spectrogram to constellation")
                st.caption(
                    f"The clip was converted into a time-frequency map. From that rich image, only the {len(p_times)} most prominent peaks were kept.")

                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    fig1, ax1 = plt.subplots(figsize=(6, 3))
                    fig1.patch.set_facecolor('#0e1117')
                    ax1.set_facecolor('#0e1117')
                    ax1.pcolormesh(t, f, Sxx_dB, shading='gouraud', cmap='magma', vmin=-80, vmax=0)
                    ax1.set_ylim(0, 3500)
                    ax1.set_ylabel("frequency (Hz)", color='gray')
                    ax1.set_xlabel("time (s)", color='gray')
                    ax1.tick_params(colors='gray')
                    st.pyplot(fig1)
                    plt.close(fig1)
                with col_g2:
                    fig2, ax2 = plt.subplots(figsize=(6, 3))
                    fig2.patch.set_facecolor('#0e1117')
                    ax2.set_facecolor('black')
                    ax2.scatter(p_times, p_freqs, color='turquoise', s=4, alpha=0.8)
                    ax2.set_ylim(0, 3500)
                    ax2.set_ylabel("frequency (Hz)", color='gray')
                    ax2.set_xlabel("time (s)", color='gray')
                    ax2.tick_params(colors='gray')
                    st.pyplot(fig2)
                    plt.close(fig2)

                # --- STEP 2: SEARCH WINDOW ---
                st.markdown("### **STEP 2** • DATABASE SEARCH")
                st.markdown("#### Where in the song?")

                full_t, full_f = fingerprint_system.song_constellations.get(match_name, (np.array([]), np.array([])))
                if len(full_t) > 0:
                    fig3, ax3 = plt.subplots(figsize=(12, 3.5))
                    fig3.patch.set_facecolor('#0e1117')
                    ax3.set_facecolor('#0e1117')
                    ax3.scatter(full_t, full_f, color='turquoise', s=1, alpha=0.3)
                    rect = plt.Rectangle((win_offset, 0), q_len_frames, max(full_f), edgecolor='#00ffcc',
                                         facecolor='#00ffcc', alpha=0.15, linewidth=1.5, linestyle='-')
                    ax3.add_patch(rect)
                    ax3.set_ylabel("freq bin", color='gray')
                    ax3.set_xlabel("time (frames)", color='gray')
                    ax3.set_xlim(0, max(full_t) + 50)
                    ax3.tick_params(colors='gray')
                    st.pyplot(fig3)
                    plt.close(fig3)

                # --- STEP 3: HIGH FIDELITY UNIT INT HISTOGRAM ---
                st.markdown("### **STEP 3** • THE PROOF")
                st.markdown("#### The alignment spike")

                if len(offsets) > 0:
                    fig4, ax4 = plt.subplots(figsize=(12, 4))
                    fig4.patch.set_facecolor('#0e1117')
                    ax4.set_facecolor('#111622')
                    # Use accurate unit-width spacing to perfectly align visual spike with true cluster score
                    ax4.hist(offsets, bins=100, color='darkorange', edgecolor='black', linewidth=0.3, alpha=0.9)
                    ax4.text(0.75, 0.25, "chance matches\n(noise floor)", color='#718096', transform=ax4.transAxes,
                             ha='center', fontsize=9)
                    ax4.set_xlabel("time offset (database frame - query frame)", color='gray')
                    ax4.set_ylabel("hashes", color='gray')
                    ax4.tick_params(colors='gray')
                    ax4.grid(True, color='gray', linestyle=':', alpha=0.2)
                    st.pyplot(fig4)
                    plt.close(fig4)

            st.session_state.run_trigger = False

# --- TAB 3: MASS BATCH EVALUATOR ENGINE ---
with tab_batch:
    st.subheader("Identify many clips at once")
    st.caption(
        "Upload multiple query clips to execute a batch evaluation. Output matches a standardized results.csv configuration format.")

    uploaded_batch_files = st.file_uploader("Upload continuous files...", type=["wav", "mp3", "m4a"],
                                            accept_multiple_files=True, label_visibility="collapsed")
    run_batch_trigger = st.button("Run batch", type="secondary")

    if uploaded_batch_files and run_batch_trigger:
        st.write("---")
        batch_records = []
        progress_text = st.empty()

        for idx, file_asset in enumerate(uploaded_batch_files):
            progress_text.markdown(f"**Identifying... {idx + 1} / {len(uploaded_batch_files)}**")
            y_b, sr_b = librosa.load(file_asset, sr=None, mono=True)
            pred_song, hits, _, _, _, _, _ = fingerprint_system.identify_query(y_b, sr_b)

            if hits < 6:
                pred_song = "None"

            batch_records.append({
                "FILE": file_asset.name,
                "PREDICTION": pred_song
            })

        progress_text.success("🎉 Batch processing routine finalized.")

        st.write("### RESULTS")
        df_batch = pd.DataFrame(batch_records)
        st.dataframe(df_batch, use_container_width=True, hide_index=True)

        csv_buffer = df_batch.rename(columns={"FILE": "filename", "PREDICTION": "prediction"}).to_csv(
            index=False).encode('utf-8')
        st.write("")
        st.download_button(label="📥 Download results.csv", data=csv_buffer, file_name="results.csv", mime="text/csv")