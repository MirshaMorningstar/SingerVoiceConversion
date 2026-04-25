"""
AI Raga Voice Changer — MovieRagaNet
Tamil Song Emotion Converter · 8 Emotions · Professional Raga Conversion
Run: streamlit run app.py
"""

import os, io, json, tempfile
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "theme_conversion_helpers"))
import streamlit as st
import pandas as pd
import numpy as np
import soundfile as sf

st.set_page_config(
    page_title="AI Raga Voice Changer",
    layout="wide",
    page_icon="🎵",
)

from core.raga_extractor    import (full_extraction_report, extract_swara_profile,
                                    extract_note_sequence, EMOTION_NOTE_PATTERNS,
                                    extract_advanced_acoustics)
from core.raga_transformer  import convert_song
from core.reference_library import (save_reference, list_all_references,
                                    delete_reference)
from core.report_generator  import (get_emotion_raga_table, get_note_change_table,
                                    get_all_pairs_change_matrix,
                                    get_song_analysis_row, build_batch_df)
from core.raga_knowledge_base  import EMOTION_RAGA_MAP, RAGA_SWARAS
from core.persistence_manager  import save_conversion, load_all_conversions, delete_conversion
from core.song_ground_truth    import lookup_song, compute_similarity, compare_s1_s2
from core.enhanced_transformer import enhance

for _k, _v in {
    "batch_rows":     [],
    "results":        [],
    "persist_loaded": False,
    "swara_profile":  None,
    "note_sequence":  None,
    "analysis_done":  False,
    "uploaded_name":  None,
    "acoustic_features": None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

if not st.session_state.persist_loaded:
    _loaded = load_all_conversions()
    if _loaded:
        st.session_state.results    = _loaded
        st.session_state.batch_rows = [r["row"] for r in _loaded]
    st.session_state.persist_loaded = True

EMOJI = {
    "Happy":"😊","Sad":"😢","Angry":"😡","Fearful":"😨",
    "Disgusted":"🤢","Surprised":"😲","Peaceful":"😌","Romantic":"💕",
}

st.title("AI Raga Voice Changer")
st.caption("Tamil Song Emotion Converter · 8 Ragas · Swara-Level Transformation")
st.divider()

(tab_convert, tab_results, tab_swara,
 tab_raga, tab_batch, tab_notechange,
 tab_allpairs, tab_ground, tab_ref, tab_math) = st.tabs([
    "Convert Song",
    "Conversion Results",
    "Swara Analysis",
    "Raga–Emotion Table",
    "Batch Results",
    "Note Change Comparison",
    "All Emotion Pairs Matrix",
    "Ground Truth & S1↔S2",
    "Reference Library",
    "Mathematical & DSP Report",
])

with tab_convert:
    col_upload, col_emotion = st.columns([3, 2])

    with col_upload:
        st.subheader("Step 1 — Upload Tamil Song")
        uploaded = st.file_uploader(
            "Choose a Tamil MP3 or WAV file",
            type=["mp3", "wav", "flac", "ogg"],
        )
        if uploaded:
            st.audio(uploaded)
            st.caption(uploaded.name)
            if st.session_state.uploaded_name != uploaded.name:
                st.session_state.swara_profile = None
                st.session_state.note_sequence = None
                st.session_state.acoustic_features = None
                st.session_state.analysis_done = False
                st.session_state.uploaded_name = uploaded.name

    with col_emotion:
        st.subheader("Step 2 — Select Target Emotion")
        target_emotion = st.selectbox(
            "Convert to emotion:",
            list(EMOTION_RAGA_MAP.keys()),
            format_func=lambda e: f"{EMOJI[e]}  {e}  →  {EMOTION_RAGA_MAP[e]}",
        )
        target_raga = EMOTION_RAGA_MAP[target_emotion]
        t_info      = RAGA_SWARAS[target_raga]
        st.info(
            f"**Raga:** {target_raga}  \n"
            f"**Melakarta:** {t_info['melakarta']}  \n"
            f"**Swaras:** {' · '.join(t_info['swaras'])}  \n"
            f"**Arohanam:** {t_info['arohanam']}  \n"
            f"**Avarohanam:** {t_info['avarohanam']}"
        )

    st.divider()

    if uploaded:
        analyse_btn = st.button(
            "Step 3 — Analyse Swaras & Identify Emotion",
            disabled=st.session_state.analysis_done,
            use_container_width=True,
        )
        if analyse_btn:
            # FIXED: save with original filename so song name lookup works correctly
            with tempfile.TemporaryDirectory() as _tdir:
                tmp_path = os.path.join(_tdir, uploaded.name)
                with open(tmp_path, "wb") as _f:
                    _f.write(uploaded.getvalue())
                prog = st.progress(0, text="Extracting pitch profile...")
                st.session_state.swara_profile = extract_swara_profile(tmp_path)
                prog.progress(50, text="Extracting note sequence (60s)...")
                st.session_state.note_sequence = extract_note_sequence(tmp_path, duration_sec=60)
                prog.progress(75, text="Extracting mathematical DSP features...")
                st.session_state.acoustic_features = extract_advanced_acoustics(tmp_path, duration_sec=60)
                prog.progress(100, text="Analysis complete.")
                st.session_state.analysis_done = True

    sp = st.session_state.swara_profile
    ns = st.session_state.note_sequence

    if sp and ns:
        st.divider()
        st.subheader("Swara Note Pattern Detected")
        st.caption(
            f"Tonic: {ns['tonic_name']}  |  "
            f"Duration: {ns['duration_analysed']}s  |  "
            f"Events: {ns['total_notes']}"
        )

        ps_rows = [
            {"Time": r["time"], "Second": r["second"],
             "Note": r["note"], "Full Name": r["full_name"]}
            for r in ns["per_second"]
        ]
        st.dataframe(pd.DataFrame(ps_rows), hide_index=True, use_container_width=True)

        st.divider()
        st.subheader("Consecutive Note Flow")
        st.caption("Each entry is one held note — consecutive duplicates merged.")
        events = ns.get("note_events", [])
        for rs in range(0, min(len(events), 80), 16):
            st.markdown("`" + "` → `".join(events[rs:rs+16]) + "`")

        st.divider()
        st.subheader("Emotion Identification — Cosine Similarity vs All 8 Ragas")
        st.caption("Highest score = detected emotion.")

        best             = sp["all_raga_scores"][0]
        detected_raga    = best["raga"]
        detected_emotion = best["emotion"]
        detected_score   = best["score"]

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Detected Raga",    detected_raga)
        mc2.metric("Detected Emotion", detected_emotion)
        mc3.metric("Cosine Score",     f"{detected_score:.4f}")

        score_rows = [
            {"Rank": i+1, "Raga": s["raga"], "Emotion": s["emotion"],
             "Cosine Score": round(s["score"], 4),
             "Match": "Detected" if i == 0 else ""}
            for i, s in enumerate(sp["all_raga_scores"])
        ]
        st.dataframe(pd.DataFrame(score_rows).astype(str), hide_index=True, use_container_width=True)

        st.divider()
        st.subheader("Raw Mathematical & Acoustic DSP Features")
        st.caption(f"Specifically calculated for: {st.session_state.uploaded_name}")
        af = st.session_state.acoustic_features
        if af:
            af_df = pd.DataFrame(list(af.items()), columns=["Acoustic Feature", "Calculated Value"])
            st.dataframe(af_df, hide_index=True, use_container_width=True)

        st.divider()
        st.subheader(
            f"Step 4 — Convert:  {detected_raga} ({detected_emotion})  →  "
            f"{target_raga} ({target_emotion})"
        )
        convert_btn = st.button("Convert Song", use_container_width=True)

        if convert_btn:
            original_bytes = uploaded.getvalue()
            original_name  = uploaded.name

            with tempfile.TemporaryDirectory() as tmpdir:
                in_path  = os.path.join(tmpdir, original_name)
                out_path = os.path.join(tmpdir, f"converted_{target_emotion.lower()}.wav")

                with open(in_path, "wb") as f:
                    f.write(original_bytes)

                report = full_extraction_report(in_path)

                prog2 = st.progress(0, text="Step 1 / 3 — Raga conversion (phase vocoder)...")
                log = convert_song(
                    audio_path     = in_path,
                    target_emotion = target_emotion,
                    source_raga    = report["detected_raga"],
                    tonic          = report["_tonic"],
                    y              = report["_y"],
                    sr             = report["_sr"],
                    output_path    = out_path,
                )

                prog2.progress(75, text="Step 2 / 2 — Saving...")
                enh_log = {}

                with open(out_path, "rb") as f:
                    converted_bytes = f.read()

                row = get_song_analysis_row(
                    filename         = original_name,
                    duration         = report["duration_sec"],
                    tempo            = report["tempo_bpm"],
                    dominant_swaras  = report["dominant_swaras"],
                    detected_raga    = report["detected_raga"],
                    detected_emotion = report["detected_emotion"],
                    match_score      = report["match_score"],
                    target_emotion   = target_emotion,
                    target_raga      = target_raga,
                    pitch_shift      = log["pitch_shift_st"],
                    tempo_factor     = log["tempo_factor"],
                    notes_removed    = log["notes_removed"],
                    notes_added      = log["notes_added"],
                )

                folder = save_conversion(
                    original_bytes  = original_bytes,
                    original_name   = original_name,
                    converted_bytes = converted_bytes,
                    report          = report,
                    log             = log,
                    row             = row,
                    target_emotion  = target_emotion,
                    target_raga     = target_raga,
                )

                st.session_state.results.append({
                    "original_bytes":  original_bytes,
                    "original_name":   original_name,
                    "converted_bytes": converted_bytes,
                    "report":          report,
                    "log":             log,
                    "row":             row,
                    "target_emotion":  target_emotion,
                    "target_raga":     target_raga,
                    "enh_log":         enh_log,
                    "folder":          folder,
                })
                st.session_state.batch_rows.append(row)
                prog2.progress(100, text="Done.")

            st.success(
                f"Converted: **{detected_raga}** ({detected_emotion})  →  "
                f"**{target_raga}** ({target_emotion}). "
                f"Go to Conversion Results tab."
            )

            src_set_ = set(RAGA_SWARAS.get(detected_raga, {}).get("semitones", []))
            tgt_set_ = set(RAGA_SWARAS.get(target_raga,   {}).get("semitones", []))
            overlap  = len(src_set_ & tgt_set_) / max(len(tgt_set_), 1)
            cc1,cc2,cc3,cc4 = st.columns(4)
            cc1.metric("Source Raga",  detected_raga)
            cc2.metric("Target Raga",  target_raga)
            cc3.metric("ID Score",     f"{detected_score:.4f}")
            cc4.metric("Note Overlap", f"{overlap:.0%}")


with tab_results:
    st.subheader("Conversion Results")
    st.caption("Every conversion is saved permanently.")

    if not st.session_state.results:
        st.info("No conversions yet.")
    else:
        labels = [
            f"{i+1}.  {r['original_name']}  →  {r['target_emotion']}  ({r['target_raga']})"
            for i, r in enumerate(st.session_state.results)
        ]
        idx = st.selectbox("Select conversion:", range(len(labels)),
                           format_func=lambda i: labels[i],
                           index=len(labels)-1)
        res         = st.session_state.results[idx]
        report      = res["report"]
        log         = res["log"]
        row         = res["row"]
        tgt_emotion = res["target_emotion"]
        tgt_raga    = res["target_raga"]
        src_raga    = report.get("detected_raga", "Unknown")
        src_emotion = report.get("detected_emotion", "Unknown")

        def _vals(d):
            if isinstance(d, dict): return [str(v) for v in d.values()]
            if isinstance(d, list): return [str(v) for v in d]
            return []

        st.divider()
        a1, a2 = st.columns(2)
        with a1:
            st.markdown("**Original Song**")
            st.caption(f"{res['original_name']}  |  {src_raga}  ({src_emotion})")
            st.audio(res["original_bytes"])
            st.download_button("Download Original", res["original_bytes"],
                               res["original_name"], key=f"dl_orig_{idx}")
        with a2:
            st.markdown(f"**Converted — {tgt_emotion}**")
            st.caption(f"Raga: {tgt_raga}  |  Shift: {log.get('pitch_shift_st',0):+.1f} st  |  Tempo: ×{log.get('tempo_factor',1.0)}")
            st.audio(res["converted_bytes"], format="audio/wav")
            st.download_button(
                f"Download {tgt_emotion} version",
                res["converted_bytes"],
                f"{res['original_name'].rsplit('.',1)[0]}_{tgt_emotion}.wav",
                mime="audio/wav", key=f"dl_conv_{idx}",
            )

        st.divider()
        st.subheader("Detection Summary")
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("Source Raga",    src_raga)
        m2.metric("Source Emotion", src_emotion)
        m3.metric("Cosine Score",   f"{report.get('match_score',0):.4f}")
        m4.metric("Tempo BPM",      f"{report.get('tempo_bpm',0):.0f}")
        m5.metric("Duration",       f"{report.get('duration_sec',0):.1f}s")
        m6.metric("Target Raga",    tgt_raga)

        st.divider()
        st.subheader(f"Note Changes : {src_raga} → {tgt_raga}")
        nc1,nc2,nc3 = st.columns(3)
        with nc1:
            st.markdown("**Removed**")
            rl = _vals(log.get("notes_removed",{}))
            for n in rl: st.markdown(f"- `{n}`")
            if not rl: st.markdown("- None")
        with nc2:
            st.markdown("**Added**")
            al = _vals(log.get("notes_added",{}))
            for n in al: st.markdown(f"- `{n}`")
            if not al: st.markdown("- None")
        with nc3:
            st.markdown("**Shared**")
            sl = _vals(log.get("notes_shared",{}))
            for n in sl: st.markdown(f"- `{n}`")
            if not sl: st.markdown("- None")

        st.divider()
        st.subheader("Detailed Note-Change Table")
        df_nc = get_note_change_table(src_emotion, tgt_emotion)
        st.dataframe(df_nc, hide_index=True, use_container_width=True)
        st.download_button("Download Note-Change Table (CSV)",
                           df_nc.to_csv(index=False).encode(),
                           f"note_changes_{src_emotion}_{tgt_emotion}.csv",
                           key=f"dl_nc_{idx}")

        st.divider()
        st.subheader("Full Song Parameters")
        df_row = pd.DataFrame([row])
        st.dataframe(df_row.T.rename(columns={0:"Value"}).astype(str), use_container_width=True)

        st.divider()
        enh_log = res.get("enh_log", {})
        if enh_log:
            st.subheader("Signal Processing Applied")
            ec1,ec2,ec3,ec4,ec5 = st.columns(5)
            ec1.metric("Spectral Tilt",  f"{enh_log.get('spectral_tilt_db',0):+.1f} dB")
            ec2.metric("Formant Shift",  f"×{enh_log.get('formant_shift_ratio',1):.2f}")
            ec3.metric("Harmonic Boost", f"+{enh_log.get('harmonic_boost_db',0):.1f} dB")
            ec4.metric("Vibrato Depth",  f"{enh_log.get('vibrato_depth_st',0):.2f} st")
            ec5.metric("Blend",          enh_log.get("blend_ratio","–"))
            with st.expander("Processing steps"):
                for s in enh_log.get("steps_applied",[]): st.markdown(f"- {s}")

        sep_method = log.get("separation_method", "–")
        ref_count  = log.get("ref_profiles_used", 0)
        si1, si2 = st.columns(2)
        si1.info(
            f"**Vocal Separation:** {'Demucs (deep learning)' if sep_method=='demucs' else 'HPSS (spectral masking)'}  \n"
            f"BGM is preserved and recombined with converted vocals."
        )
        si2.info(
            f"**Reference Library Used:** {ref_count} song(s) for {tgt_emotion}  \n"
            f"{'Shift map calibrated from real recordings.' if ref_count > 0 else 'No references — theory-only shifts applied.'}"
        )

        st.divider()
        d1,d2 = st.columns(2)
        with d1:
            if st.button("Delete this conversion", key=f"del_{idx}"):
                f = res.get("folder")
                if f: delete_conversion(f)
                st.session_state.results.pop(idx)
                st.session_state.batch_rows = [r["row"] for r in st.session_state.results]
                st.rerun()
        with d2:
            if st.button("Clear ALL conversions"):
                for r in st.session_state.results:
                    f = r.get("folder")
                    if f: delete_conversion(f)
                st.session_state.results        = []
                st.session_state.batch_rows     = []
                st.session_state.persist_loaded = False
                st.rerun()


with tab_swara:
    st.subheader("Swara Analysis — Note Sequence & Ground Truth")
    ns = st.session_state.note_sequence
    sp = st.session_state.swara_profile

    if ns is None:
        st.info("Upload a song and click Analyse Swaras in the Convert Song tab first.")
    else:
        st.subheader("Note Sequence — Line by Line (First 60s)")
        st.caption(f"Tonic: {ns['tonic_name']}  |  Events: {ns['total_notes']}  |  Duration: {ns['duration_analysed']}s")
        ps_rows = [{"Time": r["time"], "Second": r["second"],
                    "Note": r["note"], "Full Name": r["full_name"]}
                   for r in ns["per_second"]]
        st.dataframe(pd.DataFrame(ps_rows), hide_index=True, use_container_width=True)
        st.divider()

        st.subheader("Consecutive Note Flow")
        events = ns.get("note_events", [])
        for rs in range(0, min(len(events), 80), 16):
            st.markdown("`" + "` → `".join(events[rs:rs+16]) + "`")
        st.divider()

        st.subheader("Note Frequency Count")
        hist = ns.get("note_histogram", {})
        if hist:
            hs = sorted(hist.items(), key=lambda x: x[1], reverse=True)
            hc = st.columns(min(len(hs), 8))
            for i,(n,c) in enumerate(hs[:8]): hc[i].metric(n, f"{c}s")
        st.divider()

        st.subheader("Ground Truth — 8 Emotion Reference Patterns")
        st.caption("Compare your song's notes against each emotion's reference. Higher overlap = stronger match.")
        detected_notes = set(r["note"] for r in ns["per_second"] if r["note"] not in ("–", None))
        emotion_overlaps = []
        for emotion, data in EMOTION_NOTE_PATTERNS.items():
            ref_set     = set(data["typical_seq"])
            overlap     = detected_notes & ref_set
            overlap_pct = round(len(overlap)/max(len(ref_set),1)*100, 1)
            emotion_overlaps.append((emotion, data, overlap, overlap_pct))
        emotion_overlaps.sort(key=lambda x: x[3], reverse=True)

        for emotion, data, overlap, overlap_pct in emotion_overlaps:
            with st.expander(
                f"{EMOJI[emotion]}  {emotion}  —  {data['raga']}  |  Overlap: {overlap_pct}%",
                expanded=(overlap_pct == emotion_overlaps[0][3])
            ):
                ca, cb = st.columns([2,3])
                with ca:
                    st.markdown(f"**Raga:** {data['raga']}")
                    st.markdown(f"**Scale:** {data['scale']}")
                    st.markdown(f"**Character:** {data['character']}")
                    st.metric("Overlap", f"{overlap_pct}%")
                with cb:
                    st.markdown("**Reference sequence:**")
                    st.markdown("`" + "` → `".join(data["typical_seq"]) + "`")
                    st.markdown(f"**Matched notes:** `{'  ·  '.join(sorted(overlap)) or 'None'}`")
                    ref_seq  = data["typical_seq"]
                    song_seq = [r["note"] for r in ns["per_second"]
                                if r["note"] not in ("–",None)][:len(ref_seq)]
                    cmp_rows = [{
                        "Position": i+1,
                        "Ref Note": ref_seq[i] if i < len(ref_seq) else "–",
                        "Song Note": song_seq[i] if i < len(song_seq) else "–",
                        "Match": "Yes" if (i<len(ref_seq) and i<len(song_seq)
                                           and ref_seq[i]==song_seq[i]) else "No",
                    } for i in range(max(len(ref_seq), len(song_seq)))]
                    st.dataframe(pd.DataFrame(cmp_rows), hide_index=True, use_container_width=True)

        st.divider()
        if sp:
            best = sp["all_raga_scores"][0]
            be   = emotion_overlaps[0]
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Detected Raga",    best["raga"])
            c2.metric("Detected Emotion", best["emotion"])
            c3.metric("Cosine Score",     f"{best['score']:.4f}")
            c4.metric("Note Overlap",     f"{be[3]}%")
        st.download_button("Download Note Sequence (CSV)",
                           pd.DataFrame(ps_rows).to_csv(index=False).encode(),
                           "note_sequence.csv")


with tab_raga:
    st.subheader("Master Raga–Emotion Comparison Table")
    st.caption("35 columns · All 8 ragas · Swara presence · Theory · Scale · Gamaka")
    df_raga = get_emotion_raga_table()
    st.dataframe(df_raga, hide_index=True, use_container_width=True)
    st.divider()
    for emotion, raga in EMOTION_RAGA_MAP.items():
        info = RAGA_SWARAS[raga]
        with st.expander(f"{EMOJI[emotion]} {emotion} → {raga}"):
            ca,cb,cc = st.columns(3)
            ca.metric("Swaras",       len(info["swaras"]))
            cb.metric("Tempo Factor", info["tempo_factor"])
            cc.metric("Melakarta",    info["melakarta"])
            st.markdown(f"**Arohanam:** {info['arohanam']}")
            st.markdown(f"**Avarohanam:** {info['avarohanam']}")
            st.markdown(f"**Tamil Songs:** {' / '.join(info['tamil_songs'])}")
            st.caption(info["description"])


with tab_batch:
    st.subheader("Batch Song Analysis Table")
    if st.session_state.batch_rows:
        df_b = build_batch_df(st.session_state.batch_rows).astype(str)
        st.dataframe(df_b, hide_index=True, use_container_width=True)
        st.download_button("Download CSV", df_b.to_csv(index=False).encode(), "raga_results.csv")
        if st.button("Clear Batch"):
            st.session_state.batch_rows = []
            st.rerun()
    else:
        st.info("No conversions yet.")


with tab_notechange:
    st.subheader("Note-by-Note Change Between Any Two Emotions")
    cc1,cc2 = st.columns(2)
    with cc1:
        src_emo = st.selectbox("Source Emotion", list(EMOTION_RAGA_MAP.keys()),
                               key="nc_src", format_func=lambda e: f"{EMOJI[e]} {e}")
    with cc2:
        tgt_emo = st.selectbox("Target Emotion", list(EMOTION_RAGA_MAP.keys()),
                               index=1, key="nc_tgt", format_func=lambda e: f"{EMOJI[e]} {e}")
    if src_emo == tgt_emo:
        st.warning("Same emotion — no notes change.")
    else:
        df_ch = get_note_change_table(src_emo, tgt_emo)
        st.dataframe(df_ch, hide_index=True, use_container_width=True)
        a = (df_ch["Change Type"]=="Added").sum()
        r = (df_ch["Change Type"]=="Removed").sum()
        s = (df_ch["Change Type"]=="Kept").sum()
        st.markdown(f"+{a} added  |  {r} removed  |  {s} shared")
        st.download_button("Download CSV", df_ch.to_csv(index=False).encode(),
                           f"note_changes_{src_emo}_{tgt_emo}.csv")


with tab_allpairs:
    st.subheader("All Emotion Pairs Change Matrix — 8 × 8 = 64 Rows")
    fc1,fc2 = st.columns(2)
    with fc1:
        fsrc = st.selectbox("Filter Source", ["All"]+list(EMOTION_RAGA_MAP.keys()), key="ap_src")
    with fc2:
        ftgt = st.selectbox("Filter Target", ["All"]+list(EMOTION_RAGA_MAP.keys()), key="ap_tgt")
    df_mat = get_all_pairs_change_matrix()
    if fsrc != "All": df_mat = df_mat[df_mat["Source Emotion"]==fsrc]
    if ftgt != "All": df_mat = df_mat[df_mat["Target Emotion"]==ftgt]
    st.dataframe(df_mat, hide_index=True, use_container_width=True)
    st.markdown(f"Showing **{len(df_mat)}** pair(s)  |  **{len(df_mat.columns)}** columns")
    st.download_button("Download CSV", df_mat.to_csv(index=False).encode(), "all_emotion_pairs.csv")


with tab_ground:
    st.subheader("Ground Truth Comparison — Extracted vs Actual Swaras")
    st.caption("Compares algorithm-extracted notes against published notation for known songs.")

    sp  = st.session_state.swara_profile
    res = st.session_state.results

    # ── Section 1: Ground Truth vs Extracted ──────────────────────────────
    st.markdown("### Section 1 — Extracted Notes vs Ground Truth")
    uploaded_name = st.session_state.get("uploaded_name", None)

    if uploaded_name is None:
        st.info("Upload a song and run Analyse in the Convert Song tab first.")
    else:
        gt = lookup_song(uploaded_name)
        if gt is None:
            st.warning(f"No ground truth found for '{uploaded_name}'. Ground truth available for: Po Nee Po, Venmathiye, Manjal Veyil, Dhimu Dhimu, Aathadi, Kuchi Mittai, Thee Illai.")
        else:
            st.success(f"Ground truth found: {gt['song']} ({gt['film']}) — {gt['emotion']} / {gt['raga']}")

            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Song",    gt["song"])
            g2.metric("Emotion", gt["emotion"])
            g3.metric("Raga",    gt["raga"])
            g4.metric("Tonic",   gt["tonic"])

            st.markdown("**Actual Swara Sequence (Ground Truth — first 60 notes):**")
            seq = gt["sequence"]
            for rs in range(0, min(len(seq), 60), 15):
                st.markdown(" → ")

            st.markdown(f"**Arohanam:** {gt['arohanam']}  |  **Avarohanam:** {gt['avarohanam']}")

            if sp:
                st.divider()
                st.markdown("**Similarity: Extracted Profile vs Ground Truth**")
                sim = compute_similarity(sp["chroma_vector"], gt["semitone_profile"])

                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Cosine Similarity", f"{sim['cosine_similarity']:.4f}")
                sc2.metric("Precision",         f"{sim['precision']:.4f}")
                sc3.metric("Recall",            f"{sim['recall']:.4f}")
                sc4.metric("F1 Score",          f"{sim['f1_score']:.4f}")

                c1, c2, c3 = st.columns(3)
                c1.markdown("**Correctly Detected:**")
                for n in sim["correct_notes"]: c1.markdown(f"- ")
                c2.markdown("**Extra (not in ground truth):**")
                for n in sim["false_positives"]: c2.markdown(f"- ")
                c3.markdown("**Missed (in ground truth):**")
                for n in sim["false_negatives"]: c3.markdown(f"- ")

                st.markdown("**Per-Note Comparison:**")
                st.dataframe(
                    pd.DataFrame(sim["per_note"]).astype(str),
                    hide_index=True, use_container_width=True
                )

    st.divider()

    # ── Section 2: S1 vs S2 Comparison ────────────────────────────────────
    st.markdown("### Section 2 — S1 (Original) vs S2 (Converted) Analysis")

    if not res:
        st.info("No conversions yet. Run a conversion first.")
    else:
        labels = [
            f"{i+1}. {r['original_name']} → {r['target_emotion']}"
            for i, r in enumerate(res)
        ]
        idx = st.selectbox("Select conversion:", range(len(labels)),
                           format_func=lambda i: labels[i],
                           index=len(labels)-1, key="gt_sel")
        r       = res[idx]
        report  = r["report"]
        log     = r["log"]

        s1_raga    = report.get("detected_raga",    "Unknown")
        s1_emotion = report.get("detected_emotion", "Unknown")
        s2_raga    = r["target_raga"]
        s2_emotion = r["target_emotion"]

        st.markdown(f"**S1:** {r['original_name']} — {s1_raga} ({s1_emotion})")
        st.markdown(f"**S2:** Converted — {s2_raga} ({s2_emotion})")

        # Get S1 chroma from swara profile if available
        s1_chroma = sp["chroma_vector"] if sp else report.get("_chroma", [0]*12)
        if hasattr(s1_chroma, "tolist"): s1_chroma = s1_chroma.tolist()

        # S2 chroma: use raga semitone profile as proxy
        from core.raga_knowledge_base import RAGA_SWARAS
        s2_info = RAGA_SWARAS.get(s2_raga, {})
        s2_semis = [int(s)%12 for s in s2_info.get("semitones",[]) if isinstance(s,(int,float))]
        s2_chroma = [0.0]*12
        for s in s2_semis: s2_chroma[s] = 1.0
        tot = sum(s2_chroma)+1e-9
        s2_chroma = [x/tot for x in s2_chroma]

        cmp = compare_s1_s2(s1_chroma, s2_chroma, s1_emotion, s2_emotion)

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("S1 Emotion",       s1_emotion)
        m2.metric("S2 Emotion",       s2_emotion)
        m3.metric("Cosine Distance",  f"{cmp['cosine_distance']:.4f}")
        m4.metric("Notes Changed",    len(cmp['notes_gained'])+len(cmp['notes_lost']))

nc1, nc2, nc3 = st.columns(3)

with nc1:
    st.markdown("**Notes Gained (S2 has, S1 didn't):**")
    for n in cmp["notes_gained"]:
        st.markdown(f"- `{n}`")
    if not cmp["notes_gained"]:
        st.markdown("- None")

with nc2:
    st.markdown("**Notes Lost (S1 had, S2 doesn't):**")
    for n in cmp["notes_lost"]:
        st.markdown(f"- `{n}`")
    if not cmp["notes_lost"]:
        st.markdown("- None")

with nc3:
    st.markdown("**Notes Shared:**")
    for n in cmp["notes_shared"]:
        st.markdown(f"- `{n}`")
    if not cmp["notes_shared"]:
        st.markdown("- None")        

        if cmp["note_changes"]:
            st.markdown("**Note Weight Changes (S1 → S2):**")
            st.dataframe(
                pd.DataFrame(cmp["note_changes"]).astype(str),
                hide_index=True, use_container_width=True
            )

        st.divider()
        st.markdown("### Section 3 — Full Song Features")
        features = {
            "File":             r["original_name"],
            "Duration (s)":     report.get("duration_sec", "–"),
            "Tempo (BPM)":      report.get("tempo_bpm", "–"),
            "Tonic Semitone":   report.get("tonic_semitone", "–"),
            "Dominant Swaras":  " · ".join(report.get("dominant_swaras", [])),
            "S1 Raga":          s1_raga,
            "S1 Emotion":       s1_emotion,
            "S1 Match Score":   report.get("match_score", "–"),
            "S2 Raga":          s2_raga,
            "S2 Emotion":       s2_emotion,
            "Pitch Shift (st)": log.get("pitch_shift_st", 0),
            "Tempo Factor":     log.get("tempo_factor", 1.0),
            "Separation Method":log.get("separation_method", "–"),
            "Ref Profiles Used":log.get("ref_profiles_used", 0),
            "Windows Analysed": report.get("windows_analysed", "–"),
            "Notes Removed":    ", ".join(log.get("notes_removed", {}).values()),
            "Notes Added":      ", ".join(log.get("notes_added", {}).values()),
            "Notes Shared":     ", ".join(log.get("notes_shared", {}).values()),
        }
        st.dataframe(
            pd.DataFrame(list(features.items()), columns=["Feature","Value"]).astype(str),
            hide_index=True, use_container_width=True
        )
        st.download_button(
            "Download Features CSV",
            pd.DataFrame(list(features.items()), columns=["Feature","Value"]).to_csv(index=False).encode(),
            f"features_{r['original_name']}.csv"
        )


with tab_ref:
    st.subheader("Reference Song Library")
    ra,rb = st.columns([2,1])
    with ra:
        ref_file = st.file_uploader("Upload reference song", type=["mp3","wav"], key="ref_up")
    with rb:
        ref_emotion = st.selectbox("Emotion", list(EMOTION_RAGA_MAP.keys()), key="ref_emo")
        ref_name    = st.text_input("Song name", key="ref_name")
    if st.button("Save to Library") and ref_file and ref_name:
        suffix = ".mp3" if ref_file.name.endswith(".mp3") else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(ref_file.read())
            tmp_path = tmp.name
        with st.spinner("Extracting pitch profile..."):
            import librosa as _lr
            y_ref, sr_ref = _lr.load(tmp_path, mono=True, sr=16000)
            save_reference(y_ref, sr_ref, ref_name, ref_emotion)
        os.unlink(tmp_path)
        st.success(f"Saved: {ref_name} ({ref_emotion})")
    st.divider()
    st.subheader("Saved References")
    all_refs = list_all_references()
    if not all_refs:
        st.info("No reference songs saved yet.")
    else:
        for emotion, songs in sorted(all_refs.items()):
            with st.expander(f"{emotion} — {len(songs)} song(s)"):
                for sname in songs:
                    c1,c2 = st.columns([4,1])
                    c1.write(sname)
                    with c2:
                        if st.button("Delete", key=f"del_{emotion}_{sname}"):
                            delete_reference(emotion, sname)
                            st.rerun()

with tab_math:
    st.subheader("Mathematical Foundations & Acoustic Feature Tables")
    st.caption("Raw formulae, DSP parameters, and Chromagram matrices.")
    try:
        with open("raga_mathematical_acoustics_report.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())
    except Exception as e:
        st.warning("Could not load the mathematical report file.")

st.divider()
st.caption("AI Raga Voice Changer · MovieRagaNet · Tamil Song Emotion Converter · Final Year Project")