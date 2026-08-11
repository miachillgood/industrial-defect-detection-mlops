"""Streamlit annotation-review tool for SPADE predictions.

A human-in-the-loop front end: the model proposes a verdict and a defect mask,
an inspector confirms or overrides it, and every decision is appended to a
review log that can be used to re-tune the decision threshold.

    streamlit run apps/streamlit/app.py

This tool is *not* part of the upstream SPADE reference implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import streamlit as st  # noqa: E402
from mlops.review_store import ReviewRecord, ReviewStore  # noqa: E402
from spade.inference import SpadePredictor  # noqa: E402
from spade.visualize import render_heatmap_overlay  # noqa: E402

BANK_DIR = REPO_ROOT / "artifacts" / "banks"
DATA_ROOT = REPO_ROOT / "data" / "mvtec_anomaly_detection"
REVIEW_PATH = REPO_ROOT / "artifacts" / "reviews" / "reviews.jsonl"

st.set_page_config(page_title="SPADE defect review", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading memory bank…")
def load_predictor(category: str, device: str) -> SpadePredictor:
    return SpadePredictor(BANK_DIR / f"spade_{category}.pt", device=device)


@st.cache_data(show_spinner=False)
def list_test_images(category: str) -> list[tuple[str, str]]:
    """(display label, path) for every test image of a category."""
    test_dir = DATA_ROOT / category / "test"
    if not test_dir.is_dir():
        return []
    items = []
    for defect_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        for img in sorted(defect_dir.glob("*.png")):
            items.append((f"{defect_dir.name}/{img.name}", str(img)))
    return items


def available_banks() -> list[str]:
    if not BANK_DIR.is_dir():
        return []
    return sorted(p.stem.removeprefix("spade_") for p in BANK_DIR.glob("spade_*.pt"))


def review_key(path: str) -> str:
    """Repo-relative key for the review log.

    Absolute paths would make the log useless on any other machine (and leak the
    reviewer's home directory into a file that gets shared).
    """
    if not path or path.startswith("upload://"):
        return path
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def ground_truth_for(path: str) -> str | None:
    parts = Path(path).parts
    if "test" not in parts:
        return None
    defect_type = Path(path).parent.name
    return "ok" if defect_type == "good" else "defect"


def gt_mask_for(path: str) -> Path | None:
    p = Path(path)
    if p.parent.name == "good" or "test" not in p.parts:
        return None
    category_root = p.parents[2]
    candidate = category_root / "ground_truth" / p.parent.name / f"{p.stem}_mask.png"
    return candidate if candidate.exists() else None


# ----------------------------------------------------------------- sidebar
st.sidebar.title("🔎 SPADE review")
st.sidebar.caption(
    "SPADE (Cohen & Hoshen, arXiv:2005.02357), configured after "
    "byungjae89/SPADE-pytorch. Unrelated to NVlabs/SPADE."
)

banks = available_banks()
if not banks:
    st.title("No memory bank found")
    st.warning(f"Expected `spade_<category>.pt` files under `{BANK_DIR}`.")
    st.code("python scripts/build_bank.py --category bottle", language="bash")
    st.caption("Or fetch a previously versioned bank with `dvc pull`.")
    st.stop()

category = st.sidebar.selectbox("Category", banks, index=0)
device = st.sidebar.selectbox("Device", ["auto", "cpu", "mps", "cuda"], index=0)
reviewer = st.sidebar.text_input("Reviewer", value="inspector-01")

predictor = load_predictor(category, device)
info = predictor.info()

st.sidebar.divider()
st.sidebar.metric("Training images in bank", info["n_train_images"])
st.sidebar.metric("K (neighbours)", info["top_k"])
default_img_th = float(info["image_threshold"])
default_pix_th = float(info["pixel_threshold"])

st.sidebar.divider()
st.sidebar.subheader("Thresholds")
st.sidebar.caption("Calibrated leave-one-out on the defect-free training split. Override to explore.")
image_threshold = st.sidebar.slider(
    "Image threshold", 0.0, round(default_img_th * 3, 2), default_img_th, step=0.01
)
pixel_threshold = st.sidebar.slider(
    "Pixel threshold", 0.0, round(default_pix_th * 3, 2), default_pix_th, step=0.01
)

store = ReviewStore(REVIEW_PATH)

tab_review, tab_dashboard = st.tabs(["Review queue", "Dashboard"])

# ------------------------------------------------------------------ review
with tab_review:
    source = st.radio("Image source", ["MVTec test split", "Upload"], horizontal=True)

    image: Image.Image | None = None
    image_path = ""
    truth: str | None = None

    if source == "MVTec test split":
        items = list_test_images(category)
        if not items:
            st.warning(f"No test images found under `{DATA_ROOT / category / 'test'}`.")
        else:
            reviewed = set(store.latest_per_image())
            hide_done = st.checkbox("Hide already-reviewed images", value=True)
            pool = [it for it in items if not (hide_done and review_key(it[1]) in reviewed)]
            if not pool:
                st.success("Every image in this category has been reviewed.")
            else:
                label = st.selectbox(f"Image ({len(pool)} pending of {len(items)})",
                                     [it[0] for it in pool])
                image_path = dict(pool)[label]
                image = Image.open(image_path)
                truth = ground_truth_for(image_path)
    else:
        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "tiff"])
        if uploaded is not None:
            image = Image.open(uploaded)
            image_path = f"upload://{uploaded.name}"

    if image is not None:
        with st.spinner("Scoring…"):
            result = predictor.predict(image)

        model_verdict = "defect" if result.image_score > image_threshold else "ok"
        rgb = predictor.preprocessed_rgb(image)
        overlay = render_heatmap_overlay(rgb, result.score_map)
        pred_mask = (result.score_map > pixel_threshold).astype(np.uint8) * 255

        c1, c2, c3, c4 = st.columns(4)
        c1.image(rgb, caption="Input (224×224 crop)", width="stretch")
        c2.image(overlay, caption="Anomaly heatmap", width="stretch")
        c3.image(pred_mask, caption=f"Predicted mask (>{pixel_threshold:.2f})",
                 width="stretch")
        gt_path = gt_mask_for(image_path) if image_path else None
        if gt_path:
            c4.image(Image.open(gt_path).resize(pred_mask.shape[::-1]),
                     caption="Ground-truth mask", width="stretch")
        else:
            c4.info("No ground-truth mask\n\n(defect-free sample or uploaded image)")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Image score", f"{result.image_score:.3f}")
        m2.metric("Threshold", f"{image_threshold:.3f}")
        m3.metric("Model verdict", "DEFECT" if model_verdict == "defect" else "OK")
        m4.metric("Anomalous pixels", f"{(result.score_map > pixel_threshold).mean() * 100:.2f} %")
        if truth:
            (st.success if truth == model_verdict else st.error)(
                f"Dataset label: **{truth.upper()}** — model {'agrees' if truth == model_verdict else 'disagrees'}"
            )

        st.divider()
        st.subheader("Your verdict")
        col_a, col_b = st.columns([2, 3])
        with col_a:
            human_verdict = st.radio("Verdict", ["ok", "defect", "unsure"],
                                     index=["ok", "defect", "unsure"].index(model_verdict),
                                     horizontal=True)
            defect_type = st.text_input("Defect type (optional)",
                                        value=Path(image_path).parent.name
                                        if image_path and "test" in image_path
                                        and Path(image_path).parent.name != "good" else "")
        with col_b:
            notes = st.text_area("Notes", height=110,
                                 placeholder="e.g. heatmap fires on the reflection, not the crack")

        if st.button("Save review", type="primary", width="stretch"):
            store.append(
                ReviewRecord(
                    image_path=review_key(image_path) or "unknown",
                    category=category,
                    human_verdict=human_verdict,
                    model_score=float(result.image_score),
                    model_threshold=float(image_threshold),
                    model_verdict=model_verdict,
                    reviewer=reviewer,
                    defect_type=defect_type or None,
                    notes=notes,
                    ground_truth=truth,
                )
            )
            st.success(f"Saved to {REVIEW_PATH.relative_to(REPO_ROOT)}")
            list_test_images.clear()
            st.rerun()

# --------------------------------------------------------------- dashboard
with tab_dashboard:
    stats = store.stats()
    if stats["n_reviewed"] == 0:
        st.info("No reviews recorded yet.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Reviewed", stats["n_reviewed"])
        c2.metric("Agreement with model", f"{stats['agreement_rate'] * 100:.1f} %")
        c3.metric("Precision (vs human)", f"{stats['precision'] * 100:.1f} %")
        c4.metric("Recall (vs human)", f"{stats['recall'] * 100:.1f} %")

        cm = stats["confusion"]
        st.subheader("Model vs human")
        st.table(
            {
                "": ["model: defect", "model: ok"],
                "human: defect": [cm["tp"], cm["fn"]],
                "human: ok": [cm["fp"], cm["tn"]],
            }
        )

        rows = list(store.latest_per_image().values())
        st.subheader("Review log")
        st.dataframe(rows, width="stretch", height=340)

        disagreements = [r for r in rows if r["human_verdict"] != r["model_verdict"]]
        if disagreements:
            st.subheader(f"Disagreements ({len(disagreements)}) — candidates for threshold re-tuning")
            st.dataframe(disagreements, width="stretch")

        if st.button("Export CSV"):
            out = store.export_csv(REPO_ROOT / "artifacts" / "reviews" / "reviews.csv")
            st.success(f"Wrote {out.relative_to(REPO_ROOT)}")
