"""Rice leaf disease checker — Streamlit demo of the final E5 model.

Run from the repository root:
    streamlit run app/streamlit_app.py

Upload a leaf image to see the E5 prediction, the probability table and the
FedAvg baseline. A random image from the held-out test split can be sampled
from assets/samples/ (19 classes, one image each).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e5_infer import (  # noqa: E402
    ASSETS,
    UNKNOWN,
    E5Model,
    load_manifest,
    md5_bytes,
    pretty,
)

SAMPLES = ASSETS / "samples"
SAMPLE_CSV = SAMPLES / "samples.csv"

st.set_page_config(page_title="Rice Leaf Disease Checker — E5", page_icon="🌾", layout="wide")


@st.cache_resource(show_spinner="Loading EfficientNetV2-S + DINOv2 + E5 head…")
def get_model() -> E5Model:
    return E5Model()


@st.cache_data(show_spinner=False)
def get_manifest() -> pd.DataFrame:
    return load_manifest()


@st.cache_data(show_spinner=False)
def get_samples() -> pd.DataFrame:
    if not SAMPLE_CSV.exists():
        return pd.DataFrame(columns=["file", "md5", "true_class", "source"])
    return pd.read_csv(SAMPLE_CSV, keep_default_na=False)


def predict_cached(model: E5Model, raw: bytes, md5: str) -> dict:
    store = st.session_state.setdefault("preds", {})
    if md5 not in store:
        store[md5] = model.predict(raw)
    return store[md5]


model = get_model()
man = get_manifest()
samples = get_samples()
CLASSES = model.classes
st.session_state.setdefault("score", {})  # md5 -> (true, e5_pred, fedavg_pred)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🌾 About the model")
    st.markdown(
        "**E5** (final system, backbone seed 42)\n\n"
        "- FedAvg EfficientNetV2-S features (1280-d)\n"
        "- + frozen DINOv2-base features (1536-d)\n"
        "- → federated linear head, 19 classes\n\n"
        "**Held-out test** (n = 2,793): 92.34% accuracy, 89.58 macro-F1.\n\n"
        "The FedAvg baseline (same backbone, its own classifier) is shown for "
        "comparison: 85.82% / 67.39."
    )
    show_fedavg = st.toggle("Show FedAvg baseline too", value=True)
    st.divider()
    st.subheader("Session scoreboard")
    board = st.container()
    if st.button("Reset scoreboard"):
        st.session_state.score = {}
        st.session_state.pop("random_pick", None)
        st.rerun()
    with st.expander("The 19 classes"):
        st.markdown("\n".join(f"- {pretty(c)}" for c in CLASSES))

# ---------------------------------------------------------------- input
st.title("Rice Leaf Disease Checker")
st.write(
    "Upload rice leaf images. The model predicts the disease. Give the true label, "
    "or let the app find it when the image is from the project dataset, and you'll "
    "see whether the prediction was right."
)

c1, c2 = st.columns([3, 1])
with c1:
    uploads = st.file_uploader(
        "Leaf images", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True
    )
with c2:
    st.write("")
    st.write("")
    if st.button(
        "🎲 Random test image",
        width="stretch",
        help="Picks one real image from the held-out test split",
        disabled=samples.empty,
    ):
        st.session_state.random_pick = samples.sample(1).iloc[0].file

items = [(u.name, u.getvalue()) for u in uploads or []]
if st.session_state.get("random_pick"):
    p = SAMPLES / st.session_state.random_pick
    items.insert(0, (f"[random test] {p.name}", p.read_bytes()))

if not items:
    st.info("Upload an image or click **Random test image** to start.")

# ---------------------------------------------------------------- results
for name, raw in items:
    md5 = md5_bytes(raw)
    row = man[man.md5 == md5]
    row = None if row.empty else row.iloc[0]
    with st.container(border=True):
        left, right = st.columns([1, 2])
        with left:
            st.image(raw, caption=name, width="stretch")
        with right:
            if row is not None:
                true = row.canonical_class
                st.caption(
                    f"Found in the dataset: **{row.source}**, split **{row.split}**. "
                    "True label taken from the manifest."
                )
                if row.split == "train":
                    st.warning("This image was in the **training** set, so the model has seen it. It is not a fair test.")
                elif row.split == "excluded":
                    st.warning(f"This image was excluded from the split ({row.exclusion_reason or 'duplicate'}).")
                elif row.split == "test":
                    st.caption("Official held-out test image.")
            else:
                choice = st.selectbox(
                    "True label of this image",
                    [UNKNOWN] + [pretty(c) for c in CLASSES],
                    key=f"lbl_{md5}",
                )
                true = None if choice == UNKNOWN else CLASSES[[pretty(c) for c in CLASSES].index(choice)]
                st.caption("Not found in the project dataset (exact file match), so this is a new image.")

            out = predict_cached(model, raw, md5)
            pe, pf = out["e5_prob"], out["fedavg_prob"]
            e5_pred, fa_pred = CLASSES[pe.argmax()], CLASSES[pf.argmax()]

            st.markdown(f"### E5 predicts: **{pretty(e5_pred)}** ({100 * pe.max():.1f}%)")
            if true is not None:
                if e5_pred == true:
                    st.success(f"✅ Correct. True label: **{pretty(true)}**")
                else:
                    st.error(
                        f"❌ Wrong. True label: **{pretty(true)}** "
                        f"(model gave it {100 * pe[CLASSES.index(true)]:.1f}%)"
                    )
                st.session_state.score[md5] = (true, e5_pred, fa_pred)
            else:
                st.session_state.score.pop(md5, None)
                st.info("Choose the true label above to check the prediction.")

            order = pe.argsort()[::-1][:5]
            df = pd.DataFrame({"class": [pretty(CLASSES[k]) for k in order], "probability": pe[order]})
            st.dataframe(
                df,
                hide_index=True,
                width="stretch",
                column_config={
                    "probability": st.column_config.ProgressColumn(
                        "probability", format="percent", min_value=0, max_value=1
                    )
                },
            )
            if show_fedavg:
                mark = "" if true is None else (" ✅" if fa_pred == true else " ❌")
                st.caption(f"FedAvg baseline: {pretty(fa_pred)} ({100 * pf.max():.1f}%){mark}")

# ---------------------------------------------------------------- scoreboard
with board:
    sc = [v for v in st.session_state.score.values() if v[0] is not None]
    if sc:
        e5_ok = sum(t == p for t, p, _ in sc)
        st.metric("E5 correct", f"{e5_ok} / {len(sc)}", f"{100 * e5_ok / len(sc):.1f}%")
        if show_fedavg:
            fa_ok = sum(t == f for t, _, f in sc)
            st.metric("FedAvg correct", f"{fa_ok} / {len(sc)}", f"{100 * fa_ok / len(sc):.1f}%")
    else:
        st.caption("Images with a known true label are counted here.")
