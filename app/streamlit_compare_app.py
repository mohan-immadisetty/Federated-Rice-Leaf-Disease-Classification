"""Rice leaf disease recognition demo: predicted disease name from the final model and the baseline FedAvg model.

Shows only the predicted class names (no confidence values).

Run:  D:\\rice\\rice_handoff\\.venv\\Scripts\\python.exe -m streamlit run D:\\rice\\app\\streamlit_compare_app.py
"""
import json
import random
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from e5_infer import E5Model, load_manifest, load_audit, md5_bytes # noqa: E402


TITLE = "Robust Federated Learning for Rice Leaf Disease Recognition under non-IID Client Distributions"

st.set_page_config(page_title=TITLE, page_icon="🌾", layout="wide")


def pretty(c):
    return c.replace("_", " ").capitalize()


@st.cache_resource(show_spinner="Loading models…")
def get_model():
    return E5Model()


@st.cache_data(show_spinner=False)
def get_test_set():
    """Official held-out test split: image paths, and md5 -> true class."""

    man = load_manifest()

    # Read excluded images from the existing audit file
    leak = set(
        load_audit().get("eval_images_excluded_by_reaudit", [])
    )

    test = man[
        (man.split == "test") &
        ~man.image_path.isin(leak)
    ]

    return test.image_path.tolist(), dict(
        zip(test.md5, test.canonical_class)
    )


def predict_cached(model, raw, md5):
    store = st.session_state.setdefault("preds", {})
    if md5 not in store:
        out = model.predict(raw)
        store[md5] = (int(out["e5_prob"].argmax()), int(out["fedavg_prob"].argmax()))
    return store[md5]


model = get_model()
CLASSES = model.classes
TEST_PATHS, TEST_LABELS = get_test_set()
st.session_state.setdefault("score", {})        # md5 -> (true, final_pred, fedavg_pred)

st.title(TITLE)
st.write("Upload rice leaf images, or pick a random image from the test split, to see the disease predicted by "
         "our final model and by the baseline FedAvg model.")

c1, c2 = st.columns([3, 1])
with c1:
    uploads = st.file_uploader("Leaf images", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True)
with c2:
    st.write("")
    st.write("")
    if st.button("🎲 Random test image", width="stretch", help="Picks a random image from the held-out test split"):
        st.session_state.random_pick = random.choice(TEST_PATHS)

st.subheader("Score card")
board = st.container(border=True)

items = [(u.name, u.getvalue()) for u in uploads or []]
if st.session_state.get("random_pick"):
    p = Path(st.session_state.random_pick)
    items.insert(0, (f"[random test image] {p.name}", p.read_bytes()))

if not items:
    st.info("Upload rice leaf images or click **Random test image** to start.")

for name, raw in items:
    md5 = md5_bytes(raw)
    final_idx, fedavg_idx = predict_cached(model, raw, md5)
    final_pred, fedavg_pred = CLASSES[final_idx], CLASSES[fedavg_idx]
    true = TEST_LABELS.get(md5)
    if true is not None:
        st.session_state.score[md5] = (true, final_pred, fedavg_pred)
    with st.container(border=True):
        left, right = st.columns([1, 2])
        with left:
            st.image(raw, caption=name, width="stretch")
        with right:
            if true is not None:
                st.markdown(f"True disease: **{pretty(true)}**")
            a, b = st.columns(2)
            with a:
                st.caption("Final model (ours)")
                st.subheader(pretty(final_pred) + ("" if true is None else (" ✅" if final_pred == true else " ❌")))
                if true is not None and final_pred != true:
                    st.error(f"Wrong. Correct disease: **{pretty(true)}**")
            with b:
                st.caption("Baseline FedAvg")
                st.subheader(pretty(fedavg_pred) + ("" if true is None else (" ✅" if fedavg_pred == true else " ❌")))
                if true is not None and fedavg_pred != true:
                    st.error(f"Wrong. Correct disease: **{pretty(true)}**")
            if true is None:
                st.caption("Not a test-split image, so it is not counted in the score card.")

# score card is filled last so it includes the images shown in this run
with board:
    sc = list(st.session_state.score.values())
    n = len(sc)
    m1, m2, m3, m4 = st.columns([2, 2, 2, 1])
    m1.metric("Test images checked", n)
    m2.metric("Final model (ours) correct", f"{sum(t == f for t, f, _ in sc)} / {n}")
    m3.metric("Baseline FedAvg correct", f"{sum(t == b for t, _, b in sc)} / {n}")
    with m4:
        st.write("")
        if st.button("Reset", width="stretch"):
            st.session_state.score = {}
            st.session_state.pop("random_pick", None)
            st.rerun()
    if n == 0:
        st.caption("Each test-split image you check is counted here.")
