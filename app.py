"""
app.py
------
Browser-based dashboard for the AI-Enhanced Intrusion Detection System.

A simple frontend to demo the trained model: pull a random traffic record
(from the test set) or type in your own values, classify it, and see the
result instantly. Also shows overall model performance.

Run from the project root:
    streamlit run app.py
"""

import os
import sys
import joblib
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from constants import COLUMN_NAMES, CATEGORICAL_COLUMNS, ATTACK_CATEGORIES  # noqa: E402
from preprocessing import add_attack_category, load_raw  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FEATURE_COLUMN_NAMES = COLUMN_NAMES[:-2]  # all 41 features (no label/difficulty)

st.set_page_config(
    page_title="AI-Enhanced Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    with open(os.path.join(MODELS_DIR, "best_model.txt")) as f:
        model_name = f.read().strip()
    return model, scaler, encoders, feature_cols, model_name


@st.cache_data
def load_test_data():
    df = add_attack_category(load_raw(os.path.join(DATA_DIR, "KDDTest+.txt")))
    return df


def classify(row_df: pd.DataFrame):
    model, scaler, encoders, feature_cols, _ = load_artifacts()
    row = row_df.copy()
    for col in CATEGORICAL_COLUMNS:
        row[col] = encoders[col].transform(row[col].astype(str))
    X = scaler.transform(row[feature_cols])
    pred = model.predict(X)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        probs = model.predict_proba(X)[0]
        proba = dict(zip(classes, probs))
    return pred, proba


def badge(label: str):
    color = "#1f9d55" if label == "normal" else "#d64545"
    st.markdown(
        f"<div style='display:inline-block;padding:10px 22px;border-radius:8px;"
        f"background:{color};color:white;font-size:22px;font-weight:700;'>"
        f"{label.upper()}</div>",
        unsafe_allow_html=True,
    )


st.title("🛡️ AI-Enhanced Intrusion Detection System")
st.caption("Machine-learning based network intrusion classification — NSL-KDD dataset")

model, scaler, encoders, feature_cols, model_name = load_artifacts()
test_df = load_test_data()

tab_live, tab_manual, tab_performance = st.tabs(
    ["🔴 Live Traffic Test", "✍️ Manual Input", "📊 Model Performance"]
)

# ---------------------------------------------------------------- Live Test
with tab_live:
    st.subheader("Classify a random traffic record from the test set")
    st.write(
        "Click the button to pull a random, never-before-seen network "
        "traffic record and see how the model classifies it."
    )

    if "sample_idx" not in st.session_state:
        st.session_state.sample_idx = None

    if st.button("🎲 Get random traffic record", type="primary"):
        st.session_state.sample_idx = test_df.sample(1).index[0]

    if st.session_state.sample_idx is not None:
        row = test_df.loc[[st.session_state.sample_idx]]
        actual_label = row["attack_category"].values[0]

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Traffic record:**")
            display_cols = ["protocol_type", "service", "flag", "duration",
                             "src_bytes", "dst_bytes", "count", "srv_count"]
            st.table(row[display_cols].T.rename(columns={row.index[0]: "value"}))

        with col2:
            pred, proba = classify(row[feature_cols])
            st.markdown("**Model prediction:**")
            badge(pred)
            st.write("")
            st.markdown(f"**Actual label (ground truth):** `{actual_label}`")
            if pred == actual_label:
                st.success("✅ Correct classification")
            else:
                st.error("❌ Misclassified")

            if proba:
                st.markdown("**Confidence by category:**")
                proba_df = pd.DataFrame(
                    {"category": list(proba.keys()), "confidence": list(proba.values())}
                ).sort_values("confidence", ascending=False)
                st.bar_chart(proba_df.set_index("category"))

# ------------------------------------------------------------- Manual Input
with tab_manual:
    st.subheader("Type in your own traffic values")
    st.write("Adjust the key fields below and classify a custom record.")

    defaults = test_df.iloc[0]
    c1, c2, c3 = st.columns(3)
    with c1:
        protocol_type = st.selectbox("protocol_type", ["tcp", "udp", "icmp"])
        service = st.selectbox(
            "service", sorted(test_df["service"].unique()),
            index=sorted(test_df["service"].unique()).index(defaults["service"])
            if defaults["service"] in test_df["service"].unique() else 0,
        )
        flag = st.selectbox("flag", sorted(test_df["flag"].unique()))
    with c2:
        duration = st.number_input("duration", min_value=0, value=0)
        src_bytes = st.number_input("src_bytes", min_value=0, value=200)
        dst_bytes = st.number_input("dst_bytes", min_value=0, value=0)
    with c3:
        count = st.number_input("count", min_value=0, value=1)
        srv_count = st.number_input("srv_count", min_value=0, value=1)
        logged_in = st.selectbox("logged_in", [0, 1])

    if st.button("🔍 Classify this record", type="primary"):
        custom = defaults.copy()
        custom["protocol_type"] = protocol_type
        custom["service"] = service
        custom["flag"] = flag
        custom["duration"] = duration
        custom["src_bytes"] = src_bytes
        custom["dst_bytes"] = dst_bytes
        custom["count"] = count
        custom["srv_count"] = srv_count
        custom["logged_in"] = logged_in

        row_df = pd.DataFrame([custom])[feature_cols]
        pred, proba = classify(row_df)
        st.markdown("**Model prediction:**")
        badge(pred)
        if proba:
            proba_df = pd.DataFrame(
                {"category": list(proba.keys()), "confidence": list(proba.values())}
            ).sort_values("confidence", ascending=False)
            st.bar_chart(proba_df.set_index("category"))

# ---------------------------------------------------------- Model Performance
with tab_performance:
    st.subheader(f"Test-set performance — model: `{model_name}`")

    report_path = os.path.join(REPORTS_DIR, "evaluation_report.md")
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")

    if os.path.exists(report_path):
        with open(report_path) as f:
            st.markdown(f.read())
    else:
        st.warning("Run `python src/evaluate.py` first to generate the report.")

    if os.path.exists(cm_path):
        st.image(cm_path, caption="Confusion Matrix", width=600)

    results_path = os.path.join(MODELS_DIR, "training_results.csv")
    if os.path.exists(results_path):
        st.markdown("**Model comparison (cross-validated during training):**")
        st.dataframe(pd.read_csv(results_path))