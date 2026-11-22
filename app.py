from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

st.set_page_config(
    page_title="Bug Report Priority Classifier",
    page_icon="🐞",
    layout="wide"
)


def clean_text(value):
    if pd.isna(value):
        return ""

    text = str(value).lower()
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " urltoken ",
        text
    )
    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource
def load_artifacts():
    model_path = (
        MODEL_DIR / "self_training_random_forest.joblib"
    )
    vectorizer_path = (
        MODEL_DIR / "tfidf_vectorizer.joblib"
    )
    metadata_path = MODEL_DIR / "metadata.json"

    for path in [model_path, vectorizer_path, metadata_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file missing: {path.name}"
            )

    # Load only your own trusted model files.
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    return model, vectorizer, metadata


def predict_reports(frame, model, vectorizer):
    required = {"Title", "Description"}

    if not required.issubset(frame.columns):
        raise ValueError(
            "CSV must contain Title and Description columns."
        )

    if frame.empty:
        raise ValueError("CSV contains no reports.")

    if len(frame) > 5000:
        raise ValueError(
            "Please upload at most 5,000 reports at a time."
        )

    output = frame.copy().reset_index(drop=True)

    titles = output["Title"].fillna("").astype(str)
    descriptions = (
        output["Description"].fillna("").astype(str)
    )

    texts = (
        titles.map(clean_text)
        + " "
        + descriptions.map(clean_text)
    ).str.strip()

    if texts.eq("").any():
        raise ValueError(
            "Every row needs a title or description."
        )

    if texts.str.len().max() > 100000:
        raise ValueError(
            "A report is too long. Limit each report "
            "to 100,000 characters."
        )

    features = vectorizer.transform(texts)
    probabilities = model.predict_proba(features)

    positions = probabilities.argmax(axis=1)
    predictions = model.classes_[positions]
    confidence = probabilities.max(axis=1)

    recognized = (
        np.asarray(features.getnnz(axis=1)).ravel() > 0
    )

    # Avoid presenting a prediction when no vocabulary is known.
    output["Predicted_Priority_Code"] = pd.Series(
        predictions.astype(int),
        dtype="Int64"
    ).mask(~recognized)

    output["Model_Confidence"] = np.where(
        recognized,
        confidence,
        np.nan
    )

    output["Recognized_Vocabulary"] = recognized

    output["Needs_Review"] = (
        ~recognized | (confidence < 0.60)
    )

    for column, code in enumerate(model.classes_):
        output[f"Score_Code_{int(code)}"] = np.where(
            recognized,
            probabilities[:, column],
            np.nan
        )

    return output


st.title("🐞 Bug Report Priority Classifier")
st.caption(
    "Sain Bux | Section B | "
    "Semi-Supervised Self-Training + Random Forest"
)

st.info(
    "This app suggests dataset priority codes from bug-report "
    "text. Predictions need human review. Codes 0–4 are kept "
    "in their original form; urgency names are not verified."
)

try:
    model, vectorizer, metadata = load_artifacts()
except Exception as exc:
    st.error(f"Could not load model files: {exc}")
    st.write(
        "Check that the models folder contains the trained "
        "model, TF-IDF vectorizer and metadata.json."
    )
    st.stop()


with st.sidebar:
    st.header("Project details")

    st.write("**Learning:** Semi-Supervised")
    st.write("**Algorithm:** Self-Training + Random Forest")
    st.write("**Input:** Title and Description")

    st.divider()

    st.write(
        "**Training reports:**",
        metadata.get("training_reports", "Not available")
    )
    st.write(
        "**Seed labels:**",
        metadata.get("seed_labels", "Not available")
    )
    st.write(
        "**Initially unlabeled:**",
        metadata.get("initially_unlabeled", "Not available")
    )
    st.write(
        "**Accepted pseudo-labels:**",
        metadata.get("accepted_pseudo_labels", "Not available")
    )
    st.write(
        "**Self-training threshold:**",
        metadata.get("selected_threshold", "Not available")
    )

    st.caption(
        "The self-training threshold was used during training. "
        "It is different from the app's low-confidence review flag."
    )


predict_tab, batch_tab, results_tab, about_tab = st.tabs([
    "Predict a Bug",
    "Batch CSV",
    "Training Results",
    "How It Works"
])


with predict_tab:
    st.subheader("Enter a bug report")

    with st.form("single_prediction"):
        title = st.text_input(
            "Bug title",
            placeholder="Application crashes when opening a tab"
        )

        description = st.text_area(
            "Bug description",
            placeholder=(
                "Explain what happened, steps to reproduce, "
                "and expected behaviour."
            ),
            height=160
        )

        submitted = st.form_submit_button(
            "Predict Priority",
            type="primary"
        )

    if submitted:
        try:
            frame = pd.DataFrame([{
                "Title": title,
                "Description": description
            }])

            prediction = predict_reports(
                frame,
                model,
                vectorizer
            )

            row = prediction.iloc[0]

            if not bool(row["Recognized_Vocabulary"]):
                st.warning(
                    "The report has no recognized vocabulary. "
                    "Please use a meaningful bug description "
                    "and request manual triage."
                )
            else:
                code = int(row["Predicted_Priority_Code"])
                confidence = float(row["Model_Confidence"])

                left, right = st.columns(2)

                left.metric(
                    "Predicted Priority Code",
                    str(code)
                )
                right.metric(
                    "Model Confidence",
                    f"{confidence:.1%}"
                )

                if bool(row["Needs_Review"]):
                    st.warning(
                        "Low model confidence: manual review "
                        "is especially important."
                    )

                st.caption(
                    "Confidence is the model's class score, "
                    "not measured prediction accuracy or a "
                    "guarantee that the priority is correct."
                )

                scores = pd.DataFrame({
                    "Priority Code": [
                        str(int(c)) for c in model.classes_
                    ],
                    "Model Score": [
                        float(row[f"Score_Code_{int(c)}"])
                        for c in model.classes_
                    ]
                }).set_index("Priority Code")

                st.bar_chart(scores)

        except ValueError as exc:
            st.error(str(exc))


with batch_tab:
    st.subheader("Classify multiple reports")

    st.write(
        "Upload a CSV containing **Title** and "
        "**Description** columns. Your Kaggle test.csv "
        "has these columns."
    )

    template = pd.DataFrame([{
        "Title": "Application crashes",
        "Description": (
            "The application closes when opening a new tab."
        )
    }])

    st.download_button(
        "Download CSV Template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="bug_report_template.csv",
        mime="text/csv"
    )

    uploaded = st.file_uploader(
        "Upload bug reports CSV",
        type=["csv"]
    )

    if st.button("Classify CSV", type="primary"):
        if uploaded is None:
            st.warning("Please upload a CSV first.")
        elif uploaded.size > 10 * 1024 * 1024:
            st.error("Please upload a CSV smaller than 10 MB.")
        else:
            try:
                frame = pd.read_csv(uploaded)

                with st.spinner("Classifying reports..."):
                    output = predict_reports(
                        frame,
                        model,
                        vectorizer
                    )

                st.session_state["batch_output"] = output

            except (
                ValueError,
                UnicodeDecodeError,
                pd.errors.ParserError,
                pd.errors.EmptyDataError
            ) as exc:
                st.error(str(exc))

    if "batch_output" in st.session_state:
        output = st.session_state["batch_output"]

        st.caption(
            "Results from the last successfully classified upload."
        )

        st.dataframe(
            output,
            hide_index=True,
            use_container_width="stretch"
        )

        st.download_button(
            "Download Predictions",
            data=output.to_csv(index=False).encode("utf-8"),
            file_name="bug_priority_predictions.csv",
            mime="text/csv"
        )


with results_tab:
    st.subheader("Saved notebook results")

    st.warning(
        "The active self-training experiment did not outperform "
        "the labeled-only baseline. The evaluation portion was "
        "inspected during development, so the updated comparison "
        "is exploratory rather than an untouched final benchmark."
    )

    comparison_path = (
        RESULT_DIR / "final_model_comparison.csv"
    )

    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)

        st.dataframe(
            comparison,
            hide_index=True,
            use_container_width="stretch"
        )

        st.caption(
            "Scores range from 0 to 1. Macro F1 gives equal "
            "importance to each priority class."
        )

        required_metrics = {
            "method", "accuracy", "macro_f1"
        }

        if required_metrics.issubset(comparison.columns):
            st.bar_chart(
                comparison.set_index("method")[
                    ["accuracy", "macro_f1"]
                ]
            )
    else:
        st.info(
            "Add results/final_model_comparison.csv "
            "to display the saved evaluation."
        )

    for filename, caption in [
        (
            "self_training_history.png",
            "Pseudo-labels accepted in each iteration"
        ),
        (
            "confusion_matrices.png",
            "Confusion matrices: actual versus predicted codes"
        ),
        (
            "priority_distribution.png",
            "Dataset priority distribution"
        )
    ]:
        image_path = RESULT_DIR / filename

        if image_path.exists():
            st.image(str(image_path), caption=caption)


with about_tab:
    st.subheader("Project workflow")

    st.markdown("""
1. Combine each report's **Title** and **Description**.
2. Clean text and convert it into **TF-IDF features**.
3. Train a **Random Forest** using the small labeled seed set.
4. Predict labels for the unlabeled training reports.
5. Accept predictions above the self-training threshold.
6. Retrain using seed labels and accepted pseudo-labels.
7. Use the saved model to classify new reports.
""")

    st.subheader("Dataset and experiment")

    st.write(
        "Source: Kaggle ISEC Data Challenge. The original "
        "training data contain labels. This project simulates "
        "limited annotation by exposing only 20% of labels "
        "within the training pool."
    )

    st.write(
        "Training, validation and evaluation reports were "
        "separated. TF-IDF vocabulary was fitted only on "
        "the training pool. Exact duplicate report texts "
        "were removed before splitting."
    )

    st.subheader("Limitations")

    st.markdown("""
- The model has limited predictive performance.
- Incorrect pseudo-labels can reinforce earlier mistakes.
- Priority depends on business context missing from the text.
- Near-duplicate reports may remain in the dataset.
- Model confidence is not calibrated correctness.
- Human review is required before deciding actual priority.
""")

    with st.expander("Saved training metadata"):
        st.json(metadata)