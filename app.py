
import pickle
import cv2
import numpy as np
import streamlit as st

from pathlib import Path
from huggingface_hub import hf_hub_download
from skimage.feature import hog, graycomatrix, graycoprops


# Hugging Face model repository
MODEL_REPO_ID = "syedtasneem/renal-ct-classification-project"


# Page setup
st.set_page_config(
    page_title="Concordant Intelligence for Renal Imaging",
    page_icon="🩺",
    layout="wide"
)


# ------------------------------------------------
# CUSTOM STYLING
# ------------------------------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f4f8ff 0%, #eefaf7 100%);
    color: #172b4d;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1150px;
}

.hero {
    background: linear-gradient(120deg, #102a43, #087e8b);
    padding: 34px;
    border-radius: 22px;
    margin-bottom: 25px;
    box-shadow: 0 8px 24px rgba(16, 42, 67, 0.15);
}

.hero h1 {
    color: white !important;
    font-size: 2.25rem;
    margin-bottom: 10px;
    line-height: 1.3;
}

.hero p {
    color: #d8f3f0 !important;
    font-size: 1.08rem;
    margin-bottom: 0;
}

.section-heading {
    color: #123b53;
    font-size: 1.55rem;
    font-weight: 700;
    margin-top: 28px;
    margin-bottom: 16px;
}

div[data-testid="stFileUploader"] {
    background: white;
    border: 2px dashed #78b9bd;
    border-radius: 16px;
    padding: 16px;
}

div[data-testid="stFileUploader"] section {
    background: #f5fbfb;
    border-radius: 12px;
}

div[data-testid="stMetric"] {
    background: white;
    padding: 23px 20px;
    border-radius: 18px;
    border: 1px solid #dce9ef;
    box-shadow: 0 5px 16px rgba(25, 65, 85, 0.07);
}

div[data-testid="stMetricLabel"] {
    color: #527184;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #087e8b;
    font-weight: 750;
}

div[data-testid="stImage"] img {
    border-radius: 16px;
}

.reliability-high {
    background: linear-gradient(120deg, #e1f8ed, #f3fff8);
    border-left: 7px solid #15966b;
    border-radius: 16px;
    padding: 22px;
    color: #116149;
    margin-top: 10px;
}

.reliability-low {
    background: linear-gradient(120deg, #fff0df, #fff9f0);
    border-left: 7px solid #d97706;
    border-radius: 16px;
    padding: 22px;
    color: #8a4b08;
    margin-top: 10px;
}

div.stButton > button {
    background: linear-gradient(100deg, #087e8b, #14a38b);
    color: white;
    border: none;
    border-radius: 10px;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------
# LOAD ALL THREE MODELS FROM HUGGING FACE
# ------------------------------------------------

@st.cache_resource(show_spinner="Downloading trained models...")
def load_models():

    # Original SVM
    svm_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename="svm_hog_glcm_model.pkl",
        repo_type="model"
    )

    # Random Forest
    rf_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename="random_forest_hog_glcm_model.pkl",
        repo_type="model"
    )

    # Calibrated SVM for probability estimation
    probability_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename="svm_probability.pkl",
        repo_type="model"
    )

    # Load original SVM
    with open(svm_path, "rb") as file:
        svm_model = pickle.load(file)

    # Load Random Forest
    with open(rf_path, "rb") as file:
        rf_model = pickle.load(file)

    # Load calibrated SVM
    with open(probability_path, "rb") as file:
        svm_probability_model = pickle.load(file)

    return svm_model, rf_model, svm_probability_model


# ------------------------------------------------
# IMAGE FEATURE EXTRACTION
# ------------------------------------------------

def extract_features(image_bytes):

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)

    image = cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError("Unable to read this image.")

    image = cv2.resize(image, (128, 128))

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    normalized = gray.astype(np.float32) / 255.0

    # HOG features
    hog_features = hog(
        normalized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    # GLCM features
    gray_uint8 = (normalized * 255).astype(np.uint8)

    glcm = graycomatrix(
        gray_uint8,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    glcm_features = np.array([
        graycoprops(glcm, "contrast")[0, 0],
        graycoprops(glcm, "homogeneity")[0, 0],
        graycoprops(glcm, "energy")[0, 0],
        graycoprops(glcm, "correlation")[0, 0]
    ], dtype=np.float64)

    # Feature fusion
    features = np.concatenate(
        [hog_features, glcm_features]
    ).reshape(1, -1)

    preview = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return features, preview


# ------------------------------------------------
# HEADER
# ------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>🩺 Concordant Intelligence for Renal Imaging</h1>
    <p>A Multi-Model Framework for Abnormality Classification</p>
</div>
""", unsafe_allow_html=True)

st.write(
    "Upload a kidney CT image to view its predicted category."
)


# ------------------------------------------------
# LOAD TRAINED MODELS
# ------------------------------------------------

try:

    svm_model, rf_model, svm_probability_model = load_models()

except Exception as error:

    st.error(
        "Unable to load the trained models from Hugging Face. "
        "Please check the repository and model filenames."
    )

    st.exception(error)

    st.stop()


# ------------------------------------------------
# IMAGE UPLOAD
# ------------------------------------------------

uploaded_image = st.file_uploader(
    "📤 Upload a kidney CT image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
)


if uploaded_image is not None:

    try:

        features, preview = extract_features(
            uploaded_image.getvalue()
        )

        # Display uploaded image
        st.markdown(
            '<div class="section-heading">🖼️ Uploaded Image</div>',
            unsafe_allow_html=True
        )

        st.image(
            preview,
            caption="Uploaded CT Image",
            width=360
        )

        # ------------------------------------------------
        # CHECK FEATURE COMPATIBILITY
        # ------------------------------------------------

        expected_features = getattr(
            svm_model,
            "n_features_in_",
            None
        )

        if (
            expected_features is not None
            and features.shape[1] != expected_features
        ):

            st.error(
                "The image features do not match the trained model. "
                "Please check the model and image-processing settings."
            )

            st.stop()

        # ------------------------------------------------
        # PREDICTIONS
        # ------------------------------------------------

        # Original SVM prediction
        svm_prediction = svm_model.predict(features)[0]

        # Random Forest prediction
        rf_prediction = rf_model.predict(features)[0]

        # ------------------------------------------------
        # CONFIDENCE SCORE CALCULATION
        # ------------------------------------------------

        # Calibrated SVM probability estimate
        svm_probabilities = svm_probability_model.predict_proba(
            features
        )[0]

        svm_confidence = (
            np.max(svm_probabilities) * 100
        )

        # Random Forest probability estimate
        rf_confidence = None

        if hasattr(rf_model, "predict_proba"):

            rf_probabilities = rf_model.predict_proba(
                features
            )[0]

            rf_confidence = (
                np.max(rf_probabilities) * 100
            )

        # ------------------------------------------------
        # PREDICTION RESULTS
        # ------------------------------------------------

        st.markdown(
            '<div class="section-heading">📊 Prediction Results</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                label="🔹 SVM Prediction",
                value=str(svm_prediction)
            )

        with col2:

            st.metric(
                label="🔸 Random Forest Prediction",
                value=str(rf_prediction)
            )

        # ------------------------------------------------
        # CONFIDENCE SCORE DISPLAY
        # ------------------------------------------------

        st.markdown(
            '<div class="section-heading">📈 Confidence Scores</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                label="SVM Confidence Estimate",
                value=f"{svm_confidence:.2f}%"
            )

        with col2:

            if rf_confidence is not None:

                st.metric(
                    label="Random Forest Confidence Estimate",
                    value=f"{rf_confidence:.2f}%"
                )

            else:

                st.metric(
                    label="Random Forest Confidence Estimate",
                    value="Unavailable"
                )

        st.info(
            "SVM confidence is estimated using a separately calibrated "
            "SVM model, while the SVM prediction is produced by the "
            "original SVM model. Their predictions may differ. "
            "Random Forest confidence is based on its predicted class "
            "probability. These estimates do not represent medical certainty."
        )

        # ------------------------------------------------
        # RELIABILITY INDICATOR
        # ------------------------------------------------

        st.markdown(
            '<div class="section-heading">✨ Reliability Indicator</div>',
            unsafe_allow_html=True
        )

        if svm_prediction == rf_prediction:

            st.markdown(f"""
            <div class="reliability-high">
                <h3>✓ HIGH — Models Agree</h3>
                <p>Both models predict:
                <strong>{svm_prediction}</strong></p>
                <p>The models agree on the predicted category.
                Agreement does not guarantee that the prediction is correct.</p>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="reliability-low">
                <h3>! LOW — Models Disagree</h3>
                <p>SVM predicts: <strong>{svm_prediction}</strong></p>
                <p>Random Forest predicts:
                <strong>{rf_prediction}</strong></p>
                <p>The result requires further review.</p>
            </div>
            """, unsafe_allow_html=True)

    except Exception as error:

        st.error(
            "Prediction failed. Please try another image and "
            "check that the trained models are compatible."
        )

        st.exception(error)


# ------------------------------------------------
# FOOTER
# ------------------------------------------------

st.markdown("""
<br>
<hr style="border: 1px solid #dce9ef;">
<p style="text-align:center; color:#668394; font-size:0.9rem;">
    Concordant Intelligence for Renal Imaging
</p>
""", unsafe_allow_html=True)
