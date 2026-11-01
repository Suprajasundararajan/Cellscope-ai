import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from src.common import CLASSES, OUT, get_device
from src.data import build_tf
from src import ensemble as ensemble_utils
from src.gradcam import GradCAM
from src.metrics import softmax
from src.models import build_model, cam_layer


st.set_page_config(
    page_title="CellScope AI | WBC Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / OUT
CHECKPOINTS = OUTPUTS / "ckpt"
RESULTS = OUTPUTS / "results"
DEVICE = get_device()
MODEL_NAMES = ["mobilenetv2", "efficientnetb0", "resnet50"]
DISPLAY_NAMES = {
    "mobilenetv2": "MobileNetV2",
    "efficientnetb0": "EfficientNet-B0",
    "resnet50": "ResNet50",
}
ENSEMBLE_WEIGHTS = {
    "mobilenetv2": 0.6,
    "efficientnetb0": 0.3,
    "resnet50": 0.1,
}
MIN_CONFIDENCE = 0.60
MAX_ENTROPY = 1.20
MAX_DISAGREEMENT = 0.08
OOD_GREEN_DOMINANCE = 0.10
OOD_MIN_COLOR_VARIATION = 0.015

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    :root { --green: #10b981; --blue: #38bdf8; --panel: rgba(15, 23, 42, .78); }
    html, body, [class*="css"] { font-family: Inter, sans-serif; }
    .hero {
        padding: 26px 30px; margin-bottom: 22px; border-radius: 18px;
        background: linear-gradient(135deg, rgba(16,185,129,.20), rgba(14,116,144,.28));
        border: 1px solid rgba(56,189,248,.30); box-shadow: 0 12px 35px rgba(0,0,0,.22);
    }
    .hero h1 { margin: 0; font-size: 2.35rem; }
    .hero p { margin: 8px 0 0; color: #cbd5e1; }
    .result-card {
        padding: 22px; border-radius: 16px; background: var(--panel);
        border: 1px solid rgba(16,185,129,.32); margin-bottom: 16px;
    }
    .prediction { font-size: 2rem; font-weight: 800; color: #34d399; }
    .muted { color: #94a3b8; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    models = {}
    missing = []
    for name in MODEL_NAMES:
        path = CHECKPOINTS / f"{name}_std_s0.pt"
        if not path.exists():
            missing.append(str(path))
            continue
        model = build_model(name, pretrained=False).to(DEVICE)
        state = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        models[name] = model
    weights = {name: ENSEMBLE_WEIGHTS[name] for name in models}
    logit_files = [OUTPUTS / "logits" / f"{name}_std_s0.npz" for name in models]
    if len(models) > 1 and all(path.exists() for path in logit_files):
        arrays = [np.load(path) for path in logit_files]
        val_probs = np.array([softmax(data["val_logits"]) for data in arrays])
        tuned = ensemble_utils.search_weights(
            val_probs, arrays[0]["val_y"], step=0.1
        )
        weights = dict(zip(models, map(float, tuned)))
    return models, missing, weights


def predict(image, models):
    image_array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    red, green, blue = image_array[..., 0], image_array[..., 1], image_array[..., 2]
    green_dominance = float(np.mean(green - np.maximum(red, blue)))
    color_variation = float(np.std(image_array, axis=2).mean())
    likely_non_wbc = (
        green_dominance > OOD_GREEN_DOMINANCE
        or color_variation < OOD_MIN_COLOR_VARIATION
    )
    tensor = build_tf(False)(image).unsqueeze(0).to(DEVICE)
    probabilities = {}
    started = time.perf_counter()
    with torch.inference_mode():
        for name, model in models.items():
            probabilities[name] = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
    elapsed_ms = (time.perf_counter() - started) * 1000

    member_probs = np.vstack(list(probabilities.values()))
    mean_probs = member_probs.mean(axis=0)
    weighted_probs = np.average(
        member_probs,
        axis=0,
        weights=[ENSEMBLE_WEIGHTS.get(name, 1.0) for name in probabilities],
    )
    predicted_index = int(weighted_probs.argmax())
    entropy = float(-(weighted_probs * np.log(weighted_probs + 1e-12)).sum())
    member_entropy = -(member_probs * np.log(member_probs + 1e-12)).sum(axis=1)
    mean_entropy = float(-(mean_probs * np.log(mean_probs + 1e-12)).sum())
    disagreement = float(mean_entropy - member_entropy.mean())  # mutual information, as in src/analyze.py
    return {
        "probabilities": weighted_probs,
        "mean_probabilities": mean_probs,
        "member_probabilities": probabilities,
        "predicted_index": predicted_index,
        "predicted_class": CLASSES[predicted_index],
        "confidence": float(weighted_probs[predicted_index]),
        "entropy": entropy,
        "disagreement": disagreement,
        "latency_ms": elapsed_ms,
        "input": tensor,
        "likely_non_wbc": likely_non_wbc,
        "green_dominance": green_dominance,
        "color_variation": color_variation,
    }


def create_gradcam(image, model_name, models):
    if model_name not in models:
        return None
    tensor = build_tf(False)(image).unsqueeze(0).to(DEVICE)
    model = models[model_name]
    cam = GradCAM(model, cam_layer(model, model_name))
    try:
        with torch.enable_grad():
            heatmap, _ = cam(tensor)
        heatmap = heatmap[0]
    finally:
        cam.close()

    base = np.asarray(image.convert("RGB").resize((224, 224))) / 255.0
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(base)
    ax.imshow(heatmap, cmap="jet", alpha=0.45)
    ax.axis("off")
    fig.tight_layout(pad=0)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def available_samples():
    candidates = []
    for class_name in CLASSES:
        folder = ROOT / "Train" / class_name
        if folder.exists():
            candidates.extend(sorted(folder.glob("*.jpg"))[:2])
            candidates.extend(sorted(folder.glob("*.jpeg"))[:2])
            candidates.extend(sorted(folder.glob("*.png"))[:2])
    return candidates


models, missing_models, tuned_weights = load_models()
ENSEMBLE_WEIGHTS.update(tuned_weights)

with st.sidebar:
    st.markdown("## ⚙️ CellScope Engine")
    selected_model = st.selectbox(
        "Inference model",
        ["Weighted Ensemble"] + [DISPLAY_NAMES[n] for n in models],
    )
    cam_model = st.selectbox(
        "Grad-CAM model",
        [DISPLAY_NAMES[n] for n in models],
    )
    st.caption(f"Device: `{DEVICE}`")
    st.caption("Ensemble weights are tuned on validation predictions.")
    st.caption("Research and educational use only; not a diagnostic device.")

st.markdown(
    """
    <div class="hero">
        <h1>🔬 CellScope AI</h1>
        <p>Explainable white-blood-cell type identification with ensemble inference.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if missing_models:
    st.warning("Some checkpoints are missing. Train the corresponding models first.")
    for path in missing_models:
        st.code(path)
if not models:
    st.error("No trained checkpoints were found in outputs/ckpt.")
    st.stop()

tab_scan, tab_xai, tab_metrics = st.tabs(
    ["🔬 WBC Classifier", "🧠 Explainability", "📊 Model Results"]
)

with tab_scan:
    source = st.radio(
        "Image source",
        ["Upload image", "Camera", "Demo sample"],
        horizontal=True,
    )
    image = None
    if source == "Upload image":
        uploaded = st.file_uploader(
            "Upload a cropped white-blood-cell image",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        )
        if uploaded:
            try:
                image = Image.open(uploaded).convert("RGB")
            except UnidentifiedImageError:
                st.error("The uploaded file is not a readable image.")
    elif source == "Camera":
        captured = st.camera_input("Capture a WBC image")
        if captured:
            try:
                image = Image.open(captured).convert("RGB")
            except UnidentifiedImageError:
                st.error("The captured file is not a readable image.")
    else:
        samples = available_samples()
        if samples:
            selected_sample = st.selectbox("Choose a training sample", samples, format_func=lambda p: p.name)
            image = Image.open(selected_sample).convert("RGB")
        else:
            st.info("No demo samples were found in the Train folder.")

    if image is not None:
        with st.spinner("Analyzing cell morphology..."):
            result = predict(image, models)

        if selected_model != "Weighted Ensemble":
            model_name = next(
                name for name, label in DISPLAY_NAMES.items() if label == selected_model
            )
            probs = result["member_probabilities"][model_name]
            pred_index = int(probs.argmax())
            prediction = CLASSES[pred_index]
            confidence = float(probs[pred_index])
        else:
            probs = result["probabilities"]
            prediction = result["predicted_class"]
            confidence = result["confidence"]
        top_indices = np.argsort(probs)[::-1][:3]
        uncertain = (
            confidence < MIN_CONFIDENCE
            or result["entropy"] > MAX_ENTROPY
            or result["disagreement"] > MAX_DISAGREEMENT
        )
        not_wbc = result["likely_non_wbc"]

        left, right = st.columns([1, 1.25], gap="large")
        with left:
            st.image(image, caption=f"Input image: {image.width} × {image.height}px", use_container_width=True)
        with right:
            prediction_label = "Not a WBC" if not_wbc else prediction
            confidence_label = (
                "Image appears outside the WBC microscopy domain"
                if not_wbc
                else f"Confidence: {confidence:.1%}"
            )
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="muted">Screening result</div>
                    <div class="prediction">{prediction_label}</div>
                    <div class="muted">{confidence_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("Inference", f"{result['latency_ms']:.1f} ms")
            m2.metric("Entropy", f"{result['entropy']:.3f}")
            m3.metric("Disagreement", f"{result['disagreement']:.4f}")
        if not_wbc:
            st.error(
                "Not a WBC image. Upload a cropped white-blood-cell microscopy image "
                "for cell-type classification."
            )
        st.markdown("### Model agreement")
        agreement_rows = []
        for model_name, model_probs in result["member_probabilities"].items():
            model_index = int(model_probs.argmax())
            agreement_rows.append({
                "Model": DISPLAY_NAMES[model_name],
                "Prediction": CLASSES[model_index],
                "Confidence": float(model_probs[model_index]),
            })
        st.dataframe(
            pd.DataFrame(agreement_rows).style.format({"Confidence": "{:.2%}"}),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Class probability breakdown")
        probability_df = pd.DataFrame(
            {
                "Rank": range(1, len(top_indices) + 1),
                "WBC type": [CLASSES[i] for i in top_indices],
                "Probability": [float(probs[i]) for i in top_indices],
            }
        )
        st.bar_chart(probability_df.set_index("WBC type"))
        st.dataframe(
            probability_df.style.format({"Probability": "{:.2%}"}),
            use_container_width=True,
            hide_index=True,
        )
        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "prediction": "Not a WBC" if not_wbc else prediction,
            "confidence": confidence,
            "uncertain": uncertain,
            "out_of_domain": not_wbc,
            "green_dominance": result["green_dominance"],
            "color_variation": result["color_variation"],
            "entropy": result["entropy"],
            "model_disagreement": result["disagreement"],
            "latency_ms": result["latency_ms"],
            "class_probabilities": {
                class_name: float(probs[index])
                for index, class_name in enumerate(CLASSES)
            },
            "model_predictions": agreement_rows,
            "device": str(DEVICE),
        }
        st.download_button(
            "Download prediction report",
            data=json.dumps(report, indent=2),
            file_name=f"cellscope_report_{'not_wbc' if not_wbc else prediction.lower()}_{int(time.time())}.json",
            mime="application/json",
        )

with tab_xai:
    st.markdown("### Grad-CAM explanation")
    st.write("The heatmap shows image regions that influenced the selected model's prediction.")
    if "image" in locals() and image is not None:
        if result.get("likely_non_wbc", False):
            st.info("Grad-CAM is not generated because this image was screened as not a WBC.")
        else:
            model_name = next(
                name for name, label in DISPLAY_NAMES.items() if label == cam_model
            )
            with st.spinner("Computing Grad-CAM..."):
                overlay = create_gradcam(image, model_name, models)
            if overlay is not None:
                col1, col2 = st.columns(2)
                col1.image(image, caption="Original image", use_container_width=True)
                col2.image(
                    overlay,
                    caption=f"{cam_model} Grad-CAM overlay",
                    use_container_width=True,
                )
            else:
                st.warning("The selected model checkpoint is unavailable.")
    else:
        st.info("Choose an image in the WBC Classifier tab first.")

with tab_metrics:
    st.markdown("### Saved evaluation results")
    result_files = {
        "summary.csv": "Summary metrics",
        "domain_gap.csv": "Cross-domain macro-F1 gap",
        "ci_and_mcnemar.csv": "Confidence intervals and McNemar tests",
        "selective_accuracy.csv": "Selective accuracy and coverage",
        "per_class_metrics.csv": "Per-class recall",
        "confusion_matrices.csv": "Confusion matrices",
    }
    shown = 0
    for filename, title in result_files.items():
        result_path = RESULTS / filename
        if not result_path.exists():
            continue
        shown += 1
        st.markdown(f"#### {title}")
        csv_data = result_path.read_bytes()
        st.dataframe(pd.read_csv(io.BytesIO(csv_data)), use_container_width=True)
        st.download_button(
            f"Download {filename}",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            key=f"download-{filename}",
        )
    if shown == 0:
        st.info("Run `python -m src.analyze --tags std --seeds 0` to generate result files.")

    report = {
        "prediction_app": "CellScope AI",
        "device": str(DEVICE),
        "available_models": [DISPLAY_NAMES[n] for n in models],
        "classes": CLASSES,
    }
    st.download_button(
        "Download app configuration",
        data=json.dumps(report, indent=2),
        file_name="cellscope_app_info.json",
        mime="application/json",
    )