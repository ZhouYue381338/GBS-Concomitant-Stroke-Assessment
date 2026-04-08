from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import joblib
import pandas as pd
import sklearn
import streamlit as st
import streamlit.components.v1 as components
from sklearn.exceptions import InconsistentVersionWarning

try:
    import shap
except Exception:
    shap = None


st.set_page_config(
    page_title="GBS Concomitant Stroke Assessment",
    page_icon="AI",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ann_model.pkl"

APP_TITLE = "Identification System for Concomitant Stroke in Guillain-Barré Syndrome"
POSITIVE_CLASS = 1

FEATURE_SPECS: dict[str, dict[str, Any]] = {
    "Complicated_with_hypertension": {
        "type": "binary",
        "label": "History of Hypertension",
        "help": "1 = presence, 0 = absence",
        "default": 0,
    },
    "Complicated_with_hyperlipidemia": {
        "type": "binary",
        "label": "History of Hyperlipidemia",
        "help": "1 = presence, 0 = absence",
        "default": 0,
    },
    "Age": {
        "type": "numeric",
        "label": "Age",
        "unit": "years",
        "default": 55.0,
        "min": 0.0,
        "max": 120.0,
        "step": 1.0,
    },
    "RBC": {
        "type": "numeric",
        "label": "Red Blood Cell Count (RBC)",
        "unit": "x10^12/L",
        "default": 4.50,
        "min": 0.0,
        "max": 10.0,
        "step": 0.01,
    },
    "eGFR": {
        "type": "numeric",
        "label": "Estimated Glomerular Filtration Rate (eGFR)",
        "unit": "mL/min/1.73 m^2",
        "default": 90.0,
        "min": 0.0,
        "max": 250.0,
        "step": 1.0,
    },
    "APTT": {
        "type": "numeric",
        "label": "Activated Partial Thromboplastin Time (APTT)",
        "unit": "s",
        "default": 30.0,
        "min": 0.0,
        "max": 200.0,
        "step": 0.1,
    },
    "FT4": {
        "type": "numeric",
        "label": "Free Thyroxine (FT4)",
        "unit": "pmol/L",
        "default": 14.0,
        "min": 0.0,
        "max": 100.0,
        "step": 0.1,
    },
    "GLU": {
        "type": "numeric",
        "label": "Fasting Plasma Glucose (GLU)",
        "unit": "mmol/L",
        "default": 5.6,
        "min": 0.0,
        "max": 50.0,
        "step": 0.1,
    },
    "HbA1c": {
        "type": "numeric",
        "label": "Glycated Hemoglobin A1c (HbA1c)",
        "unit": "%",
        "default": 5.7,
        "min": 0.0,
        "max": 20.0,
        "step": 0.1,
    },
}

FEATURE_ORDER = list(FEATURE_SPECS.keys())


class _DummyPickleState:
    def __setstate__(self, state: Any) -> None:
        self.state = state


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #153247;
            --muted: #607789;
            --panel: rgba(255, 255, 255, 0.92);
            --line: rgba(21, 50, 71, 0.10);
            --accent: #0f7d69;
            --accent-soft: rgba(15, 125, 105, 0.14);
            --danger: #b22b49;
            --danger-soft: rgba(178, 43, 73, 0.12);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 125, 105, 0.16), transparent 28%),
                radial-gradient(circle at top right, rgba(19, 50, 71, 0.12), transparent 24%),
                linear-gradient(180deg, #f7fafb 0%, #eef3f6 100%);
            color: var(--ink);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero-card, .metric-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 22px;
            box-shadow: 0 18px 44px rgba(20, 53, 80, 0.08);
            backdrop-filter: blur(8px);
        }
        .hero-card {
            padding: 28px 30px;
            margin-bottom: 18px;
        }
        .metric-card {
            padding: 20px 22px;
            min-height: 160px;
        }
        .hero-title {
            color: var(--ink);
            font-size: 2.3rem;
            line-height: 1.08;
            margin: 0;
            font-family: Georgia, "Times New Roman", serif;
            font-weight: 700;
        }
        .section-title {
            color: var(--ink);
            font-size: 1.06rem;
            font-weight: 800;
            margin-bottom: 0.75rem;
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.45rem;
        }
        .metric-value {
            color: var(--ink);
            font-size: 2.1rem;
            line-height: 1;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }
        .metric-note {
            color: var(--muted);
            font-size: 0.94rem;
            line-height: 1.58;
        }
        .outcome-chip {
            display: inline-block;
            padding: 7px 12px;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 800;
            margin-bottom: 12px;
        }
        .chip-positive {
            background: var(--danger-soft);
            color: var(--danger);
        }
        .chip-negative {
            background: var(--accent-soft);
            color: var(--accent);
        }
        .shap-shell {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 14px 16px 6px 16px;
            margin-top: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            model = joblib.load(MODEL_PATH)
    except Exception as original_exc:
        try:
            import numpy.random._pickle as np_pickle

            original_bit_generator_ctor = np_pickle.__bit_generator_ctor
            original_randomstate_ctor = np_pickle.__randomstate_ctor

            np_pickle.__bit_generator_ctor = (
                lambda bit_generator_name="MT19937": _DummyPickleState()
            )
            np_pickle.__randomstate_ctor = (
                lambda bit_generator_name="MT19937": _DummyPickleState()
            )

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", InconsistentVersionWarning)
                    model = joblib.load(MODEL_PATH)
                setattr(
                    model,
                    "_loader_note",
                    "The model was loaded with a compatibility fallback for the saved random "
                    "state object. Predictions are available, but aligning the runtime to "
                    "scikit-learn 1.7.2 remains the preferred deployment option.",
                )
            finally:
                np_pickle.__bit_generator_ctor = original_bit_generator_ctor
                np_pickle.__randomstate_ctor = original_randomstate_ctor
        except Exception as fallback_exc:
            raise RuntimeError(
                "Unable to load ann_model.pkl. This model appears to require the exact "
                "project dependencies for inference. Current environment: "
                f"scikit-learn {sklearn.__version__}. Original error: {original_exc}. "
                f"Fallback error: {fallback_exc}"
            ) from fallback_exc

    required_attrs = ["predict", "predict_proba", "classes_"]
    missing_attrs = [attr for attr in required_attrs if not hasattr(model, attr)]
    if missing_attrs:
        raise TypeError(f"The loaded object is missing required attributes: {missing_attrs}")

    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
        unknown_features = [name for name in feature_names if name not in FEATURE_SPECS]
        if unknown_features:
            raise KeyError(f"Model contains unsupported features: {unknown_features}")

    return model


def get_model_feature_order(model: Any) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return FEATURE_ORDER


def get_default_background_frame(model: Any) -> pd.DataFrame:
    ordered_features = get_model_feature_order(model)
    base_row = {name: FEATURE_SPECS[name]["default"] for name in ordered_features}
    background_rows = [base_row.copy()]

    if "Complicated_with_hypertension" in base_row:
        row = base_row.copy()
        row["Complicated_with_hypertension"] = 1
        background_rows.append(row)

    if "Complicated_with_hyperlipidemia" in base_row:
        row = base_row.copy()
        row["Complicated_with_hyperlipidemia"] = 1
        background_rows.append(row)

    if {
        "Complicated_with_hypertension",
        "Complicated_with_hyperlipidemia",
    }.issubset(base_row):
        row = base_row.copy()
        row["Complicated_with_hypertension"] = 1
        row["Complicated_with_hyperlipidemia"] = 1
        background_rows.append(row)

    return pd.DataFrame(background_rows, columns=ordered_features)


@st.cache_resource
def build_shap_explainer() -> Any:
    if shap is None:
        return None

    model = load_model()
    background_df = get_default_background_frame(model)
    return shap.KernelExplainer(model.predict_proba, background_df, link="logit")


def build_input_form(model: Any) -> tuple[bool, dict[str, Any]]:
    user_inputs: dict[str, Any] = {}
    ordered_features = get_model_feature_order(model)

    st.markdown("### Patient Inputs")

    with st.form("gbs_prediction_form", clear_on_submit=False):
        columns = st.columns(2)

        for idx, feature_name in enumerate(ordered_features):
            spec = FEATURE_SPECS[feature_name]
            with columns[idx % 2]:
                if spec["type"] == "binary":
                    option_map = {
                        "Absent ": 0,
                        "Present ": 1,
                    }
                    default_label = "Present " if spec["default"] == 1 else "Absent "
                    selected_label = st.selectbox(
                        spec["label"],
                        list(option_map.keys()),
                        index=list(option_map.keys()).index(default_label),
                        help=spec["help"],
                        key=f"field_{feature_name}",
                    )
                    user_inputs[feature_name] = option_map[selected_label]
                else:
                    label = spec["label"]
                    if spec.get("unit"):
                        label = f"{label} ({spec['unit']})"
                    user_inputs[feature_name] = st.number_input(
                        label,
                        min_value=float(spec["min"]),
                        max_value=float(spec["max"]),
                        value=float(spec["default"]),
                        step=float(spec["step"]),
                        key=f"field_{feature_name}",
                    )

        submitted = st.form_submit_button("Assessment", use_container_width=True)

    return submitted, user_inputs


def build_input_dataframe(model: Any, user_inputs: dict[str, Any]) -> pd.DataFrame:
    ordered_features = get_model_feature_order(model)
    row = [[user_inputs[name] for name in ordered_features]]
    return pd.DataFrame(row, columns=ordered_features)


def predict_case(model: Any, input_df: pd.DataFrame) -> tuple[dict[Any, float], Any]:
    probabilities = model.predict_proba(input_df)[0]
    classes = list(model.classes_)
    probability_map = {label: float(prob) for label, prob in zip(classes, probabilities)}
    predicted_class = model.predict(input_df)[0]
    return probability_map, predicted_class


def render_header() -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">{APP_TITLE}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_prediction(probability_map: dict[Any, float], predicted_class: Any) -> None:
    positive_probability = probability_map.get(POSITIVE_CLASS, 0.0)
    negative_probability = probability_map.get(0, 0.0)
    is_positive = predicted_class == POSITIVE_CLASS

    outcome_label = (
        "Assessment result: Concomitant stroke likely"
        if is_positive
        else "Assessment result: Concomitant stroke unlikely"
    )
    chip_class = "chip-positive" if is_positive else "chip-negative"
    interpretation = (
        "Based on the admission-time clinical profile, the patient is more likely to have concomitant stroke during the index clinical episode."
        if is_positive
        else "Based on the admission-time clinical profile, the patient is less likely to have concomitant stroke during the index clinical episode."
    )

    st.markdown("### Assessment")
    col1, col2 = st.columns([1.25, 1])

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="outcome-chip {chip_class}">{outcome_label}</div>
                <div class="metric-value">{positive_probability:.1%}</div>
                <div class="metric-label">Probability of Concomitant Stroke Presence</div>
                <div class="metric-note">{interpretation}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Probability of Concomitant Stroke Absence</div>
                <div class="metric-value">{negative_probability:.1%}</div>
                <div class="metric-note">
                    This value represents the model-estimated probability that concomitant stroke is absent.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def render_shap_force_plot(input_df: pd.DataFrame) -> None:
    if shap is None:
        return

    try:
        explainer = build_shap_explainer()
        shap_values = explainer.shap_values(input_df, nsamples=100)
        expected_value = explainer.expected_value

        if isinstance(shap_values, list):
            class_index = len(shap_values) - 1
            shap_row = shap_values[class_index][0]
        else:
            class_index = 1 if getattr(shap_values, "ndim", 0) == 3 else None
            if class_index is None:
                shap_row = shap_values[0]
            else:
                shap_row = shap_values[0, :, class_index]

        if hasattr(expected_value, "__len__") and not isinstance(expected_value, str):
            expected_value = expected_value[-1]

        force_plot = shap.force_plot(
            float(expected_value),
            shap_row,
            input_df.iloc[0],
            feature_names=list(input_df.columns),
            matplotlib=False,
            show=False,
        )
        html = (
            "<div class='shap-shell'>"
            f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
            "</div>"
        )
        st.markdown("### SHAP Force Plot")
        components.html(html, height=260, scrolling=True)
    except Exception:
        return


def main() -> None:
    inject_styles()

    try:
        model = load_model()
    except Exception as exc:
        st.error(f"Startup failed: {exc}")
        st.code("pip install -r requirements.txt", language="bash")
        return

    render_header()

    submitted, user_inputs = build_input_form(model)
    if not submitted:
        return

    input_df = build_input_dataframe(model, user_inputs)

    try:
        probability_map, predicted_class = predict_case(model, input_df)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    render_prediction(probability_map, predicted_class)
    render_shap_force_plot(input_df)


if __name__ == "__main__":
    main()
