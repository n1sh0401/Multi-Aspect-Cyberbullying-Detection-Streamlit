"""Streamlit helper utilities to integrate the HuggingFaceShapExplainer.

Usage (in your Streamlit app):

from explainers.streamlit_helpers import load_background_cached, run_explain_and_render

# In the top-level Streamlit application
background_texts, background_numeric, numeric_names = load_background_cached("data/background.csv", text_col="text", numeric_cols=["peerness","aggressiveness","repetition"])
explainer = HuggingFaceShapExplainer(model_id_or_path=model_id, background_texts=background_texts, background_numeric=background_numeric, numeric_feature_names=numeric_names)

# In response to a user input
run_explain_and_render(explainer, text=user_text, numeric=[0.2,0.6,0.1], label_map={0:"Not Cyberbullying", 1:"Cyberbullying"})

"""

from typing import Optional, List
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from .shap_explainer import HuggingFaceShapExplainer


@st.cache_data
def load_background_cached(path: str, text_col: str = "text", numeric_cols: Optional[List[str]] = None, max_rows: int = 200):
    """Load background CSV and cache it for Streamlit. Returns (texts, numeric_array, numeric_names)."""
    texts, numeric_arr, numeric_names = HuggingFaceShapExplainer.load_background_from_csv(path, text_col=text_col, numeric_cols=numeric_cols, max_rows=max_rows)
    return texts, numeric_arr, numeric_names


def run_explain_and_render(explainer: HuggingFaceShapExplainer, text: str, numeric: Optional[List[float]] = None, label_map: Optional[dict] = None):
    """Run explanation and render outputs inside Streamlit UI.

    - Prints prediction class & probability
    - Shows SHAP bar plot (top contributors)
    - Shows a DataFrame with token/feature contributions
    - Shows token-level explanation text (if available)
    """
    res = explainer.explain_instance(text, numeric=numeric, top_k=25)

    # Show plot (combined contributors)
    fig = res.get("figure")
    if fig is not None:
        st.pyplot(fig)
    else:
        st.write("No feature contributions available to plot.")

    # Short explanation of how to interpret SHAP values
    st.markdown(
        "**Interpreting SHAP values:** Positive SHAP values (green bars) *increase* the model's predicted probability for the shown class; negative SHAP values (red bars) *decrease* it. The length (absolute value) indicates the magnitude of influence — larger absolute values mean a stronger effect. Tokens show which words push the prediction; numeric features show contextual influence. Use these together to understand *why* the model made its prediction."
    )

    # Show consolidated table (tokens + numeric features)
    df = res.get("breakdown")
    if df is not None and not df.empty:
        display_df = df[["feature", "type", "shap_value"]].copy()
        display_df["shap_value"] = display_df["shap_value"].round(6)
        st.subheader("Feature contributions")
        st.dataframe(display_df.reset_index(drop=True))

    return res
