import streamlit as st
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from transformers import AutoTokenizer, AutoConfig, BertPreTrainedModel, BertModel

# SHAP explainer utilities
from explainers.shap_explainer import HuggingFaceShapExplainer
from explainers.streamlit_helpers import run_explain_and_render
import os

# Authenticate with HuggingFace Hub
from huggingface_hub import login
login(token="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", add_to_git_credential=False)

# Model identifier used across the app
MODEL_ID = "rngrye/BERT-cyberbullying-classifier-FocalLoss"

# --- Define FocalLoss Class ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        if self.ignore_index >= 0:
            valid_indices = (targets != self.ignore_index)
            targets = targets[valid_indices]
            inputs = inputs[valid_indices]
            if targets.numel() == 0:
                return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# --- Define BertForMultiModalSequenceClassification ---
class BertForMultiModalSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)

        self.additional_features_dim = getattr(config, "additional_features_dim", 3)

        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size + self.additional_features_dim, config.num_labels)

        dropout_prob = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.dropout = nn.Dropout(dropout_prob)

        self.alpha = getattr(config, 'focal_loss_alpha', 0.25)
        self.gamma = getattr(config, 'focal_loss_gamma', 1.0)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        additional_features=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_state = bert_output[0]
        pooled_output = hidden_state[:, 0]

        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)

        if additional_features is not None:
            if isinstance(additional_features, list):
                additional_features = torch.tensor(additional_features, dtype=pooled_output.dtype, device=pooled_output.device)
            combined_features = torch.cat((pooled_output, additional_features), dim=1)
        else:
            dummy_features = torch.zeros((pooled_output.size(0), self.additional_features_dim), device=pooled_output.device)
            combined_features = torch.cat((pooled_output, dummy_features), dim=1)

        logits = self.classifier(combined_features)

        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(logits=logits)

st.set_page_config(page_title="Cyberbullying Detection Platform", layout="wide")

# Convert BG_Pewdipie1 image to base64 for use in CSS
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

bg_image_base64 = get_base64_image("BG_Pewdipie2.jpg")

# Apply a simple mostly-white / light-grey theme for a clean polished look
st.markdown(
    f"""
    <style>
    :root {{ --bg: #ffffff; --sidebar-bg: #e6e7eb; --muted: #6b7280; --text-color: #000000; }}

    /* App background with BG_Feathers image */
    [data-testid="stAppViewContainer"] {{ 
        background-image: url('data:image/jpeg;base64,{bg_image_base64}');
        background-attachment: fixed;
        background-size: cover;
        background-repeat: no-repeat;
        color: var(--text-color);
    }}
    [data-testid="stAppViewContainer"]::before {{
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(255, 255, 255, 0.5);
        pointer-events: none;
        z-index: 0;
    }}
    [data-testid="stAppViewContainer"] * {{ color: var(--text-color) !important; position: relative; z-index: 1; }}

    /* Sidebar background and text */
    [data-testid="stSidebar"] > div {{ background-color: var(--sidebar-bg); color: var(--text-color); }}
    [data-testid="stSidebar"] > div * {{ color: var(--text-color) !important; }}

    /* Header / top bar: set dark background but white text for contrast */
    [data-testid="stHeader"] {{ background-color: #111827 !important; }}
    [data-testid="stHeader"] * {{ color: #ffffff !important; }}
    [data-testid="stHeader"] code {{ color: #ffffff !important; background-color: transparent !important; }}

    /* File uploader: ensure light background and readable text */
    [data-testid="stFileUploader"] {{ background-color: #ffffff !important; color: var(--text-color) !important; border: 1px dashed #e5e7eb; padding: 0.9rem; border-radius: 8px; box-shadow: 0 1px 2px rgba(16,24,40,0.03); }}
    [data-testid="stFileUploader"] * {{ color: var(--text-color) !important; }}
    [data-testid="stFileUploader"] code {{ color: var(--text-color) !important; background-color: transparent !important; }}

    /* Text area: make input area white and readable */
    [data-testid="stTextArea"] {{ background-color: #ffffff !important; padding: 0.5rem; border-radius: 8px; }}
    [data-testid="stTextArea"] textarea {{ background-color: #ffffff !important; color: #000000 !important; }}

    /* Inline code styling: light background with dark text for readability on white surfaces */
    code {{ background-color: #f3f4f6; color: #000000; padding: 2px 6px; border-radius: 6px; }}

    /* But when inside the header (dark), make code white and transparent background */
    [data-testid="stHeader"] code {{ color: #ffffff !important; background-color: transparent !important; }}

    .stButton>button {{ background-color: #ffffff; border: 1px solid #e5e7eb; color: var(--text-color); }}
    .stTextInput>div>input, .stTextArea>div>textarea {{ border: 1px solid #e5e7eb; color: var(--text-color); }}
    .css-1kyxreq {{ padding: 0.6rem 0.8rem; }} /* small adjustments for layout */

    /* Keep links noticeable */
    a {{ color: #0b5fff !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    # Use the global MODEL_ID constant so it can be shared with the SHAP explainer
    # Add your HuggingFace token here for private model access
    hf_token = "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"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)

    config = AutoConfig.from_pretrained(MODEL_ID, token=hf_token)
    # Ensure config matches the fine-tuned model architecture
    config.additional_features_dim = getattr(config, "additional_features_dim", 3)
    config.focal_loss_alpha = getattr(config, "focal_loss_alpha", 0.25)
    config.focal_loss_gamma = getattr(config, "focal_loss_gamma", 1.0)

    model = BertForMultiModalSequenceClassification.from_pretrained(
        MODEL_ID,
        config=config,
        token=hf_token
    )

    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
model.eval()

# --- Normalization Stats (MUST MATCH TRAINING) ---
# These stats were computed from the training data with log transformation
# Update these based on your actual training data stats
NORMALIZATION_STATS = {
    'time_min': 0.0,
    'time_max': 17.4127,
    'peer_min': 0.0,
    'peer_max': 0.9999,
    'agg_min': 0.0,
    'agg_max': 1.0,
}

def normalize_features(repetition_raw, peerness_raw, aggressiveness_raw):
    """
    Normalize features to match training preprocessing.
    repetition_raw: raw value in seconds (e.g., from slider 0.0-1.0 represents seconds 0-1,000,000+)
    peerness_raw: 0.0-1.0
    aggressiveness_raw: 0.0-1.0
    """
    # For repetition: apply log transform then normalize
    # Treat slider value 0-1.0 as mapping to actual seconds (0 to ~1,000,000)
    repetition_seconds = repetition_raw * 1e6  # Map 0-1 slider to 0-1M seconds
    repetition_log = np.log1p(repetition_seconds)
    repetition_normalized = (repetition_log - NORMALIZATION_STATS['time_min']) / (NORMALIZATION_STATS['time_max'] - NORMALIZATION_STATS['time_min'] + 1e-8)
    repetition_normalized = np.clip(repetition_normalized, 0, 1)
    
    # Peerness and aggressiveness: just min-max normalize
    peerness_normalized = (peerness_raw - NORMALIZATION_STATS['peer_min']) / (NORMALIZATION_STATS['peer_max'] - NORMALIZATION_STATS['peer_min'] + 1e-8)
    peerness_normalized = np.clip(peerness_normalized, 0, 1)
    
    aggressiveness_normalized = (aggressiveness_raw - NORMALIZATION_STATS['agg_min']) / (NORMALIZATION_STATS['agg_max'] - NORMALIZATION_STATS['agg_min'] + 1e-8)
    aggressiveness_normalized = np.clip(aggressiveness_normalized, 0, 1)
    
    return repetition_normalized, peerness_normalized, aggressiveness_normalized

# Display model info in sidebar for verification
with st.sidebar:
    st.divider()
    st.subheader("🔍 Model Info")
    st.text(f"Model: {MODEL_ID}")
    try:
        import os
        import json
        from pathlib import Path
        
        # Check locally cached model
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_dirs = list(cache_dir.glob("*BERT-cyberbullying*"))
        
        if model_dirs:
            model_dir = model_dirs[0]
            refs_file = model_dir / "refs" / "main"
            if refs_file.exists():
                commit = refs_file.read_text().strip()[:8]
                st.text(f"✅ Status: Loaded")
                st.text(f"Commit: {commit}")
                st.text(f"Cached: {model_dir.stat().st_mtime}")
            else:
                st.text(f"✅ Status: Loaded")
        else:
            st.text(f"✅ Status: Loaded")
            st.caption("Run once to cache model")
    except Exception as e:
        st.text(f"✅ Status: Loaded")
    st.divider()

# Auto-load SHAP background from training CSV (if available) so numeric SHAP works without a sidebar
BACKGROUND_CSV = "model_augmented_complete.csv"
if 'bg_loaded' not in st.session_state:
    try:
        if os.path.exists(BACKGROUND_CSV):
            try:
                import pandas as _pd
                # Inspect header and map known column names to the expected ones
                cols = [c.lower() for c in _pd.read_csv(BACKGROUND_CSV, nrows=0).columns.tolist()]

                # Determine text column
                text_candidates = ["text", "aggregated_message", "message", "aggregated_message_text"]
                text_col = next((c for c in text_candidates if c in cols), None)

                # Determine numeric columns mapping
                peerness_candidates = ["peerness"]
                aggressiveness_candidates = ["aggressiveness", "aggression_ratio", "aggressionratio"]
                repetition_candidates = ["repetition", "avg_time_diff_seconds", "avgtimediffseconds"]

                numeric_cols = []
                # helper to find a candidate present in cols
                def find_candidate(candidates):
                    for cand in candidates:
                        if cand in cols:
                            # return the original cased column name from CSV
                            for orig in _pd.read_csv(BACKGROUND_CSV, nrows=0).columns.tolist():
                                if orig.lower() == cand:
                                    return orig
                    return None

                text_col_orig = find_candidate(text_candidates)
                peerness_col = find_candidate(peerness_candidates)
                aggressiveness_col = find_candidate(aggressiveness_candidates)
                repetition_col = find_candidate(repetition_candidates)

                if text_col_orig is None:
                    raise ValueError("text column not found in training CSV; expected one of: Aggregated_Message (preferred), text")

                if peerness_col and aggressiveness_col and repetition_col:
                    numeric_cols_list = [peerness_col, aggressiveness_col, repetition_col]
                else:
                    numeric_cols_list = None

                bg_texts, bg_numeric, bg_numeric_names = HuggingFaceShapExplainer.load_background_from_csv(
                    BACKGROUND_CSV,
                    text_col=text_col_orig,
                    numeric_cols=numeric_cols_list,
                    max_rows=500,
                )

                st.session_state['bg_texts'] = bg_texts
                st.session_state['bg_numeric'] = bg_numeric
                st.session_state['bg_numeric_names'] = bg_numeric_names
                st.session_state['bg_loaded'] = True
            except Exception as e:
                st.session_state['bg_loaded'] = False
                st.warning(f"Could not load SHAP background from {BACKGROUND_CSV}: {e}")
        else:
            st.session_state['bg_loaded'] = False
    except Exception:
        st.session_state['bg_loaded'] = False

# Add pandas for CSV handling 🔍📖🧾📲📢⚙️⚔️🔋⌨️💾💡
import pandas as pd

# Keep a placeholder for the SHAP explainer in session state (built on demand by the Explain buttons)
if 'explainer' not in st.session_state:
    st.session_state['explainer'] = None

# Note: Sidebar uploader removed — explanations will use a small default text background when the explainer
# is auto-built from the Explain buttons (numeric SHAP is not available without numeric background data).


# --- Left Navigation (Sidebar) ---
st.sidebar.title("Navigation")
_nav_map = {
    "🔎 Predict": "Predict / Upload",
    "ℹ️ More about Cyberbullying": "Info about Cyberbullying",
    "📑 Dataset": "Dataset",
}
_nav_choice = st.sidebar.radio("Go to", list(_nav_map.keys()))
nav = _nav_map[_nav_choice]

if nav == "Info about Cyberbullying":
    st.title("Information about cyberbullying")
    st.header("📢What is Cyberbullying?")
    st.markdown( #🔍📖🧾📲📢⚙️⚔️🔋⌨️💾💡
        """Cyberbullying refers to bullying that is done with the use of digital communication platforms with intent
harass, threaten, or humiliate individuals (U.S. Department of Health and Human Services, 2021). 
Most commonly done by sending offensive material or participating in social violence over various types of
digital media such as forums, social media and messaging applications (Perera & Fernando, 2024)."""
    )
    st.header("⚔️Challenge of Cyberbullying Prevention")
    st.markdown(
        """Particularly hard to notice as parents and teachers may not always have access to oversee the
platforms at which cyberbullying takes place (U.S. Department of Health and Human Services, 2021).
»The aggression happening online also means the bullies have a sense of invincibility and morally
disengage when harassing the victim (Suler, 2004). Additionally, being anonymous allows users to
harass others without facing any meaningful consequences (Palomares et al., 2025)."""
    )
    st.header("💡Impact of Cyberbullying")
    st.markdown(
        """The effects of cyberbullying can be severe as an individual can feel as if they are being
attacked anywhere they are, even in their own home(UNICEF, 2025).  
»Mentally victims can suffer from depression, anxiety and if left untreated this can lead to
self-harm and suicide. In 2025, the lifetime victimization of cyberbullying victims increased
from 33.6% in 2016 to 58.2% this year(Sayed et al., 2025). """
    )
    
    # Insert AdvancementSS image
    st.image("AdvancementSS.png", caption="Advancements in Cyberbullying Detection")
    
    st.header("⚙️What is Cyberbullying Detection")
    st.markdown(
        """Cyberbullying detection involves using computational techniques, including Natural Language Processing and Machine
Learning to automatically identify and classify bullying-related content. 
»These systems aim to improve moderation efficiency and reduce the psychological impact on victims (Sayed et al., 2025).
»Classification models are supervised machine learning models that divide data into predefined classes and learn the class
characteristics based on the input data to make predictions (International Business Machines (IBM), 2024)."""
    )
    st.header("📲What is Multi-Aspect Cyberbullying ")
    st.markdown(
        """Multi-aspect means it cannot be defined by a single action or parameter but rather by a spectrum of aggressive
behaviors, delivery times, and negativity (Ejaz, Choudhury, et al., 2024). 
»This is further complicated by the different types of cyberbullying, such as attacking an individual for their age,
race, religion, sexual orientation and others (Fati et al., 2025).
»Furthermore, cyberbullying is multi-aspect in terms of the platforms used, as it can occur across various digital
venues including social media, online games, and messaging apps."""
    )
    
    # Insert SHAPss image
    st.image("SHAPss.png", caption="SHAP Explainability", width=int(768 * 0.8))
    
    st.markdown(
        """In this project, SHAP (SHapley Additive exPlanations) is applied to improve the interpretability of the deployed machine learning model by explaining individual predictions. SHAP quantifies the contribution of both textual tokens and numerical features
          by comparing each input against a representative background distribution derived from training-like data. This allows the model’s decisions to be decomposed into feature-level 
          attributions, showing how specific inputs influence the final prediction. By providing both local explanations and overall feature importance, SHAP enhances model transparency, supports debugging, and increases trust in the system’s outputs."""
    )

elif nav == "Dataset":
    st.title("Dataset used to train the model")
    st.markdown(
        "The dataset used to train the classifation model is from the baseline paper by Ejaz, Choudhury, et al., 2024. " \
        "It containts above 9500 rows of cyberbullying texts along with the various numerical features that have been implemented in the classification model. " \
        "The data contains texts from X from the year 2022 to 2024 containing tweets from and between users. However only the texts and date and time stamps are authentic, while all the other features seen across all the files in tbe dataset are synthetic" \
        
    )
    
    # Display first image above placeholder
    st.image("MendelyDataSS.png", caption="Mendeley Dataset")
    
    st.subheader("Dataset Source")
    st.markdown(
        "The dataset is sourced from [Mendeley Data](https://data.mendeley.com/datasets/wmx9jj2htd/2). There are 6 files in total as seen in the image below. " \
        "However the class labels between Cyberbullying (1) and Not Cyberbullying (0) had a massive imbalance. Hence, data augmetnation was required.  " \
        "Finally, the numerical features used to train the model; Where peerness can be found in the first file which contains each pair of users along with their peerness rating. Aggressiveness can be found in the last file title CB_Labels where the text between users are shown and labelled as aggressive or not. Repetition is measure from the date and time stamps found in the same CSV. " \
        "All these features contribute to the multi-aspect nature of cyberbullying detection and are concatenated into one dataset to train the model. "  
    )
    
    # Display second image below placeholder
    st.image("CSVFilesSS.png", caption="CSV Files", width=int(768 * 0.8))
    
    st.subheader("Dataset Snapshot (example)")
    try:
        if os.path.exists(BACKGROUND_CSV):
            df_bg = pd.read_csv(BACKGROUND_CSV)
            st.dataframe(df_bg.head(10))
        else:
            example = pd.DataFrame({
                "text": ["Sample training message: You are so annoying"],
                "peerness": [0.3],
                "aggressiveness": [0.7],
                "repetition": [0.4]
            })
            st.dataframe(example)
    except Exception as e:
        st.warning(f"Could not load dataset snapshot: {e}")
        example = pd.DataFrame({
            "text": ["Sample training message: You are so annoying"],
            "peerness": [0.3],
            "aggressiveness": [0.7],
            "repetition": [0.4]
        })
        st.dataframe(example)

elif nav == "Predict / Upload":
    st.title("Cyberbullying Detection Platform")
    st.markdown(
        "Multiaspect cyberbullying considers both the message text and contextual numerical features (peerness, aggressiveness, repetition). Use the single prediction form below or upload a CSV for batch predictions."
    )

    st.header("Case by case prediction")
    if st.session_state.get('bg_loaded'):
        st.caption("Numeric SHAP available (using model_augmented_complete.csv background).")
    text = st.text_area("Text to analyze:", height=150)

    st.subheader("Adjust Contextual Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        peerness = st.slider("Peerness", 0.0, 1.0, 0.5, 0.01)
    with col2:
        aggressiveness = st.slider("Aggressiveness", 0.0, 1.0, 0.5, 0.01)
    with col3:
        repetition = st.slider("Repetition", 0.0, 1.0, 0.5, 0.01)

    if st.button("Predict"):
        if text.strip() == "":
            st.warning("Please enter some text for analysis.")
        else:
            # Prepare text input
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

            # Normalize features using training preprocessing
            rep_norm, peer_norm, agg_norm = normalize_features(repetition, peerness, aggressiveness)
            # IMPORTANT: Feature order must match training! Should be [rep, peer, agg]
            numerical_features = torch.tensor([[rep_norm, peer_norm, agg_norm]], dtype=torch.float)
            if torch.cuda.is_available():
                numerical_features = numerical_features.to(model.device)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            # --- MODEL INFERENCE ---
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    additional_features=numerical_features
                )
            # --- SOFTMAX & CONFIDENCE --
            import torch.nn.functional as F
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            # Boost class 1 probability to account for model bias toward class 0
            boosted_prob = probs[:, 1] * 3.0  # Amplify cyberbullying probability
            pred_class = (boosted_prob >= 0.5).long().item()
            pred_prob = probs[0, pred_class].item()
            
            # DEBUG: Show raw values with scaling info
            rep_raw_scaled = repetition * NORMALIZATION_STATS['time_max']  # Show 0-17.4127 value
            st.write(f"**DEBUG - Raw Logits:** {logits.detach().cpu().numpy()}")
            st.write(f"**DEBUG - Probabilities:** {probs.detach().cpu().numpy()}")
            st.write(f"**DEBUG - Slider Values:** Repetition={repetition:.4f} (→ log-scale: {rep_raw_scaled:.4f}), Peerness={peerness:.4f}, Aggressiveness={aggressiveness:.4f}")
            st.write(f"**DEBUG - Normalized Features Sent (0-1):** [rep={rep_norm:.4f}, peer={peer_norm:.4f}, agg={agg_norm:.4f}]")

            # Persist last prediction in session state so it survives reruns
            st.session_state['last_prediction'] = {
                'text': text,
                'numeric': [peerness, aggressiveness, repetition],
                'pred_class': int(pred_class),
                'pred_prob': float(pred_prob),
                'probs': probs.detach().cpu().numpy(),
            }

            label_map = {0: "Not Cyberbullying", 1: "Cyberbullying"}
            # Prediction will be displayed in the persisted results section below (to avoid duplicate outputs)


    # Persisted prediction display (survives reruns) ---------------------------------
    if st.session_state.get('last_prediction') is not None:
        pred = st.session_state['last_prediction']
        label_map = {0: "Not Cyberbullying", 1: "Cyberbullying"}
        st.success(f"Prediction: {label_map.get(pred['pred_class'], pred['pred_class'])}")
        st.write("Class probabilities:")
        probs = pred['probs']
        try:
            if probs.shape[1] == 2:
                st.write(f"Not Cyberbullying: {probs[0,0]*100:.2f}%")
                st.write(f"Cyberbullying: {probs[0,1]*100:.2f}%")
            else:
                st.write(probs)
        except Exception:
            st.write(probs)

        # Explain this persisted input
        if st.button("Explain this input", key="explain_persist"):
            # Ensure explainer exists (build a lightweight default if not)
            if st.session_state.get('explainer') is None:
                with st.spinner("Building default SHAP explainer (this may take a moment)..."):
                    try:
                        default_bg = ["I love this!", "You are terrible", "I appreciate your help", "You're an idiot"]
                        bg_texts = st.session_state.get('bg_texts', default_bg)
                        bg_numeric = st.session_state.get('bg_numeric', None)
                        bg_numeric_names = st.session_state.get('bg_numeric_names', None)
                        st.session_state['explainer'] = HuggingFaceShapExplainer(
                            MODEL_ID,
                            background_texts=bg_texts[:200],
                            background_numeric=(bg_numeric[:200] if bg_numeric is not None else None),
                            numeric_feature_names=bg_numeric_names,
                        )
                        # Explainer built silently (no popup).
                    except Exception as e:
                        st.error(f"Could not build explainer: {e}")
            expl = st.session_state.get('explainer')
            # If a numeric background was auto-loaded after the explainer was built, rebuild explainer to include it
            if expl is not None and getattr(expl, 'background_numeric', None) is None and st.session_state.get('bg_numeric') is not None:
                with st.spinner("Updating explainer to include numeric background..."):
                    try:
                        default_bg = ["I love this!", "You are terrible", "I appreciate your help", "You're an idiot"]
                        bg_texts = st.session_state.get('bg_texts', default_bg)
                        bg_numeric = st.session_state.get('bg_numeric')
                        bg_numeric_names = st.session_state.get('bg_numeric_names')
                        st.session_state['explainer'] = HuggingFaceShapExplainer(
                            MODEL_ID,
                            background_texts=bg_texts[:200],
                            background_numeric=(bg_numeric[:200] if bg_numeric is not None else None),
                            numeric_feature_names=bg_numeric_names,
                        )
                        expl = st.session_state.get('explainer')
                        # Explainer updated silently (no popup).
                    except Exception as e:
                        st.error(f"Could not update explainer with numeric background: {e}")

            if expl is not None:
                if getattr(expl, 'background_numeric', None) is not None:
                    numeric_input = pred['numeric']
                else:
                    numeric_input = None
                    st.info("Numeric SHAP not available: no numeric background provided. Showing token-level SHAP only.")

                try:
                    with st.spinner("Computing SHAP explanation..."):
                        run_explain_and_render(expl, pred['text'], numeric=numeric_input, label_map=label_map)
                except Exception as e:
                    st.error(f"SHAP explanation failed: {e}")


    # Collapsible explanations under the prediction area
    with st.expander("Feature Explanation"):
        st.subheader("Understanding the Features")
        st.markdown('''\
            This model uses three additional numerical features to better understand the context of potential cyberbullying:

            *   **Peerness (0.0-1.0):** Peerness measures the relationship intensity between users; higher values indicate more frequent interaction.
            *   **Aggressiveness (0.0-1.0):** Aggressiveness estimates how aggressive a user's messages are (ratio of aggressive messages to total messages).
            *   **Repetition (0.0-1.0):** Repetition measures the frequency of messaging (e.g., average time difference between messages).
        ''')

    with st.expander("How This Dashboard Works"):
        st.subheader("How This Dashboard Works")
        st.markdown('''\
            Enter a piece of text that you suspect might contain cyberbullying. Adjust the 'Peerness', 'Aggressiveness', and 'Repetition' sliders to provide additional context. The app uses a fine-tuned BERT model that takes the text plus these numerical features and outputs a label ("Not Cyberbullying" or "Cyberbullying") with a confidence score.

            **Inputs:**
            1.  **Text:** The message or content to be analyzed.
            2.  **Peerness:** A numerical value between 0.0 and 1.0.
            3.  **Aggressiveness:** A numerical value between 0.0 and 1.0.
            4.  **Repetition:** A numerical value between 0.0 and 1.0.

            **Output:**
            *   A prediction label indicating whether cyberbullying is detected and the prediction confidence.
        ''')

    st.markdown("---")
    st.header("Batch CSV Upload")
    st.markdown("Drop a CSV file with columns `text`, `peerness`, `aggressiveness`, `repetition` (case-insensitive).")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"] )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Normalize column names early
            df.columns = df.columns.str.lower().str.strip()
            # Keep a working copy in session_state so user actions (auto-fix / drop) persist across reruns
            st.session_state['batch_df'] = df.copy()
            df = st.session_state['batch_df']
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
        else:
# Use working copy from session_state (may have been modified by 'Auto-fix' or 'Drop invalid rows' actions)
                df = st.session_state.get('batch_df', df)
                required = ["text", "peerness", "aggressiveness", "repetition"]
                missing = [c for c in required if c not in df.columns]
                if missing:
                    st.error(f"Missing required columns: {missing}")
                else:
                    # Coerce numerics (attempt to convert current values)
                    for col in ["peerness", "aggressiveness", "repetition"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                    # Find invalid rows
                    invalid_mask = (
                        df[["peerness", "aggressiveness", "repetition"]].isna() |
                        (df[["peerness", "aggressiveness", "repetition"]] < 0) |
                        (df[["peerness", "aggressiveness", "repetition"]] > 1)
                    )
                    invalid_rows = df[invalid_mask.any(axis=1)]

                    def _run_batch_prediction(df_input):
                        texts = df_input["text"].astype(str).tolist()
                        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
                        # Normalize each feature in the batch
                        norm_features = []
                        for idx, row in df_input.iterrows():
                            rep_norm, peer_norm, agg_norm = normalize_features(
                                row["repetition"], 
                                row["peerness"], 
                                row["aggressiveness"]
                            )
                            # IMPORTANT: Feature order must match training! Should be [rep, peer, agg]
                            norm_features.append([rep_norm, peer_norm, agg_norm])
                        
                        numerical_features = torch.tensor(norm_features, dtype=torch.float)
                        if torch.cuda.is_available():
                            numerical_features = numerical_features.to(model.device)
                            inputs = {k: v.to(model.device) for k, v in inputs.items()}

                        with torch.no_grad():
                            outputs = model(
                                input_ids=inputs["input_ids"],
                                attention_mask=inputs["attention_mask"],
                                additional_features=numerical_features
                            )

                        import torch.nn.functional as F
                        logits = outputs.logits
                        probs = F.softmax(logits, dim=1)
                        # Boost class 1 probability to account for model bias toward class 0
                        boosted_prob = probs[:, 1] * 3.0  # Amplify cyberbullying probability
                        preds = (boosted_prob >= 0.5).long().cpu().numpy()
                        confs = probs[range(len(preds)), preds].cpu().numpy()

                        label_map = {0: "Not Cyberbullying", 1: "Cyberbullying"}
                        df_result = df_input.copy()
                        df_result["prediction_label"] = [label_map[int(p)] for p in preds]
                        df_result["confidence"] = (confs * 100).round(2)

                        st.subheader("Results")
                        st.dataframe(df_result.head(50))

                        csv_bytes = df_result.to_csv(index=False).encode("utf-8")
                        st.download_button("Download results as CSV", csv_bytes, file_name="predictions.csv")

                        # --- Explain a selected row ---
                        with st.expander("Explain a row (SHAP)"):
                            idx = st.number_input("Row index to explain", min_value=0, max_value=max(0, len(df_result)-1), value=0, step=1)
                            if st.button("Explain selected row", key=f"explain_row_{idx}"):
                                # Ensure explainer exists
                                if st.session_state.get('explainer') is None:
                                    with st.spinner("Building default SHAP explainer (this may take a moment)..."):
                                        try:
                                            default_bg = ["I love this!", "You are terrible", "I appreciate your help", "You're an idiot"]
                                            bg_texts = st.session_state.get('bg_texts', default_bg)
                                            bg_numeric = st.session_state.get('bg_numeric', None)
                                            bg_numeric_names = st.session_state.get('bg_numeric_names', None)
                                            st.session_state['explainer'] = HuggingFaceShapExplainer(
                                                MODEL_ID,
                                                background_texts=bg_texts[:200],
                                                background_numeric=(bg_numeric[:200] if bg_numeric is not None else None),
                                                numeric_feature_names=bg_numeric_names,
                                            )
                                            # Explainer built silently (no popup).
                                        except Exception as e:
                                            st.error(f"Could not build explainer: {e}")

                                        expl = st.session_state.get('explainer')
                                        # If background numeric was auto-loaded after explainer built, update explainer
                                        if expl is not None and getattr(expl, 'background_numeric', None) is None and st.session_state.get('bg_numeric') is not None:
                                            with st.spinner("Updating explainer to include numeric background..."):
                                                try:
                                                    default_bg = ["I love this!", "You are terrible", "I appreciate your help", "You're an idiot"]
                                                    bg_texts = st.session_state.get('bg_texts', default_bg)
                                                    bg_numeric = st.session_state.get('bg_numeric')
                                                    bg_numeric_names = st.session_state.get('bg_numeric_names')
                                                    st.session_state['explainer'] = HuggingFaceShapExplainer(
                                                        MODEL_ID,
                                                        background_texts=bg_texts[:200],
                                                        background_numeric=(bg_numeric[:200] if bg_numeric is not None else None),
                                                        numeric_feature_names=bg_numeric_names,
                                                    )
                                                    expl = st.session_state.get('explainer')
                                                    # Explainer updated silently (no popup).
                                                except Exception as e:
                                                    st.error(f"Could not update explainer with numeric background: {e}")

                                        if expl is not None:
                                            sel_text = df_result.loc[idx, 'text']
                                            if getattr(expl, 'background_numeric', None) is not None:
                                                numeric_input = [
                                                    float(df_result.loc[idx, 'peerness']),
                                                    float(df_result.loc[idx, 'aggressiveness']),
                                                    float(df_result.loc[idx, 'repetition'])
                                                ]
                                            else:
                                                numeric_input = None
                                                st.info("Numeric SHAP not available: no numeric background provided. Showing token-level SHAP only.")

                                            try:
                                                with st.spinner("Computing SHAP explanation..."):
                                                    run_explain_and_render(expl, sel_text, numeric=numeric_input, label_map=label_map)
                                            except Exception as e:
                                                st.error(f"SHAP explanation failed: {e}")

                        if st.button("Run Batch Prediction"):
                            _run_batch_prediction(df)
                    if not invalid_rows.empty:
                        st.error("Some rows have invalid numeric features (non-numeric or not in [0,1]). Showing up to 10 examples:")
                        st.dataframe(invalid_rows.head(10))

                        col1, col2, col3 = st.columns([1,1,1])
                        with col1:
                            if st.button("Auto-fix numeric features", key="auto_fix_btn"):
                                # Attempt to auto-clean numeric columns: remove %, commas, convert, interpret 0-100 as percent
                                df_work = df.copy()
                                def clean_series(s):
                                    s2 = s.astype(str).str.strip().str.replace('%','', regex=False).str.replace(',','', regex=False)
                                    s_num = pd.to_numeric(s2, errors='coerce')
                                    mask_percent = (s_num > 1) & (s_num <= 100)
                                    s_num.loc[mask_percent] = s_num.loc[mask_percent] / 100.0
                                    s_num = s_num.clip(0.0,1.0)
                                    return s_num
                                for c in ["peerness", "aggressiveness", "repetition"]:
                                    try:
                                        df_work[c] = clean_series(df_work[c])
                                    except Exception:
                                        df_work[c] = pd.to_numeric(df_work[c], errors='coerce')

                                st.session_state['batch_df'] = df_work
                                if hasattr(st, 'experimental_rerun'):
                                    st.experimental_rerun()
                                else:
                                    st.session_state['batch_ready'] = True
                                    st.success("Changes applied — preview updated.")

                        with col2:
                            if st.button("Drop invalid rows", key="drop_invalid_btn"):
                                df_work = df[~invalid_mask.any(axis=1)].copy()
                                st.session_state['batch_df'] = df_work
                                if hasattr(st, 'experimental_rerun'):
                                    st.experimental_rerun()
                                else:
                                    st.session_state['batch_ready'] = True
                                    st.success("Invalid rows removed — preview updated.")

                        with col3:
                            st.info("Auto-fix will try to convert percentages and remove commas, then clip to [0,1]. 'Drop invalid rows' removes offending rows.")

                        # If a fix/drop was applied but the page could not auto-rerun, provide a CTA to run predictions
                        if st.session_state.get('batch_ready'):
                            st.success("Cleaned dataset is ready for prediction.")
                            if st.button("Run Batch Prediction on cleaned data", key="run_after_fix"):
                                _run_batch_prediction(st.session_state.get('batch_df'))
                                st.session_state['batch_ready'] = False
