import streamlit as st
import torch
from transformers import AutoTokenizer, AutoConfig
from model.bert_multimodal import BertForMultiModalSequenceClassification

st.set_page_config(layout="wide")

@st.cache_resource
def load_model():
    model_id = "rngrye/BERT-cyberbullying-classifier-MLPLeakage"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    config = AutoConfig.from_pretrained(model_id)
    model = BertForMultiModalSequenceClassification.from_pretrained(
        model_id,
        config=config
    )

    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
model.eval()

st.title("Cyberbullying Detection App")

# Add pandas for CSV handling
import pandas as pd

# --- Left Navigation (Sidebar) ---
st.sidebar.title("Navigation")
nav = st.sidebar.radio("Go to", ["Explanation", "Dataset", "Predict / Upload"])

if nav == "Explanation":
    # --- Dashboard Description ---
    st.header("What is Cyberbullying?")
    st.markdown(
        "Cyberbullying refers to bullying that is done with the use of digital communication platforms with intent harass, threaten, or humiliate individuals (U.S. Department of Health and Human Services, 2021) "
        "Most commonly done by sending offensive material or participating in social violence over various types of digital media such as forums, social media and messaging applications (Perera & Fernando, 2024) "
        #--Challenges of cyberbullying
        "Cyberbullying can be Particularly hard to notice as parents and teachers may not always have access to oversee the platforms at which cyberbullying takes place (U.S. Department of Health and Human Services, 2021)."
        "The aggression happening online also means the bullies have a sense of invincibility and morally disengage when harassing the victim (Suler, 2004). Additionally, being anonymous allows users to harass others without facing any meaningful consequences (Palomares et al., 2025)."
    )

    st.header("Understanding the Features")
    st.markdown('''
        This model uses three additional numerical features to better understand the context of potential cyberbullying:

        *   **Peerness (0.0-1.0):** Peerness is a measure of the relationship between any two users, a higher peerness indicates the users interact with each other more frequently, potentially reducing probability of cyberbullying, ranging from 0.0 to 1.0
        *   **Aggressiveness (0.0-1.0):** Aggressiveness is a feature that ranges from 0.0 to 1.0 , which is a measure of how aggressive post by a user is, which is measured by taking the ratio of total aggressive messages over the total messages between each pair of users.
        *   **Repetition (0.0-1.0):** Repetition is a measure from the time and date features to see how frequent posts and replies happens between to users. In the dataset, this is labelled as Average Time Difference between Messages in Seconds (Avg_Time_Diff_Secs)." 
    ''')

    st.markdown("---")
    st.header("How This Dashboard Works")
    st.markdown('''
        Enter a piece of text that you suspect might contain cyberbullying. Adjust the 'Peerness', 'Aggressiveness', and 'Repetition' sliders to provide additional context. The app will then use a fine-tuned BERT model to predict whether the text, combined with the numerical context, constitutes cyberbullying. The output will be a label ("Not Cyberbullying" or "Cyberbullying") plus a confidence score.

        **Inputs:**
        1.  **Text:** The message or content to be analyzed.
        2.  **Peerness:** A numerical value between 0.0 and 1.0.
        3.  **Aggressiveness:** A numerical value between 0.0 and 1.0.
        4.  **Repetition:** A numerical value between 0.0 and 1.0.

        **Output:**
        *   A prediction label indicating whether cyberbullying is detected and the prediction confidence.
    ''')

elif nav == "Dataset":
    st.header("Dataset Requirements and Explanation")
    st.markdown("Your CSV must include the following columns: `text`, `peerness`, `aggressiveness`, `repetition`.")
    st.markdown(
        "- `text`: the message to analyze (string).\n"
        "- `peerness`: float in [0.0, 1.0] representing how peer-like the relationship is.\n"
        "- `aggressiveness`: float in [0.0, 1.0] indicating how aggressive the content is.\n"
        "- `repetition`: float in [0.0, 1.0] indicating frequency / repetition of messaging behavior.\n"
    )
    st.markdown("Column names are treated case-insensitively (they will be lowercased). Values outside [0,1] or non-numeric values will be flagged during validation.")

    st.subheader("Example row")
    example = pd.DataFrame({
        "text": ["You are so annoying"],
        "peerness": [0.2],
        "aggressiveness": [0.8],
        "repetition": [0.9]
    })
    st.dataframe(example)

elif nav == "Predict / Upload":
    st.header("Single Prediction")
    st.subheader("Enter Text for Analysis")
    text = st.text_area("Text to analyze:", height=150)

    # Numerical Inputs
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

            # Prepare numerical inputs as a tensor
            numerical_features = torch.tensor([[peerness, aggressiveness, repetition]], dtype=torch.float)
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

            # --- SOFTMAX & CONFIDENCE ---
            import torch.nn.functional as F
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            pred_prob = probs[0, pred_class].item()

            label_map = {0: "Not Cyberbullying", 1: "Cyberbullying"}
            st.success(f"Prediction: {label_map[pred_class]} (Confidence: {pred_prob*100:.2f}%)")

            # Optional: show full probability distribution
            st.write("Class probabilities:")
            st.write(f"Not Cyberbullying: {probs[0,0]*100:.2f}%")
            st.write(f"Cyberbullying: {probs[0,1]*100:.2f}%")

    st.markdown("---")
    st.header("Batch CSV Upload")
    st.markdown("Drop a CSV file with columns `text`, `peerness`, `aggressiveness`, `repetition` (case-insensitive).")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"] )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
        else:
            # Normalize column names
            df.columns = df.columns.str.lower().str.strip()
            required = ["text", "peerness", "aggressiveness", "repetition"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                # Coerce numerics
                for col in ["peerness", "aggressiveness", "repetition"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                # Find invalid rows
                invalid_mask = (
                    df[["peerness", "aggressiveness", "repetition"]].isna() |
                    (df[["peerness", "aggressiveness", "repetition"]] < 0) |
                    (df[["peerness", "aggressiveness", "repetition"]] > 1)
                )
                invalid_rows = df[invalid_mask.any(axis=1)]
                if not invalid_rows.empty:
                    st.error("Some rows have invalid numeric features (non-numeric or not in [0,1]). Showing up to 10 examples:")
                    st.dataframe(invalid_rows.head(10))
                else:
                    st.success("CSV looks good. You can preview and run batch prediction below.")
                    st.subheader("Preview")
                    st.dataframe(df.head(10))

                    if st.button("Run Batch Prediction"):
                        texts = df["text"].astype(str).tolist()
                        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)

                        numerical_features = torch.tensor(df[["peerness", "aggressiveness", "repetition"]].values, dtype=torch.float)
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
                        preds = torch.argmax(probs, dim=1).cpu().numpy()
                        confs = probs[range(len(preds)), preds].cpu().numpy()

                        label_map = {0: "Not Cyberbullying", 1: "Cyberbullying"}
                        df_result = df.copy()
                        df_result["prediction_label"] = [label_map[int(p)] for p in preds]
                        df_result["confidence"] = (confs * 100).round(2)

                        st.subheader("Results")
                        st.dataframe(df_result.head(50))

                        csv_bytes = df_result.to_csv(index=False).encode("utf-8")
                        st.download_button("Download results as CSV", csv_bytes, file_name="predictions.csv")
