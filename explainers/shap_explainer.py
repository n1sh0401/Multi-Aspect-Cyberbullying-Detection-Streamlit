"""SHAP explainer utilities for Hugging Face PyTorch models.

This module provides a HuggingFaceShapExplainer class which:
- Loads model + tokenizer from Hugging Face (supports custom local class fallback)
- Keeps prediction wrappers that return class probabilities
- Builds text SHAP explainer and numeric SHAP explainer (KernelExplainer)
- Explains single instances (text-only or text + numeric)
- Produces matplotlib figures and pandas DataFrame summaries suitable for Streamlit

Design choices:
- For text explanations we use shap.Explainer with a Text masker (fast and token-level).
- For numerical features we use shap.KernelExplainer per-instance with a representative background (small number of features makes kernel feasible).
- Combined explanations aggregate token contributions and numeric feature contributions and plot top contributors.

Note: Requires shap (pip install shap), transformers, torch, numpy, pandas, matplotlib.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import shap
except Exception as e:
    raise ImportError("shap must be installed (pip install shap). Error: %s" % e)

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import torch
import os


class HuggingFaceShapExplainer:
    def __init__(
        self,
        model_id_or_path: str,
        device: Optional[str] = None,
        background_texts: Optional[List[str]] = None,
        background_numeric: Optional[np.ndarray] = None,
        numeric_feature_names: Optional[List[str]] = None,
        load_model_args: Optional[dict] = None,
    ):
        """Initialize the explainer manager.

        Parameters
        - model_id_or_path: HF repo id or local path to model
        - device: 'cuda'|'cpu' or None to auto-detect
        - background_texts: representative list of background texts for SHAP text explainer
        - background_numeric: numpy array (K x M) for numeric feature background (if any)
        - numeric_feature_names: list of M names for numeric features
        - load_model_args: extra kwargs passed to from_pretrained
        """
        self.model_id = model_id_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.background_texts = background_texts
        self.background_numeric = background_numeric
        self.numeric_feature_names = numeric_feature_names
        self.load_model_args = load_model_args or {}

        self.tokenizer = None
        self.model = None
        self.num_labels = None
        self.text_explainer = None
        # kernel explainer created per-instance for numeric features
        # cache for per-numeric-value text explainers (keyed by rounded numeric tuple)
        self._instance_text_explainers: Dict[Tuple[float, ...], Any] = {}

        self._load_tokenizer_and_model()
        if self.background_texts is not None:
            self._maybe_build_text_explainer()

    def _load_tokenizer_and_model(self):
        # Extract token from load_model_args if provided
        token = self.load_model_args.get('token', None) if self.load_model_args else None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=token)

        # Try loading auto classification model; if it fails, try local custom class
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, **self.load_model_args)
        except Exception:
            # Attempt to load with local class name (BertForMultiModalSequenceClassification)
            # This requires the local class in the import path (as in this repo).
            from model.bert_multimodal import BertForMultiModalSequenceClassification
            config = AutoConfig.from_pretrained(self.model_id, token=token)
            try:
                # Try normal load first
                self.model = BertForMultiModalSequenceClassification.from_pretrained(self.model_id, config=config, **self.load_model_args)
            except RuntimeError as e:
                # Common case: size mismatch for classifier when mlp sizes differ. Retry allowing mismatched sizes
                try:
                    self.model = BertForMultiModalSequenceClassification.from_pretrained(
                        self.model_id,
                        config=config,
                        ignore_mismatched_sizes=True,
                        **self.load_model_args,
                    )
                    import warnings
                    warnings.warn(
                        "Model loaded with some mismatched parameter sizes (ignore_mismatched_sizes=True). "
                        "Weights for incompatible layers were randomly initialized.",
                        UserWarning,
                    )
                except Exception:
                    # Last resort: set sensible defaults for mlp sizes and try again with ignore_mismatched_sizes
                    config.mlp_hidden_size = getattr(config, "mlp_hidden_size", 32)
                    config.additional_features_dim = getattr(config, "additional_features_dim", 3)
                    self.model = BertForMultiModalSequenceClassification.from_pretrained(
                        self.model_id,
                        config=config,
                        ignore_mismatched_sizes=True,
                        **self.load_model_args,
                    )

        self.model.to(self.device)
        self.model.eval()

        # Attempt to determine num_labels
        try:
            self.num_labels = int(self.model.num_labels)
        except Exception:
            self.num_labels = getattr(self.model.config, "num_labels", 2)

    def predict_proba_from_texts(self, texts: List[str]) -> np.ndarray:
        """Return class probabilities (n_samples, num_labels) for given raw texts."""
        self.model.eval()
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
            logits = out if isinstance(out, torch.Tensor) else out.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict_positive_proba_from_texts(self, texts: List[str]) -> np.ndarray:
        """Return probability of positive class (class index 1) or probability for class 0 if two-class inverted."""
        probs = self.predict_proba_from_texts(texts)
        if probs.shape[1] == 1:
            # Single-label regression/binary sigmoid
            return probs[:, 0]
        elif probs.shape[1] == 2:
            return probs[:, 1]
        else:
            # For multi-class, return max-prob class for compatibility (user can adapt)
            return probs

    def _maybe_build_text_explainer(self):
        if self.background_texts is None:
            raise ValueError("background_texts are required to build text explainer")
        # Create a text masker using the tokenizer
        masker = shap.maskers.Text(self.tokenizer)
        # Our model wrapper for shap should accept list of raw strings and return probabilities
        def wrapped(texts: List[str]):
            # shap may pass numpy arrays of dtype object
            texts = [str(t) for t in texts]
            return self.predict_proba_from_texts(texts)

        # Build explainer for the full-probability output
        self.text_explainer = shap.Explainer(wrapped, masker, output_names=[f"class_{i}" for i in range(self.num_labels)])

    def _get_text_explainer_for_numeric(self, numeric: Optional[List[float]]):
        """Return a token-level explainer conditioned on the numeric vector (cached).

        If numeric is None this returns the base text explainer (built from background_texts).
        """
        # If no numeric specified, return default text explainer
        if numeric is None:
            return self.text_explainer

        # Build a stable cache key from numeric values (rounded to avoid FP noise)
        key = tuple(round(float(x), 6) for x in numeric)
        if key in self._instance_text_explainers:
            return self._instance_text_explainers[key]

        # Ensure we have a background text set; fallback to a small default if not
        bg_texts = self.background_texts if self.background_texts is not None else [
            "I love this!",
            "You are terrible",
            "I appreciate your help",
            "You're an idiot",
            "That was very kind of you",
        ]

        masker = shap.maskers.Text(self.tokenizer)

        def wrapped_with_numeric(texts: List[str]):
            texts = [str(t) for t in texts]
            enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            numeric_np = np.array(numeric, dtype=float).reshape(1, -1)
            numeric_repeated = np.repeat(numeric_np, len(texts), axis=0)
            numeric_tensor = torch.tensor(numeric_repeated, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                try:
                    out = self.model(**enc, additional_features=numeric_tensor)
                except TypeError:
                    out = self.model(**enc)
                logits = out if isinstance(out, torch.Tensor) else out.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            return probs

        expl = shap.Explainer(wrapped_with_numeric, masker, output_names=[f"class_{i}" for i in range(self.num_labels)])
        self._instance_text_explainers[key] = expl
        return expl

    def explain_instance(self, text: str, numeric: Optional[List[float]] = None, top_k: int = 15) -> Dict[str, Any]:
        """Explain a single prediction.

        Returns a dictionary with:
         - prediction_class
         - prediction_prob
         - text_shap: list of (token, shap_value)
         - numeric_shap: dict feature_name->value (if numeric provided)
         - figure: matplotlib Figure combining top contributions (bar plot)
         - explanation_objects: raw shap_EXPLANATIONs for text and numeric (if available)
        """
        # Prediction: compute text-only probability first
        text_probs = self.predict_proba_from_texts([text])
        if text_probs.shape[1] == 1:
            pred_prob_text = float(text_probs[0, 0])
        elif text_probs.shape[1] == 2:
            pred_prob_text = float(text_probs[0, 1])
        else:
            pred_prob_text = float(np.max(text_probs))
        # Initialize final prediction as text-only; if numeric provided we'll recompute
        pred_prob = pred_prob_text
        pred_class = int(np.argmax(text_probs[0])) if text_probs.shape[1] != 1 else int(round(pred_prob_text))
        # If numeric is supplied, compute numeric-conditioned prediction using the model (if supported)
        if numeric is not None:
            enc = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            numeric_tensor = torch.tensor(np.array(numeric, dtype=float).reshape(1, -1), dtype=torch.float32).to(self.device)
            with torch.no_grad():
                try:
                    out = self.model(**enc, additional_features=numeric_tensor)
                except TypeError:
                    out = self.model(**enc)
                logits = out if isinstance(out, torch.Tensor) else out.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            if probs.shape[1] == 1:
                pred_prob = float(probs[0, 0])
            elif probs.shape[1] == 2:
                pred_prob = float(probs[0, 1])
            else:
                pred_prob = float(np.max(probs))
            pred_class = int(np.argmax(probs[0])) if probs.shape[1] != 1 else int(round(pred_prob))

        text_shap = []
        numeric_shap = {}
        text_expl_obj = None
        numeric_expl_obj = None

        # Text explanation (token-level SHAP). Use an explainer conditioned on 'numeric' when provided
        text_explainer_to_use = None
        if numeric is not None:
            text_explainer_to_use = self._get_text_explainer_for_numeric(numeric)
        else:
            text_explainer_to_use = self.text_explainer if self.text_explainer is not None else None

        if text_explainer_to_use is not None:
            text_expl = text_explainer_to_use([text])
            # shap returns (1,) Explanation; values shape (1, n_tokens, num_output)
            if self.num_labels == 2:
                # choose values for positive class (index 1)
                vals = text_expl.values[0, :, 1]
            else:
                vals = text_expl.values[0, :, pred_class]
            tokens = text_expl.data[0]
            text_shap = list(zip(tokens, vals.tolist()))
            text_expl_obj = text_expl

        # Numeric explanation via KernelExplainer (per-instance with background sample)
        if numeric is not None:
            if self.background_numeric is None:
                raise ValueError("background_numeric must be provided to explain numeric features")
            # Build a model function that takes NxM and returns positive class probabilities (for fixed text)
            arr_background = self.background_numeric

            def model_given_numeric(x_numeric_np: np.ndarray) -> np.ndarray:
                # x_numeric_np shape (N, M); create predictions with repeating the single text
                texts = [text] * x_numeric_np.shape[0]
                # Tokenize texts
                enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                # numeric features to tensor
                numeric_tensor = torch.tensor(x_numeric_np, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    # Some models accept additional_features kwarg; pass if model supports it
                    try:
                        out = self.model(**enc, additional_features=numeric_tensor)
                    except TypeError:
                        # Fallback: model only uses text
                        out = self.model(**enc)
                    logits = out if isinstance(out, torch.Tensor) else out.logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                if probs.shape[1] == 1:
                    return probs[:, 0]
                elif probs.shape[1] == 2:
                    return probs[:, 1]
                else:
                    return np.max(probs, axis=1)

            # Use a small subset of background for KernelExplainer to keep speed
            bg = arr_background
            # KernelExplainer expects a function that returns 1-D array per row
            kernel = shap.KernelExplainer(model_given_numeric, bg)
            numeric_arr = np.array(numeric, dtype=float).reshape(1, -1)
            numeric_shap_vals = kernel.shap_values(numeric_arr, nsamples=100)
            # shap_values returns list for each class (for multi-class) or array for single output
            if isinstance(numeric_shap_vals, list):
                # take positive class (index 1) if exists, otherwise predicted class (argmax)
                if len(numeric_shap_vals) == 2:
                    vals = numeric_shap_vals[1].reshape(-1)
                else:
                    # pick predicted class
                    vals = numeric_shap_vals[pred_class].reshape(-1)
            else:
                vals = np.array(numeric_shap_vals).reshape(-1)

            # Map to feature names
            names = self.numeric_feature_names or [f"num_{i}" for i in range(vals.shape[0])]
            numeric_shap = dict(zip(names, vals.tolist()))
            numeric_expl_obj = (kernel, numeric_shap_vals)

        # Aggregate top contributors (combine token-level sums and numeric features)
        # For tokens, use raw token-level values; for numeric, use numeric_shap
        contributions = []
        if text_shap:
            for tok, val in text_shap:
                contributions.append({"feature": tok, "type": "token", "shap_value": float(val)})
        if numeric_shap:
            for k, v in numeric_shap.items():
                contributions.append({"feature": k, "type": "numeric", "shap_value": float(v)})

        df = pd.DataFrame(contributions)
        if df.empty:
            fig = None
        else:
            df["abs_shap"] = df["shap_value"].abs()
            df = df.sort_values("abs_shap", ascending=False).head(top_k)
            # Plot horizontal bar: positive/negative colors
            fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(df))))
            colors = df["shap_value"].apply(lambda x: "tab:green" if x >= 0 else "tab:red")
            ax.barh(df["feature"].astype(str), df["shap_value"], color=colors)
            ax.set_xlabel("SHAP value (contribution to output probability)")
            ax.set_title("Top feature/token contributions")
            plt.gca().invert_yaxis()

        return {
            "prediction_class": int(pred_class),
            "prediction_prob": float(pred_prob),                # final prob (text+numeric when numeric supplied)
            "prediction_prob_text": float(pred_prob_text),     # text-only probability
            "text_shap": text_shap,
            "numeric_shap": numeric_shap,
            "figure": fig,
            "breakdown": df,
            "text_explanation": text_expl_obj,
            "numeric_explanation": numeric_expl_obj,
        }

    @staticmethod
    def load_background_from_csv(path: str, text_col: str = "text", numeric_cols: Optional[List[str]] = None, max_rows: int = 1000) -> Tuple[List[str], Optional[np.ndarray], Optional[List[str]]]:
        """Load background data from a CSV file and return (texts, numeric_array, numeric_names)."""
        import pandas as pd
        df = pd.read_csv(path)
        if text_col not in df.columns:
            raise ValueError(f"text_col '{text_col}' not found in {path}")
        texts = df[text_col].astype(str).tolist()[:max_rows]
        numeric_arr = None
        numeric_names = None
        if numeric_cols is not None:
            for c in numeric_cols:
                if c not in df.columns:
                    raise ValueError(f"numeric column '{c}' not present in {path}")
            arr = df[numeric_cols].astype(float).values
            numeric_arr = arr[:max_rows]
            numeric_names = numeric_cols
        return texts, numeric_arr, numeric_names
