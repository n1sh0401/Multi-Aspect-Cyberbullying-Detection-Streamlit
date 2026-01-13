import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertPreTrainedModel, BertModel


# --- 1. Corrected Focal Loss with Label Smoothing ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, label_smoothing=0.1, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply label smoothing manually for cross entropy
        num_classes = inputs.size(-1)
        with torch.no_grad():
            smoothed_labels = torch.full_like(inputs, self.label_smoothing / (num_classes - 1))
            smoothed_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = F.log_softmax(inputs, dim=-1)
        ce_loss = -(smoothed_labels * log_probs).sum(dim=-1)

        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        f_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return f_loss.mean()
        return f_loss.sum()


# --- 2. Model Definition ---
class BertForMultiModalSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.additional_features_dim = getattr(config, "additional_features_dim", 3)

        # Retrieve config parameters for MLP and dropout
        self.mlp_hidden_size = getattr(config, "mlp_hidden_size", 32)
        self.numerical_feature_dropout_prob = getattr(config, "numerical_feature_dropout_prob", 0.3)
        self.text_weight_scale = getattr(config, "text_weight_scale", 1.0)

        # MLP for Additional Features
        self.feature_mlp = nn.Sequential(
            nn.Linear(self.additional_features_dim, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.mlp_hidden_size),
            nn.ReLU()
        )
        self.numerical_feature_dropout = nn.Dropout(self.numerical_feature_dropout_prob)

        # Classifier layers
        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        # Increased dropout for the head
        self.dropout = nn.Dropout(0.4)
        # Modified classifier input size
        self.classifier = nn.Linear(config.hidden_size + self.mlp_hidden_size, config.num_labels)

        self.alpha = getattr(config, 'focal_loss_alpha', 0.5)
        self.gamma = getattr(config, 'focal_loss_gamma', 2.0)
        self.init_weights()

        # Initialize text layer weights
        if self.text_weight_scale != 1.0:
            print(f"Scaling text embeddings by {self.text_weight_scale}")
            torch.nn.init.xavier_uniform_(self.bert.embeddings.word_embeddings.weight.data)
            self.bert.embeddings.word_embeddings.weight.data.mul_(self.text_weight_scale)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, additional_features=None, **kwargs):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]

        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.Tanh()(pooled_output)  # Using Tanh for stability
        pooled_output = self.dropout(pooled_output)

        if additional_features is not None:
            # Process additional features through MLP and dropout
            processed_additional_features = self.feature_mlp(additional_features)
            processed_additional_features = self.numerical_feature_dropout(processed_additional_features)
            combined_features = torch.cat((pooled_output, processed_additional_features), dim=1)
        else:
            # Fallback for dummy features, ensuring correct size
            dummy = torch.zeros((pooled_output.size(0), self.mlp_hidden_size), device=pooled_output.device)
            combined_features = torch.cat((pooled_output, dummy), dim=1)

        logits = self.classifier(combined_features)

        loss = None
        if labels is not None:
            loss_fct = FocalLoss(alpha=self.alpha, gamma=self.gamma, label_smoothing=0.1)
            loss = loss_fct(logits, labels)

        return (loss, logits) if loss is not None else logits


# --- 3. Data Collator for Multimodal Data ---
class MultiModalDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        add_feats = [f.pop('additional_features') for f in features]
        
        # Manually pad input_ids and attention_mask
        max_length = max(len(f['input_ids']) for f in features)
        
        batch_input_ids = []
        batch_attention_mask = []
        
        for f in features:
            input_ids = f['input_ids']
            attention_mask = f['attention_mask']
            
            # Pad to max_length
            padding_length = max_length - len(input_ids)
            batch_input_ids.append(input_ids + [self.tokenizer.pad_token_id] * padding_length)
            batch_attention_mask.append(attention_mask + [0] * padding_length)
        
        batch = {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long),
            'additional_features': torch.tensor(add_feats, dtype=torch.float32),
        }
        
        # Add labels if present
        if 'labels' in features[0]:
            batch['labels'] = torch.tensor([f['labels'] for f in features], dtype=torch.long)
        
        return batch
