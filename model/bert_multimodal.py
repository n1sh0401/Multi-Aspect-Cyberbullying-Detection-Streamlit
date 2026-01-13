from transformers import BertPreTrainedModel, BertModel
from torch import nn
import torch

class BertForMultiModalSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)

        # multimodal MLP for additional features — use values from config for reproducibility
        self.additional_features_dim = getattr(config, "additional_features_dim", 3)
        self.mlp_hidden_size = getattr(config, "mlp_hidden_size", 8)
        self.feature_mlp = nn.Sequential(
            nn.Linear(self.additional_features_dim, self.mlp_hidden_size),  # e.g. 3 -> 8
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.mlp_hidden_size, self.mlp_hidden_size),
            nn.ReLU()
        )

        # ensure important metadata is present in the config so it is saved with the model
        config.additional_features_dim = self.additional_features_dim
        config.mlp_hidden_size = self.mlp_hidden_size
        # optional metadata keys may be set externally (e.g. feature names, scaling stats)
        config.feature_names = getattr(config, "feature_names", None)
        config.scaling_stats = getattr(config, "scaling_stats", None)

        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(
            config.hidden_size + self.mlp_hidden_size,
            config.num_labels
        )

        dropout_prob = (
            config.classifier_dropout
            if config.classifier_dropout is not None 
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        additional_features=None,
        labels=None,
        return_dict=True,
        **kwargs
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)

        if additional_features is None:
            # create a zero vector matching the processed MLP output size
            processed = torch.zeros(
                pooled_output.size(0),
                self.mlp_hidden_size,
                device=pooled_output.device
            )
        else:
            processed = self.feature_mlp(additional_features)

        pooled_output = torch.cat((pooled_output, processed), dim=1)

        logits = self.classifier(pooled_output)

        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(logits=logits)
