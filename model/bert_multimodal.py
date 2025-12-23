from transformers import BertPreTrainedModel, BertModel
from torch import nn
import torch

additional_features_dim = 3

class BertForMultiModalSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)

        self.mlp_hidden_size = 32
        self.feature_mlp = nn.Sequential(
            nn.Linear(additional_features_dim, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.mlp_hidden_size),
            nn.ReLU()
        )

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
            additional_features = torch.zeros(
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
