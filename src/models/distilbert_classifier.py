"""
DistilBERT-based classifier for WMT21 Task 3 Critical Error Detection.

This module implements a fine-tuned DistilBERT model for binary classification
of critical translation errors.
"""

import torch
import torch.nn as nn
from transformers import (
    DistilBertModel,
    DistilBertPreTrainedModel,
    DistilBertConfig
)
from typing import Optional, Tuple, Dict, Any


class DistilBERTClassifier(DistilBertPreTrainedModel):
    """
    DistilBERT model for sequence classification.
    
    This model fine-tunes DistilBERT for binary classification of critical
    translation errors in machine translation output.
    """

    def __init__(self, config: DistilBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.distilbert = DistilBertModel(config)
        
        # Classification head
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        
        # Initialize weights
        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            head_mask: Head mask
            inputs_embeds: Input embeddings
            labels: Labels for computing loss
            output_attentions: Whether to return attentions
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return dict
            
        Returns:
            Tuple containing loss and logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get DistilBERT outputs
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Use [CLS] token representation
        hidden_state = distilbert_output[0]  # (batch_size, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (batch_size, dim)
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': distilbert_output.hidden_states,
            'attentions': distilbert_output.attentions,
        }

    def get_model_size(self) -> Dict[str, int]:
        """
        Calculate model size information for WMT21 submission.
        
        Returns:
            Dictionary with model size metrics
        """
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate disk footprint (rough approximation)
        # Each parameter is typically 4 bytes (float32)
        disk_footprint_bytes = total_params * 4
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'disk_footprint_bytes': disk_footprint_bytes
        } 