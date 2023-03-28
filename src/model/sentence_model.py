import os

import numpy as np
import torch
import torch.nn as nn

from src.model.model import Model


class SentenceModel(Model):
    def __init__(self, configuration, device):
        super(SentenceModel, self).__init__(configuration, device)

        self.num_classes = len(configuration['data']['labels'])
        self.linear = nn.Linear(self.encoder.config.hidden_size, self.num_classes)
        self.label_criteria = torch.nn.CrossEntropyLoss()

    def forward(self, batch):
        outputs = self.encoder(batch['input_ids'], attention_mask=batch['attention_mask'])
        cls = outputs.last_hidden_state[:, 0]
        logits = self.linear(cls)

        loss = self.label_criteria(logits, batch['label'].to(self.device))

        return loss

    def predict(self, batch):
        with torch.no_grad():
            outputs = self.encoder(batch['input_ids'], attention_mask=batch['attention_mask'])
            cls = outputs.last_hidden_state[:, 0]
            logits = self.linear(cls)

        if 'label' in batch:
            loss = self.label_criteria(logits, batch['label'].to(self.device))
            loss = loss.cpu().numpy()
        else:
            loss = None

        # we can't directly convert any tensor requiring gradients to numpy arrays.
        # so we need to call .detach() first to remove the computational graph tracking.
        # .cpu is in case the tensor is on the GPU, in which case you need to move it back to the CPU to convert it to a tensor
        logits_cpu = logits.detach().cpu()

        softmax = torch.softmax(logits_cpu, dim=-1)  # do a softmax over the prediction probabilities
        argmax_probs = torch.max(softmax, dim=-1).values.numpy()  # probability of the argmax

        preds = logits_cpu.numpy()  # the raw logit scores
        preds = np.argmax(preds, axis=-1)  # label index of the argmax

        if 'label' in batch:
            out_labels = batch['label'].detach().cpu().numpy()
        else:
            out_labels = None

        return preds, argmax_probs, out_labels, loss
