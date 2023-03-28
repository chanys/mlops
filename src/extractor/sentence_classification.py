import os

import torch

from src.data.data import SpanDataset
from src.extractor.extractor import Extractor
from src.model.sentence_model import SentenceModel


class SentenceClassification(Extractor):
    def __init__(self, configuration):
        super(SentenceClassification, self).__init__(configuration)
        self.model, self.tokenizer = self._create_model()
        self.input_fields = self.tokenizer.model_input_names
        if self.config['processing']['mode'] == 'train' or self.config['processing']['mode'] == 'test':
            self.input_fields += ['label']

    def _create_model(self):
        model = None

        if self.config['processing']['mode'] == 'inference':
            model = SentenceModel(self.config, self.device)
            checkpoint = torch.load(os.path.join(self.config['processing']['model_path'], 'checkpoint.pth.tar'))
            model.load_state_dict(checkpoint['state_dict'])
        elif self.config['processing']['mode'] == 'test':
            model = SentenceModel(self.config, self.device)
            checkpoint = torch.load(os.path.join(self.config['processing']['model_path'], 'checkpoint.pth.tar'))
            model.load_state_dict(checkpoint['state_dict'])
        else:           # train from scratch
            model = SentenceModel(self.config, self.device)

        model.to(self.device)

        return model, model.tokenizer

    def _prepare_dataset(self, data):
        dataset = SpanDataset(data, self.config, self.tokenizer)
        dataset.encode()
        return dataset.data_loader()

