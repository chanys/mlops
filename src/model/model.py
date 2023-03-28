import logging
import os

import torch
import torch.nn as nn

from transformers import AutoTokenizer, BertModel, DistilBertModel

from src.utils import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Model(nn.Module):
    def __init__(self, configuration, device):
        super().__init__()

        self.encoder_name = configuration['model']['encoder_name']
        self.device = device

        set_seed()

        if self.encoder_name.startswith('bert'):
            self.encoder = BertModel.from_pretrained(self.encoder_name, output_hidden_states=True, cache_dir=configuration['processing']['cache_dir'])
        elif self.encoder_name.startswith('distilbert'):
            self.encoder = DistilBertModel.from_pretrained(self.encoder_name, output_hidden_states=True, cache_dir=configuration['processing']['cache_dir'])

        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name, cache_dir=configuration['processing']['cache_dir'])

    def forward(self, batch):
        pass

    def predict(self, batch):
        pass

    def save_model(self, model_dir, optimizer):
        print('=> Saving model to {}'.format(model_dir))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        # checkpoint = {'state_dict': self.model.state_dict(), 'optimizer': optimizer.state_dict()}
        checkpoint = {'state_dict': self.state_dict()}
        torch.save(checkpoint, os.path.join(model_dir, 'checkpoint.pth.tar'))

    def print_trainable_weights(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params

        print('n_trainable_params=', n_trainable_params)
        print('n_nontrainable_params=', n_nontrainable_params)

        logger.info('-' * 100)
        logger.info('> trainable params:')
        for name, param in self.named_parameters():
            if param.requires_grad:
                logger.info('>>> {0}: {1}'.format(name, param.shape))
        logger.info(
            'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('-' * 100)


