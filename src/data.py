import logging

import torch
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpanDataset(Dataset):
    def __init__(self, data, configuration, tokenizer):
        """
        :type data: datasets.arrow_dataset.Dataset
        :type configuration: dict
        :type tokenizer: transformers.tokenization_utils_fast.PreTrainedTokenizerFast
        """
        self.data = data
        self.config = configuration
        self.tokenizer = tokenizer
        self.batch_size = self.config['hyperparams']['batch_size']
        self.max_seq_length = self.config['model']['max_seq_length']
        self.text_field_name = self.config['data']['text_field_name']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def tokenize(self, batch):
        # padding=True will pad the examples with 0s to the size of the longest example in the batch
        # truncation=True will truncate the examples to the model's max sequence length
        # tokenizer.model_input_names = ['input_ids', 'attention_mask']
        return self.tokenizer(batch[self.text_field_name], padding='max_length', truncation=True, max_length=self.max_seq_length)

    def encode(self):
        # batch_size=None will tokenize the entire dataset at once, ensuring we pad each example to be the same length
        self.data = self.data.map(self.tokenize, batched=True, batch_size=None)
        columns = ['input_ids', 'attention_mask'] + [self.text_field_name]
        if self.config['processing']['mode'] == 'train' or self.config['processing']['mode'] == 'test':
            columns += ['label']
        self.data.set_format('torch', columns=columns)

    def data_loader(self):
        if self.config['processing']['mode'] == 'train':
            return torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=True)
        else:
            return torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=False)
