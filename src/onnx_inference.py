
import torch
from datasets import Dataset
import numpy as np
import onnxruntime as ort
from torch import softmax
from transformers import AutoTokenizer, BertTokenizer

from src.data import SpanDataset
from src.sentence_classification import SentenceClassification
from src.utils import to_numpy


class ONNXPredictor(object):
    def __init__(self, configuration):
        self.config = configuration
        model_path = '{}/model.onnx'.format(self.config['processing']['model_path'])
        print('In ONNXPredictor __init__, model_path=', model_path)
        # self.ort_session = ort.InferenceSession(model_path)
        print('instantiated self.ort_session')
        self.labels = ["unacceptable", "acceptable"]
        print('In ONNXPredictor __init__, loading AutoTokenizer.from_pretrained, config[model][encoder_name]=', self.config['model']['encoder_name'])
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['encoder_name'])
        print('Done loading tokenizer')

    def predict(self, text):
        ds = Dataset.from_dict({'sentence': [text]})
        # extractor = SentenceClassification(self.config)
        #data_loader = extractor.prepare_dataset(ds)

        encoded_text = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.config['model']['max_seq_length'])

        ort_inputs = {
            "input_ids": np.expand_dims(encoded_text["input_ids"], axis=0),
            "attention_mask": np.expand_dims(encoded_text["attention_mask"], axis=0),
        }

        # dataset = SpanDataset(ds, self.config, self.tokenizer)
        # dataset.encode()
        # data_loader = dataset.data_loader()
        #
        # input_batch = next(iter(data_loader))
        # # ort_inputs = {
        # #     'input_ids': to_numpy(input_batch['input_ids'][0].unsqueeze(0).to(extractor.device)),
        # #     'attention_mask': to_numpy(input_batch['attention_mask'][0].unsqueeze(0).to(extractor.device)),
        # # }
        # ort_inputs = {
        #     'input_ids': to_numpy(input_batch['input_ids'][0].unsqueeze(0)),
        #     'attention_mask': to_numpy(input_batch['attention_mask'][0].unsqueeze(0)),
        # }

        # print(ort_inputs)

        ort_outs = self.ort_session.run(None, ort_inputs)
        scores = softmax(torch.from_numpy(ort_outs[0]), dim=-1)[0]
        scores = scores.tolist()
        print('scores=', scores)
        print('labels=', self.labels)

        predictions = []
        for score, label in zip(scores, self.labels):
            predictions.append({"label": label, "score": score})
        return predictions