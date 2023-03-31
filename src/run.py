import logging
import argparse

import torch
from datasets import load_dataset, Dataset
import wandb
from hydra import initialize, compose
from omegaconf import OmegaConf
import onnxruntime as ort
from torch import softmax
from transformers import AutoTokenizer

from src.onnx_inference import ONNXPredictor
from src.sentence_classification import SentenceClassification
from src.utils import to_numpy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
model:
  max_seq_length: 128
  encoder_name: bert-base-uncased

hyperparams:
  epoch: 3
  lr: 2e-05
  warmup_steps: 0
  batch_size: 32

processing:
  seed: 42
  model_path: model_output
  cache_dir: /home/chanys/repos/mlops/cache_huggingface
  
data:
  text_field_name: sentence
"""
# def set_config():
#     configuration = dict()
#     configuration['batch_size'] = 32
#     configuration['max_seq_length'] = 128
#     configuration['text_field_name'] = 'sentence'
#     configuration['encoder_name'] = 'bert-base-uncased'
#     configuration['epoch'] = 3
#     configuration['lr'] = 2e-05
#     configuration['warmup_steps'] = 0
#     configuration['seed'] = 42
#     configuration['model_path'] = 'model_output'
#     configuration['cache_dir'] = '/home/chanys/repos/mlops/cache_huggingface'
#     return configuration

"""
Login to wandb: (dl) chanys@chanys-tower:~/repos/mlops/expts$ wandb login
"""


def convert_model(configuration, examples):
    configuration['processing']['mode'] = 'inference'
    extractor = SentenceClassification(configuration)

    data_loader = extractor.prepare_dataset(examples)
    input_batch = next(iter(data_loader))
    input_sample = {
        'input_ids': input_batch['input_ids'][0].unsqueeze(0).to(extractor.device),
        'attention_mask': input_batch['attention_mask'][0].unsqueeze(0).to(extractor.device),
    }
    extractor.model.eval()

    # Export the model
    logger.info('Converting the model into ONNX format')
    torch.onnx.export(
        extractor.model,  # model being run
        (
            input_sample['input_ids'],
            input_sample['attention_mask'],
        ),  # model input (or a tuple for multiple inputs)
        '{}/model.onnx'.format(config['processing']['model_path']),  # where to save the model
        export_params=True,
        opset_version=15,
        input_names=['input_ids', 'attention_mask'],  # the model's input names
        output_names=['logits'],  # the model's output names
        dynamic_axes={
            'input_ids': {0: 'batch_size'},  # variable length axes
            'attention_mask': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
        },
    )

    logger.info(
        'Model converted successfully. ONNX format model is at: {}/models/model.onnx'.format(config['processing']['model_path'])
    )






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True)
    args = parser.parse_args()

    # configuration
    initialize("../expts/configs")  # this must be a relative directory
    cfg = compose(config_name="config.yaml")
    config = OmegaConf.to_object(cfg)
    config['processing']['mode'] = args.mode

    ds = load_dataset("glue", 'cola', cache_dir=config['processing']['cache_dir'])
    # ds = load_dataset('emotion')
    train_data = ds['train']
    val_data = ds['validation']
    test_data = ds['test']

    # wandb logging
    wandb_config = {'epochs': config['hyperparams']['epoch'], 'learning_rate': config['hyperparams']['lr'],
                    'batch_size': config['hyperparams']['batch_size'], 'seed': config['processing']['seed'],
                    'encoder_name': config['model']['encoder_name'],
                    'max_seq_length': config['model']['max_seq_length']}
    wandb_run = wandb.init(project='mlops-glue-cola', name='baseline', notes='Logging GLUE cola', config=wandb_config)

    if args.mode == 'train':
        """
        (dl) chanys@chanys-tower:~/repos/mlops/expts$ CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/chanys/repos/mlops python ../src/run.py --mode train
        """
        config['data']['labels'] = train_data.features['label'].names
        classifier = SentenceClassification(config)
        classifier.train(train_data, val_data)

    elif args.mode == 'test':
        config['data']['labels'] = train_data.features['label'].names
        classifier = SentenceClassification(config)
        classifier.test(val_data)

    elif args.mode == 'inference':
        config['data']['labels'] = train_data.features['label'].names
        classifier = SentenceClassification(config)
        #sentence = 'Huawei reportedly said it has developed its own chip design tools, a move aimed at side-stepping U.S. sanctions.'
        #sentence2 = 'This is sentence grammatical but not.'
        sentence3 = 'How what think I space now'
        ds = Dataset.from_dict({'sentence': [sentence3]})
        predictions = classifier.inference(ds)
        print(predictions)

    elif args.mode == 'onnx_convert':
        config['data']['labels'] = train_data.features['label'].names
        convert_model(config, train_data)

    elif args.mode == 'onnx_predict':
        config['data']['labels'] = train_data.features['label'].names
        onnx_predictor = ONNXPredictor(config)
        text = 'How what think I space now'
        predictions = onnx_predictor.predict(text)

        print(predictions)
