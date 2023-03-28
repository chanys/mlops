import logging
import argparse

from datasets import load_dataset, Dataset
import wandb
from hydra import initialize, compose
from omegaconf import OmegaConf

from src.extractor.sentence_classification import SentenceClassification

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
        (dl) chanys@chanys-tower:~/repos/mlops/expts$ CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/chanys/repos/mlops python ../src/run.py --mode train
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
        sentence = 'Huawei reportedly said it has developed its own chip design tools, a move aimed at side-stepping U.S. sanctions.'
        sentence2 = 'This is sentence grammatical but not.'
        sentence3 = 'How what think I space now'
        ds = Dataset.from_dict({'sentence': [sentence, sentence2, sentence3]})
        predictions = classifier.inference(ds)
        print(predictions)


