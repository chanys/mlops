import logging

import numpy as np
import torch
import wandb
from tqdm import tqdm

from src.utils import set_seed, prepare_optimizer_and_scheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Extractor(object):
    def __init__(self, configuration):
        self.config = configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(self.config['processing']['seed'])
        self.model = None
        self.tokenizer = None
        self.input_fields = None

    def _create_model(self):
        pass

    def _prepare_dataset(self, data):
        pass

    def inference(self, test_examples):
        dataloader = self._prepare_dataset(test_examples)

        self.model.eval()
        predictions = []
        for batch in dataloader:
            inputs = {k: v.to(self.device) for k, v in batch.items() if k in self.input_fields}
            preds, argmax_probs, out_labels, loss = self.model.predict(inputs)
            predictions.extend(preds)

        return predictions

    def test(self, test_examples):
        test_dataloader = self._prepare_dataset(test_examples)

        self.model.eval()
        predictions = []
        gold_labels = []
        losses = []
        for batch in test_dataloader:
            inputs = {k: v.to(self.device) for k, v in batch.items() if k in self.input_fields}
            preds, argmax_probs, out_labels, loss = self.model.predict(inputs)
            predictions.extend(preds)
            gold_labels.extend(out_labels)
            losses.append(loss)
            # sys.exit(0)
        predictions = np.asarray(predictions)
        gold_labels = np.asarray(gold_labels)

        mean_loss = sum(losses) / len(losses)
        print('Test loss is %.5f' % mean_loss)
        print('Test accuracy is %.2f' % (float(np.sum(predictions == gold_labels)) / len(predictions)))

    def train(self, train_examples, dev_examples):
        train_dataloader = self._prepare_dataset(train_examples)
        test_dataloader = self._prepare_dataset(dev_examples)

        self.model.print_trainable_weights()

        lr = self.config['hyperparams']['lr']
        num_warmup_steps = self.config['hyperparams']['warmup_steps']
        num_training_steps = len(train_dataloader) * self.config['hyperparams']['epoch']

        optimizer, scheduler = prepare_optimizer_and_scheduler(self.model, lr, num_warmup_steps, num_training_steps)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Num Epochs = %d", self.config['hyperparams']['epoch'])
        logger.info("  train_batch_size = %d", self.config['hyperparams']['batch_size'])

        for epoch_counter in range(self.config['hyperparams']['epoch']):
            self.model.train()  # sets module to training mode
            # optimizer.zero_grad()

            losses = []
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
            for batch_index, batch in enumerate(epoch_iterator):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k in self.input_fields}
                optimizer.zero_grad()
                loss = self.model(inputs)
                losses.append(loss)

                loss.backward()  # calculate and accumulate the gradients

                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                scheduler.step()  # Update learning rate schedule
                optimizer.step()  # nudge the parameters in the opposite direction of the gradient, in order to decrease the loss

            mean_loss = sum(losses) / len(losses)
            print('Training loss at epoch %d is %.5f' % (epoch_counter, mean_loss))
            wandb.log({'training loss': mean_loss})     # wandb logging

            self.model.eval()
            predictions = []
            gold_labels = []
            losses = []
            for batch in test_dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k in self.input_fields}
                preds, argmax_probs, out_labels, loss = self.model.predict(inputs)
                predictions.extend(preds)
                gold_labels.extend(out_labels)
                losses.append(loss)
                # sys.exit(0)
            predictions = np.asarray(predictions)
            gold_labels = np.asarray(gold_labels)

            mean_loss = sum(losses) / len(losses)
            print('Validation loss at epoch %d is %.5f' % (epoch_counter, mean_loss))
            val_accuracy = float(np.sum(predictions == gold_labels)) / len(predictions)
            print('Validation accuracy at epoch %d is %.2f' % (epoch_counter, val_accuracy))
            wandb.log({'validation loss': mean_loss})           # wandb logging
            wandb.log({'validation accuracy': val_accuracy})    # wandb logging

        self.model.save_model(self.config['processing']['model_path'], optimizer)
