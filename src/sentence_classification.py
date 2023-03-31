import os
import logging

import torch
import wandb
from tqdm import tqdm
import numpy as np

from src.data import SpanDataset
from src.sentence_model import SentenceModel
from src.utils import set_seed, prepare_optimizer_and_scheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentenceClassification(object):
    def __init__(self, configuration):
        self.config = configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(self.config['processing']['seed'])

        self.model, self.tokenizer = self._create_model()
        self.input_fields = self.tokenizer.model_input_names
        if self.config['processing']['mode'] == 'train' or self.config['processing']['mode'] == 'test':
            self.input_fields += ['label']

    def _create_model(self):
        model = None
        if self.config['processing']['mode'] in ['test', 'inference']:
            model = SentenceModel(self.config, self.device)
            model_path = os.path.join(self.config['processing']['model_path'], 'checkpoint.pth.tar')
            logger.info('Loading model from {}'.format(model_path))
            if self.config['processing']['device'] == 'cpu':
                # checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                checkpoint = torch.load(model_path)
            else:
                checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['state_dict'])
        else:           # train from scratch
            model = SentenceModel(self.config, self.device)

        model.to(self.device)
        return model, model.tokenizer

    def prepare_dataset(self, data):
        dataset = SpanDataset(data, self.config, self.tokenizer)
        dataset.encode()
        return dataset.data_loader()

    def inference(self, test_examples):
        dataloader = self.prepare_dataset(test_examples)

        self.model.eval()
        predictions = []
        for batch in dataloader:
            inputs = {k: v.to(self.device) for k, v in batch.items() if k in self.input_fields}
            preds, argmax_probs, out_labels, loss = self.predict(inputs['input_ids'], inputs['attention_mask'])
            predictions.extend(preds)

        return predictions

    def predict(self, input_ids, attention_masks, labels=None):
        with torch.no_grad():
            outputs = self.model.forward(input_ids, attention_masks, labels)
            #outputs = self.encoder(input_ids, attention_masks)
            #cls = outputs.last_hidden_state[:, 0]
            #logits = self.linear(cls)

        if labels is not None:
            loss = self.model.label_criteria(outputs.logits, labels.to(self.device))
            loss = loss.cpu().numpy()
        else:
            loss = None

        # we can't directly convert any tensor requiring gradients to numpy arrays.
        # so we need to call .detach() first to remove the computational graph tracking.
        # .cpu is in case the tensor is on the GPU, in which case you need to move it back to the CPU to convert it to a tensor
        logits_cpu = outputs.logits.detach().cpu()
        softmax = torch.softmax(logits_cpu, dim=-1)  # do a softmax over the prediction probabilities
        argmax_probs = torch.max(softmax, dim=-1).values.numpy()  # probability of the argmax

        preds = logits_cpu.numpy()  # the raw logit scores
        preds = np.argmax(preds, axis=-1)  # label index of the argmax

        if labels is not None:
            out_labels = labels.detach().cpu().numpy()
        else:
            out_labels = None

        return preds, argmax_probs, out_labels, loss

    def test(self, test_examples):
        test_dataloader = self.prepare_dataset(test_examples)

        self.model.eval()
        predictions = []
        gold_labels = []
        losses = []
        for batch in test_dataloader:
            inputs = {k: v.to(self.device) for k, v in batch.items() if k in self.input_fields}
            preds, argmax_probs, out_labels, loss = self.predict(inputs['input_ids'], inputs['attention_mask'], labels=inputs['label'])
            predictions.extend(preds)
            gold_labels.extend(out_labels)
            losses.append(loss)

        predictions = np.asarray(predictions)
        gold_labels = np.asarray(gold_labels)

        mean_loss = sum(losses) / len(losses)
        print('Test loss is %.5f' % mean_loss)
        print('Test accuracy is %.2f' % (float(np.sum(predictions == gold_labels)) / len(predictions)))

    def train(self, train_examples, dev_examples):
        train_dataloader = self.prepare_dataset(train_examples)
        test_dataloader = self.prepare_dataset(dev_examples)

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

            losses = []
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
            for batch_index, batch in enumerate(epoch_iterator):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k in self.input_fields}
                optimizer.zero_grad()
                outputs = self.model(inputs['input_ids'], inputs['attention_mask'], labels=inputs['label'])
                loss = self.model.label_criteria(outputs.logits, inputs['label'].to(self.device))
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
                preds, argmax_probs, out_labels, loss = self.predict(inputs['input_ids'], inputs['attention_mask'], labels=inputs['label'])
                predictions.extend(preds)
                gold_labels.extend(out_labels)
                losses.append(loss)

            predictions = np.asarray(predictions)
            gold_labels = np.asarray(gold_labels)

            mean_loss = sum(losses) / len(losses)
            print('Validation loss at epoch %d is %.5f' % (epoch_counter, mean_loss))
            val_accuracy = float(np.sum(predictions == gold_labels)) / len(predictions)
            print('Validation accuracy at epoch %d is %.2f' % (epoch_counter, val_accuracy))
            wandb.log({'validation loss': mean_loss})           # wandb logging
            wandb.log({'validation accuracy': val_accuracy})    # wandb logging

        self.model.save_model(self.config['processing']['model_path'], optimizer)

