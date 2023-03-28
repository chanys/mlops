import logging
import random

import numpy as np
import torch

from transformers import AdamW, get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_optimizer_and_scheduler(model, lr, num_warmup_steps, num_training_steps):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_groups = [
        {
            'params': [p for n, p in list(model.named_parameters()) if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in list(model.named_parameters()) if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    optimizer = AdamW(params=param_groups, lr=lr, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    return optimizer, scheduler


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

