import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.nn import BCEWithLogitsLoss
from transformers import Trainer


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits))
    preds = (probs > 0.5).int()

    return {
        "micro_f1": f1_score(labels, preds, average="micro"),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def compute_pos_weights(train_ds, device, max_clip=20):
    all_labels = np.array([ex["labels"] for ex in train_ds])
    pos_weights = []

    for i in range(all_labels.shape[1]):
        pos = np.sum(all_labels[:, i])
        neg = len(all_labels) - pos
        if pos == 0:
            pos_weights.append(1.0)
        else:
            w = np.log1p(neg / pos)
            pos_weights.append(min(w, max_clip))

    return torch.tensor(pos_weights, device=device)


class CustomTrainer(Trainer):
    def __init__(self, pos_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = BCEWithLogitsLoss(pos_weight=pos_weights)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss
