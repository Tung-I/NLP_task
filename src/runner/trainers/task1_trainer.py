import torch.nn.functional as F
import torch
from src.runner.trainers import BaseTrainer


class Task1Trainer(BaseTrainer):
    """
    """

    def __init__(self, meta_data=None, **kwargs):
        super().__init__(**kwargs)
        self.meta_data = meta_data

    def _train_step(self, batch):
        inputs, targets, masks = batch['inputs'].to(self.device), batch['targets'].to(self.device), batch['masks'].to(self.device)
        _, logits = self.net(inputs, token_type_ids=None, attention_mask=masks, labels=targets)
      
        losses = {loss.get_name():loss(logits, targets.view(-1)) for loss in self.loss_fns}
        loss = 0
        for i, loss_name in enumerate(losses.keys()):
            loss += losses[loss_name] * self.loss_weights[i]

        metrics = {metric.get_name():metric(logits, targets.view(-1)) for metric in self.metric_fns}

        # print('{l1}, {l2}'.format(l1=loss, l2=l))

        return {
            'loss': loss,
            'losses': losses,
            'metrics': metrics,
            'outputs': logits,
            'b_labels': targets
        }

    def _valid_step(self, batch):
        inputs, targets, masks = batch['inputs'].to(self.device), batch['targets'].to(self.device), batch['masks'].to(self.device)
        _, logits = self.net(inputs, token_type_ids=None, attention_mask=masks, labels=targets)
      
        losses = {loss.get_name():loss(logits, targets.view(-1)) for loss in self.loss_fns}
        loss = 0
        for i, loss_name in enumerate(losses.keys()):
            loss += losses[loss_name] * self.loss_weights[i]

        metrics = {metric.get_name():metric(logits, targets.view(-1)) for metric in self.metric_fns}

        return {
            'loss': loss,
            'losses': losses,
            'metrics': metrics,
            'outputs': logits
        }
