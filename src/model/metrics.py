import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


class Accuracy(nn.Module):
    """The accuracy for the classification task.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, C): The model output.
            target (torch.LongTensor) (N, 1): The data target.
        Returns:
            metric (torch.Tensor) (0): The accuracy.
        """
        pred = output.argmax(dim=1, keepdim=False)
        return (pred == target).float().mean()

    def reset_count(self):
        pass

    def get_name(self):
        return 'Accuracy'


class F1Score(nn.Module):
    """The F1Score for the classification task.
    """
    def __init__(self):
        super().__init__()
        self.store_label = np.array([])
        self.store_pred = np.array([])
        # self.TP = 1e-10
        # self.FP = 1e-10
        # self.TN = 1e-10
        # self.FN = 1e-10

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, C): The model output.
            target (torch.LongTensor) (N, 1): The data target.
        Returns:
            metric (torch.Tensor) (0): The f1 score.
        """
        # pred = output.argmax(dim=1, keepdim=True)
        # pred = torch.zeros_like(output).scatter_(1, pred, 1)
        # target_encode = torch.zeros_like(output).scatter_(1, target.unsqueeze(1), 1)
        
        # TP = (pred * target_encode).sum()
        # FP = (pred * (1 - target_encode)).sum()
        # TN = ((1 - pred) * (1 - target_encode)).sum()
        # FN = ((1 - pred) * target_encode).sum()
        b_label = target.view(-1, 1)
        logits = output.detach().cpu().numpy()
        b_label = b_label.to('cpu').numpy()
        logits = np.argmax(logits, axis=1)
        self.store_label = np.concatenate((self.store_label, b_label.reshape(-1)))
        self.store_pred = np.concatenate((self.store_pred, logits.reshape(-1)))

        _,_,train_f1,_ = precision_recall_fscore_support(self.store_label, self.store_pred, labels=[0,1], average='weighted')
        return train_f1

        # pred = output.argmax(dim=1, keepdim=False)
        # TP = (pred * target).sum()
        # FN = ((1 - pred) * target).sum()
        # FP = (pred * (1 - target)).sum()
        # TN = ((1 - pred) * (1 - target)).sum()
        
        # # print('TP:{tp}, FN:{fn}, TN{tn}, FP{fp}'.format(tp=TP, fn=FN, tn=TN, fp=FP))
        # self.TP += TP
        # self.FP += FP
        # self.TN += TN
        # self.FN += FN

        # return (2*self.TP / (2*self.TP + self.FN + self.FP)).cpu().detach().numpy()

    def reset_count(self):
        self.store_label = np.array([])
        self.store_pred = np.array([])

    def get_name(self):
        return 'F1Score'
