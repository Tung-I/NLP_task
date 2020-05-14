import logging
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from src.runner.utils import EpochLog

LOGGER = logging.getLogger(__name__.split('.')[-1])


class BasePredictor:
    """The base class for all predictors.
    Args:
        saved_dir (Path): The root directory of the saved data.
        device (torch.device): The device.
        test_dataloader (Dataloader): The testing dataloader.
        net (BaseNet): The network architecture.
        loss_fns (LossFns): The loss functions.
        loss_weights (LossWeights): The corresponding weights of loss functions.
        metric_fns (MetricFns): The metric functions.
    """

    def __init__(self, saved_dir, device, test_dataloader,
                 net, loss_fns=None, loss_weights=None, metric_fns=None, id_file=None):
        self.saved_dir = saved_dir
        self.device = device
        self.test_dataloader = test_dataloader
        self.net = net.to(device)
        self.loss_fns = loss_fns
        self.loss_weights = loss_weights
        self.metric_fns = metric_fns
        self.id_file = id_file

    def predict(self):
        """The testing process.
        """
        self.net.eval()
        dataloader = self.test_dataloader
        pbar = tqdm(dataloader, desc='test', ascii=True)

        epoch_log = EpochLog()
        prediction = []
        for i, batch in enumerate(pbar):
            b_id, b_mask = batch['inputs'].to(self.device), batch['masks'].to(self.device)
            with torch.no_grad():
                output = self.net(b_id, token_type_ids=None, attention_mask=b_mask)
                logits = output[0]
                logits = logits.detach().cpu().numpy()
                prediction.append(logits)
                # test_dict = self._test_step(batch)
                # loss = test_dict['loss']
                # losses = test_dict.get('losses')
                # metrics = test_dict.get('metrics')

            # if (i + 1) == len(dataloader) and not dataloader.drop_last:
            #     batch_size = len(dataloader.dataset) % dataloader.batch_size
            # else:
            #     batch_size = dataloader.batch_size
            # epoch_log.update(batch_size, loss, losses, metrics)

            # pbar.set_postfix(**epoch_log.on_step_end_log)
        # test_log = epoch_log.on_epoch_end_log
        # LOGGER.info(f'Test log: {test_log}.')
        prediction = np.vstack(prediction)
        # np.save('./'+config.model.name+'_pred.npy',prediction)
        out = open(str(self.saved_dir / Path('ans.csv')), 'w')
        out.write('Index,Gold\n')
        pred = np.argmax(prediction, axis=1)
        test = pd.read_csv(self.id_file, delimiter='\t', header=None, dtype={'id': str,'text':str}, names=['id', 'sentence'])
        test_ids = test.id.values
        for index, p in zip(test_ids, pred):
            out.write("{},{}\n".format(index, p))

    def _test_step(self, batch):
        """The user-defined testing logic.
        Args:
            batch (dict or sequence): A batch of the data.

        Returns:
            test_dict (dict): The computed results.
                test_dict['loss'] (torch.Tensor)
                test_dict['losses'] (dict, optional)
                test_dict['metrics'] (dict, optional)
        """
        raise NotImplementedError

    def load(self, path):
        """Load the model checkpoint.
        Args:
            path (Path): The path to load the model checkpoint.
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.net.load_state_dict(checkpoint['net'])
