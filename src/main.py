import argparse
import copy
import logging
import random
import torch
import transformers
from box import Box
from collections import namedtuple
from pathlib import Path
from shutil import copyfile
from transformers import get_linear_schedule_with_warmup

import src


def main(args):
    logging.info(f'Load the config from "{args.config_path}".')
    config = Box.from_yaml(filename=args.config_path)
    saved_dir = Path(config.main.saved_dir)
    if not saved_dir.is_dir():
        saved_dir.mkdir(parents=True)

    logging.info(f'Save the config to "{saved_dir}".')
    copyfile(args.config_path, saved_dir / 'config.yaml')

    if not args.test:

        random_seed = config.main.get('random_seed')
        if random_seed is None:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            logging.info('Make the experiment results deterministic.')
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        logging.info('Create the device.')
        if 'cuda' in config.trainer.kwargs.device and not torch.cuda.is_available():
            raise ValueError("The cuda is not available. Please set the device to 'cpu'.")
        device = torch.device(config.trainer.kwargs.device)

        logging.info('Create the training and validation datasets.')
        config.dataset.setdefault('kwargs', {}).update(pretrain_weight=config.net.kwargs.pretrain_weight)
        config.dataset.setdefault('kwargs', {}).update(type_='train')
        train_dataset = _get_instance(src.data.datasets, config.dataset)
        config.dataset.setdefault('kwargs', {}).update(type_='valid')
        valid_dataset = _get_instance(src.data.datasets, config.dataset)

        logging.info('Create the training and validation dataloaders.')
        train_kwargs = config.dataloader.kwargs.pop('train', {})
        valid_kwargs = config.dataloader.kwargs.pop('valid', {})
        config_dataloader = copy.deepcopy(config.dataloader)
        config_dataloader.kwargs.update(train_kwargs)
        train_dataloader = _get_instance(src.data.dataloader, config_dataloader, train_dataset)
        config_dataloader = copy.deepcopy(config.dataloader)
        config_dataloader.kwargs.update(valid_kwargs)
        valid_dataloader = _get_instance(src.data.dataloader, config_dataloader, valid_dataset)

        logging.info('Create the network architecture.')
        net_class = getattr(transformers, config.net.name)
        net = net_class.from_pretrained(config.net.kwargs.pretrain_weight, num_labels=config.net.kwargs.num_labels, 
                          output_attentions=config.net.kwargs.output_attentions, output_hidden_states=config.net.kwargs.output_hidden_states)
        net = net.to(device)

        logging.info('Create the loss functions and corresponding weights.')
        loss_fns, loss_weights = [], []
        defaulted_loss_fns = [loss_fn for loss_fn in dir(torch.nn) if 'Loss' in loss_fn]
        for config_loss in config.losses:
            if config_loss.name in defaulted_loss_fns:
                loss_fn = _get_instance(torch.nn, config_loss)
            else:
                loss_fn = _get_instance(src.model.losses, config_loss)
            loss_fns.append(loss_fn)
            loss_weights.append(config_loss.weight)

        logging.info('Create the metric functions.')
        metric_fns = [_get_instance(src.model.metrics, config_metric) for config_metric in config.metrics]

        logging.info('Create the optimizer.')
        optimizer = _get_instance(torch.optim, config.optimizer, net.parameters())

        total_steps = len(train_dataloader) * config.trainer.kwargs.num_epochs
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        logging.info('Create the logger.')
        config.logger.setdefault('kwargs', {}).update(log_dir=saved_dir / 'log',
                                                      net=net)
        logger = _get_instance(src.callbacks.loggers, config.logger)

        logging.info('Create the monitor.')
        config.monitor.setdefault('kwargs', {}).update(checkpoints_dir=saved_dir / 'checkpoints')
        monitor = _get_instance(src.callbacks.monitor, config.monitor)

        logging.info('Create the trainer.')
        kwargs = {
            'saved_dir': saved_dir,
            'device': device,
            'train_dataloader': train_dataloader,
            'valid_dataloader': valid_dataloader,
            'net': net,
            'loss_fns': loss_fns,
            'loss_weights': loss_weights,
            'metric_fns': metric_fns,
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'logger': logger,
            'monitor': monitor
        }
        config.trainer.kwargs.update(kwargs)
        trainer = _get_instance(src.runner.trainers, config.trainer)

        loaded_path = config.main.get('loaded_path')
        if loaded_path is None:
            logging.info('Start training.')
        else:
            logging.info(f'Load the previous checkpoint from "{loaded_path}".')
            trainer.load(Path(loaded_path))
            logging.info('Resume training.')
        trainer.train()
        logging.info('End training.')
    else:
        logging.info('Create the device.')
        if 'cuda' in config.predictor.kwargs.device and not torch.cuda.is_available():
            raise ValueError("The cuda is not available. Please set the device to 'cpu'.")
        device = torch.device(config.predictor.kwargs.device)

        logging.info('Create the testing dataset.')
        config.dataset.setdefault('kwargs', {}).update(type_='test')
        test_dataset = _get_instance(src.data.datasets, config.dataset)

        logging.info('Create the testing dataloader.')
        cls = getattr(src.data.datasets, config.dataset.name)
        sampler = getattr(cls, 'sampler', None)
        batch_sampler = getattr(cls, 'batch_sampler', None)
        collate_fn = getattr(cls, 'collate_fn', None)
        worker_init_fn = getattr(cls, 'worker_init_fn', None)
        config.dataloader.setdefault('kwargs', {}).update(sampler=sampler,
                                                          batch_sampler=batch_sampler,
                                                          collate_fn=collate_fn,
                                                          worker_init_fn=worker_init_fn)
        test_dataloader = _get_instance(src.data.dataloader, config.dataloader, test_dataset)

        logging.info('Create the network architecture.')
        net = _get_instance(src.model.nets, config.net).to(device)

        logging.info('Create the loss functions and corresponding weights.')
        loss_names, loss_fns, loss_weights = [], [], []
        defaulted_loss_fns = tuple(loss_fn for loss_fn in dir(torch.nn) if 'Loss' in loss_fn)
        for config_loss in config.losses:
            loss_name = config_loss.get('alias', config_loss.name)
            if config_loss.name in defaulted_loss_fns:
                loss_fn = _get_instance(torch.nn, config_loss).to(device)
            else:
                loss_fn = _get_instance(src.model.losses, config_loss).to(device)
            loss_weight = config_loss.get('weight', 1 / len(config.losses))
            loss_names.append(loss_name)
            loss_fns.append(loss_fn)
            loss_weights.append(loss_weight)
        LossFns, LossWeights = namedtuple('LossFns', loss_names), namedtuple('LossWeights', loss_names)
        loss_fns, loss_weights = LossFns(*loss_fns), LossWeights(*loss_weights)

        if 'metrics' in config:
            logging.info('Create the metric functions.')
            metric_names, metric_fns = [], []
            defaulted_metric_fns = tuple(metric_fn for metric_fn in dir(torch.nn) if 'Loss' in metric_fn)
            for config_metric in config.metrics:
                metric_name = config_metric.get('alias', config_metric.name)
                if config_metric.name in defaulted_metric_fns:
                    metric_fn = _get_instance(torch.nn, config_metric).to(device)
                else:
                    metric_fn = _get_instance(src.model.metrics, config_metric).to(device)
                metric_names.append(metric_name)
                metric_fns.append(metric_fn)
            MetricFns = namedtuple('MetricFns', metric_names)
            metric_fns = MetricFns(*metric_fns)
        else:
            logging.info('Not using the metric functions.')
            metric_fns = None

        logging.info('Create the predictor.')
        kwargs = {
            'saved_dir': saved_dir,
            'device': device,
            'test_dataloader': test_dataloader,
            'net': net,
            'loss_fns': loss_fns,
            'loss_weights': loss_weights,
            'metric_fns': metric_fns
        }
        config.predictor.kwargs.update(kwargs)
        predictor = _get_instance(src.runner.predictors, config.predictor)

        loaded_path = config.main.loaded_path
        logging.info(f'Load the previous checkpoint from "{loaded_path}".')
        predictor.load(Path(loaded_path))
        logging.info('Start testing.')
        predictor.predict()
        logging.info('End testing.')


def _parse_args():
    parser = argparse.ArgumentParser(description="The main pipeline script.")
    parser.add_argument('config_path', type=Path, help='The path of the config file.')
    parser.add_argument('--test', action='store_true',
                        help='Perform testing if specified; otherwise perform training.')
    args = parser.parse_args()
    return args


def _get_instance(module, config, *args):
    """
    Args:
        module (MyClass): The defined module (class).
        config (Box): The config to create the class instance.

    Returns:
        instance (MyClass): The defined class instance.
    """
    cls = getattr(module, config.name)
    return cls(*args, **config.get('kwargs', {}))


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(name)-16s | %(levelname)-8s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
