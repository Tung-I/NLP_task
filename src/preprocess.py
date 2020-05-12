import numpy as np
import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split
from box import Box
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser(description="The main pipeline script.")
    parser.add_argument('config_path', type=Path, help='The path of the config file.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = _parse_args()

    # config = Box.from_yaml(filename='/home/tony/NLP_task/configs/train/base_config.yaml')
    config = Box.from_yaml(filename=args.config_path)
    root_path = Path(config.dataset.kwargs.data_dir)
    data = pd.read_csv(str(root_path / Path('train.csv')), sep=';', engine='python')
    bert_data = pd.DataFrame({'id': np.arange(len(data)),
                                'label': data['Gold'],
                                'alpha': ['a'] * data.shape[0],
                                'text': data['Text']
    })
    bert_train, bert_val = train_test_split(bert_data,test_size=0.2, stratify=bert_data.label.values)
    bert_train.to_csv(str(root_path / Path('train.tsv')), sep='\t', index=False, header=False)
    bert_val.to_csv(str(root_path / Path('val.tsv')), sep='\t', index=False, header=False)

    test_data = pd.read_csv(str(root_path / Path('test.csv')), sep=';', dtype={'Index': str,'Text':str})
    bert_test = pd.DataFrame({'id': test_data['Index'],
                                'text': test_data['Text']
    })
    bert_test.to_csv(str(root_path / Path('test.tsv')), sep='\t', index=False, header=False)