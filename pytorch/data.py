import os
import time
import pickle
import csv
import torch
import torchaudio
import tqdm
import numpy as np
import pandas as pd
from abc import abstractmethod
from typing import Tuple, Optional, Union, List, Callable, Any
from torch import Tensor
from torch.nn.functional import one_hot

from pytorch.utils import fix_len, load_wav, create_task_label
from src.manifolds import set_logger, make_folder, float32_to_int16, int16_to_float32
from src.esc_meta import esc_hierarchy

# Set the backend for audio processing
torchaudio.set_audio_backend("sox_io")


class NaiveDataset(object):
    """" An abstract dataset class"""
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass


class ESC50(NaiveDataset):
    """ ESC-50 dataset."""
    def __init__(
            self,
            wav_dir: str,
            csv_path: str,
            fold: list,
            num_class: int = 50,
            sample_rate: int = 44100
    ) -> None:
        self.wav_dir = wav_dir
        self.num_class = num_class
        self.sample_rate = sample_rate
        self.meta = self._load_meta(csv_path, fold)  # {filename: child_id}
        assert len(self.meta) > 0
        self.pid = self._build_hierarchy(csv_path)  # {id: pid}
        self.indices = list(self.meta.keys())  # create indices for filename

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: Union[str, int, tuple]) -> Tuple[Tensor, Tensor]:
        if isinstance(item, tuple):
            fname, label = item
            if not isinstance(label, Tensor):
                label = torch.tensor(label)
        else:
            fname = self.indices[item] if isinstance(item, int) else item
            label = torch.tensor(self.meta[fname])
        return load_wav(os.path.join(self.wav_dir, fname), sr=self.sample_rate), label

    def _load_meta(self, csv_path: str, fold: list) -> dict:
        """ Load meta info in line with cross-validation
            Args:
                csv_path: str, path to esc50.csv
                fold: list, fold id(s) needed for train/val dataset
        """
        meta = {}
        with open(csv_path, 'r') as f:  # esc50.csv organise the meta in the terms of
            rows = csv.DictReader(f)  # ['filename', 'fold', 'target', 'category', 'esc10', 'src_file', 'take']
            for r in rows:
                if int(r['fold']) in fold:
                    meta[r['filename']] = int(r['target'])  # convert to int idx from str type from csv file

        return meta  # format = {filename: (child_id, parent_id)}

    def _build_hierarchy(self, csv_path: str):
        _, _, vocabulary = self.create_vocabulary(csv_path)
        map = dict()  # {cid: pid}
        for value in vocabulary.values():
            map[value['id']] = value['pid']

        return map

    @classmethod
    def create_vocabulary(cls, csv_path: str):
        """ Create vocabulary sets needs for few-shot learning.
        :param csv_path: path/to/dir/meta/esc50.csv
        :return: voc = {'class_name': class_id},
                 parent_voc = {'parent_class_name': parent_class_id}
                 vocabulary = {'class_name': {'pid', 'id'}}
        """
        voc = dict()
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)

            for d in reader:
                if d['target'] not in voc.keys():
                    voc[d['category']] = int(d['target'])

        parent_voc = dict()
        for idx, item in enumerate(esc_hierarchy.keys()):
            parent_voc[item] = idx

        vocabulary = dict()
        for key, value in voc.items():
            for p_cat in parent_voc.keys():
                if key in esc_hierarchy[p_cat]:
                    pid = parent_voc[p_cat]
                vocabulary[key] = {'pid': pid, 'id': value}
        return voc, parent_voc, vocabulary


class NaiveSampler(object):
    """" An abstract dataset class"""
    def __len__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass


class SimpleSampler(NaiveSampler):
    """ Sampler for classical learning."""
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, fix_class=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.fix_class = fix_class

    def __len__(self):
        if self.fix_class:
            slt_ratio = len(
                self.fix_class) / self.dataset.num_class  # calculate the ratio of samples in fixed_class to all
        else:
            slt_ratio = 1

        if self.drop_last:
            return int(len(self.dataset) * slt_ratio // self.batch_size)
        else:
            return int((len(self.dataset) + self.batch_size - 1) * slt_ratio // self.batch_size)

    def __iter__(self):
        if self.fix_class:
            # Sample an instance if its label belongs to the `fix_class`
            indices = list()
            for key, value in self.dataset.meta.items():
                if value in self.fix_class:
                    indices.append(key)
        else:
            indices = self.dataset.indices

        if self.shuffle:
            indices = np.random.permutation(indices)

        batch = []
        for fname in indices:
            batch.append(fname)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch


# TODO： batch_size
class ESC50FewShotSampler(NaiveSampler):
    """ Sampler for few shot learning.
        Args:
            dataset: data source, e.g. instance of torch.data.Dataset, data[item] = {filename: labels}
            num_nvl_cls: number of novel classes in an episode (i.e., 'n' ways)
            num_sample_per_cls: number of support samples in each novel class (i.e., 'k' shot)
            num_queries_per_cls: number of queries for each novel class
            num_task: total number of tasks (or episodes) in one epoch
            batch_size: number of tasks (or episodes) in one batch
            fix_split: class indices from which samples would be extracted
            require_pid: return pid along with its id if true. Default is false.
    """
    def __init__(
            self,
            dataset: torch.nn.Module,
            num_nvl_cls: int,
            num_sample_per_cls: int,
            num_queries_per_cls: int,
            num_task: int,
            batch_size: int,
            fix_split: list = None,
            require_pid: bool = False,
            **kwargs
    ) -> None:
        if (batch_size != 1) and require_pid:
            raise ValueError("`batch_size` is expected to be 1 when `require_pid` is True")

        self.dataset = dataset
        self.num_nvl_cls = num_nvl_cls
        self.num_sample_per_cls = num_sample_per_cls
        self.num_queries_per_cls = num_queries_per_cls
        self.num_task = num_task
        self.batch_size = batch_size
        self.fix_split = None
        self.require_pid = require_pid

        if fix_split != None and len(fix_split) > num_nvl_cls:
            self.fix_split = fix_split
        else:
            raise ValueError(f"Make sure `fix_split` contains enough novel classes.")

    def __len__(self):
        """ we assume user can set both params properly so that there won't be reminder at the end."""
        return self.num_task // self.batch_size

    def __iter__(self):
        num_batch = self.num_task // self.batch_size
        for _ in range(num_batch):
            batch = []
            for task in range(self.batch_size):
                if not self.fix_split == None:
                    cls_set = self.fix_split
                else:
                    # Randomly select `num_nvl_cls` novel classes
                    cls_set = list(set(list(self.dataset.meta.values())))
                nvl_cls_ids = np.random.choice(cls_set, size=self.num_nvl_cls, replace=False)
                # Get a subset of metadata containing novel classes only
                subset = {'fname': [], 'label': []}
                for fname, cid in self.dataset.meta.items():
                    if cid in nvl_cls_ids:
                        # `label` = (cid, pid) if `require_pid` is true
                        subset['fname'].append(fname)
                        subset['label'].append(cid)

                subset['fname'] = np.stack(subset['fname'])
                subset['label'] = np.stack(subset['label'])

                # Get ids of support samples
                support_samples = dict()
                for n in nvl_cls_ids:
                    # randomly select 'num_sample_per_cls' support samples from one novel class
                    slt_cls = subset['fname'][subset['label'] == n]
                    samples = np.random.choice(slt_cls, size=self.num_sample_per_cls, replace=False)
                    batch.extend([str(x) for x in list(samples)])

                    support_samples[n] = samples

                # Get ids of query samples
                for n in nvl_cls_ids:
                    # select 'num_queries_per_cls' queries from one novel class
                    # there must be no overlaps (samples) between queries and support samples
                    slt_cls = subset['fname'][subset['label'] == n]
                    samples = np.random.choice(slt_cls[np.isin(slt_cls, support_samples[n], invert=True)],
                                               size=self.num_queries_per_cls, replace=False)
                    batch.extend([str(x) for x in list(samples)])  # match the input format in the class ESC50

                # Create task-specific labels
                support_cids = create_task_label(self.num_nvl_cls, self.num_sample_per_cls)
                query_cids = create_task_label(self.num_nvl_cls, self.num_queries_per_cls)
                batch_y = torch.cat([support_cids, query_cids], dim=0)

                if self.require_pid:
                    _pid, query_pids = list(), list()
                    for n in nvl_cls_ids:
                        pid = self.dataset.pid[n]
                        _pid.append(torch.tensor([pid] * self.num_sample_per_cls))
                        query_pids.append(torch.tensor([pid] * self.num_queries_per_cls))

                    _pid.extend(query_pids)
                    _pid = torch.cat(_pid, dim=0)

                    # Force the label pid from [0, n)
                    _, batch_pid = torch.unique(_pid, sorted=False, return_inverse=True)

                    batch_y = list(zip(batch_y, batch_pid))

            yield list(zip(batch, batch_y))


if __name__ == '__main__':
    """ This is a test module."""
    pass