from typing import Any, Iterator
from torch.utils.data.dataloader import DataLoader
from .ava256_dataset import AvaDataLoader, AvaMultiCaptureDataset, AvaSingleCaptureDataset
from .nersemble_dataset import NersembleDataLoader, NersembleMultiCaptureDataset, NersembleSingleCaptureDataset
from match.utils import log_console_util
import torch
import copy
import numpy as np
import bisect

logger = log_console_util.getLogger(__name__)


def get_loader_cls(s:str):
  str_to_cls = dict(ava256=AvaDataLoader,
                    nersemble=NersembleDataLoader,
                    multi=MultiDataLoader,)
  return str_to_cls[s]

def get_dataset_cls(s):
  str_to_cls = dict(ava256=AvaMultiCaptureDataset,
                    nersemble=NersembleMultiCaptureDataset,
                    )
  return str_to_cls[s]


class MultiDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataloader_kwargs, dataset_configs:list, training:bool|None=None):
        # setting kwarg defaults for training / validation dataloaders and datasets
        dataset_configs = copy.deepcopy(dataset_configs)
        for i in range(len(dataset_configs)):
            dataset_configs[i][1]['training'] = dataset_configs[i][1].get('training', training)
        if training is not None and training:
            dataloader_kwargs['shuffle'] = dataloader_kwargs.get('shuffle', True)
        elif not training:  # validation
            dataloader_kwargs['shuffle'] = dataloader_kwargs.get('shuffle', False)
        
        datasets = list()
        for i in range(len(dataset_configs)):
            ds_class_name = dataset_configs[i][0]
            ds_class = get_dataset_cls(ds_class_name)
            datasets.append(ds_class(**dataset_configs[i][1]))
        multi_dataset = MultiDataset(datasets)
        self.n_datasets = len(datasets)

        logger.info(f'MultiDataLoader loaded {len(datasets)} datasets: \n' 
                    + f'\n'.join([f'{dataset_configs[i][0]}: {len(datasets[i])} samples ({len(datasets[i])/len(multi_dataset)*100:.1f}%)' for i in range(len(datasets))]) 
                    + f'\nTotal Length: {len(multi_dataset)}')

        super().__init__(dataset=multi_dataset, **dataloader_kwargs)


class MultiDataset(torch.utils.data.DataLoader):
  def __init__(self, datasets):
    # Dataset lengths
    self.datasets = datasets
    self.cumulative_sizes = np.cumsum([len(x) for x in self.datasets])
    self.total_len = self.cumulative_sizes[-1]
  

  def __getitem__(self, idx: int):
      """Inspired from PyTorch's ConcatDataset"""

      if idx < 0:
          if -idx > len(self):
              raise ValueError("absolute value of index should not exceed dataset length")
          idx = len(self) + idx
      
      dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
      if dataset_idx == 0:
          sample_idx = idx
      else:
          sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

      dataset = self.datasets[dataset_idx]
      sample = dataset[sample_idx]

      if sample is not None:
          sample["dataset_idx"] = dataset_idx

      return sample

  def __len__(self):
     return self.total_len
  

# Copied from https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/loader.py
class MultiEpochsChunkedDataLoader(DataLoader):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self._DataLoader__initialized = False
    if self.batch_sampler is None:
      self.sampler = _RepeatSampler(self.sampler)
    else:
      self.batch_sampler = _RepeatSampler(self.batch_sampler)
    self._DataLoader__initialized = True
    self.iterator = super().__iter__()

  # def __len__(self):
  #     return len(self.sampler) if self.batch_sampler is None else len(self.batch_sampler.sampler)

  def __iter__(self):
    for i in range(len(self)):
      yield next(self.iterator)


class _RepeatSampler:
  """Sampler that repeats forever.

  Args: sampler (Sampler)
  """

  def __init__(self, sampler):
    self.sampler = sampler

  # def __len__(self):
  #     return len(self.sampler)

  def __iter__(self):
    while True:
      yield from iter(self.sampler)


def yield_forever(iterator: Iterator[Any]):
  while True:
    for x in iterator:
      yield x
