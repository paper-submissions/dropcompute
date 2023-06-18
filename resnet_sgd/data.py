"""Data loading module. taken from Habana -
https://github.com/HabanaAI/Model-References/blob/master/PyTorch/computer_vision/classification/torchvision/data_loaders.py
"""


import os
import hashlib
import glob
import pathlib
import torch.utils.data
import time
import torchvision
from torchvision import transforms
import habana_utils
import logging

import habana_dataloader


def _get_cache_path(filepath):
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def get_data_loader(dataset, dl_worker_type, **kwargs):
    if dl_worker_type == 'HABANA':
        data_loader = habana_dataloader.HabanaDataLoader(dataset, **kwargs)
    elif dl_worker_type == 'MP':
        data_loader = torch.utils.data.DataLoader(dataset, **kwargs)
    else:
        raise ValueError(f'Supported dl_worker_type are [HABANA, MP]. Provided'
                         f' dl_worker_type={dl_worker_type}')
    return data_loader


def prepare_dataset_manifest(args):
    if args.data_path is not None:
        # get files list
        dataset_dir = os.path.join(args.data_path, 'train')
        logging.info(f'dataset dir: {dataset_dir}')
        manifest_data = {
            'file_list':
                sorted(glob.glob(dataset_dir + "/*/*.{}".format('JPEG')))
        }
        # get class list
        data_dir = pathlib.Path(dataset_dir)
        manifest_data['class_list'] = sorted(
            [item.name for item in data_dir.glob('*') if item.is_dir()])
        file_sizes = {}
        for filename in manifest_data['file_list']:
            file_sizes[filename] = os.stat(filename).st_size
        manifest_data['file_sizes'] = file_sizes
        return manifest_data


def load_data(train_dir: str, val_dir: str, args, manifest=None):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset_transforms = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    dataset_test_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    if args.dl_worker_type == 'HABANA':
        dataset_loader_train = (
            habana_dataloader.habana_dataset.ImageFolderWithManifest)
    else:
        dataset_loader_train = torchvision.datasets.ImageFolder
    dataset_loader_eval = torchvision.datasets.ImageFolder
    loader_params = {'root': train_dir,
                     'transform': dataset_transforms}
    loader_test_params = {'root': val_dir,
                          'transform': dataset_test_transforms}
    if args.dl_worker_type == 'HABANA':
        if manifest is None:
            manifest = prepare_dataset_manifest(args)
        loader_params['manifest'] = manifest

    def _load_data(data_dir, cache_dataset, dataset_loader, data_loader_params):
        cache_path = _get_cache_path(data_dir)
        if cache_dataset and os.path.exists(cache_path):
            # Attention, as the transforms are also cached!
            logging.info(f'Loading dataset_train from {cache_path}')
            dataset, _ = torch.load(cache_path)
        else:
            dataset = dataset_loader(**data_loader_params)
            if cache_dataset:
                logging.info(f'Saving dataset_train to {cache_path}')
                habana_utils.mkdir(os.path.dirname(cache_path))
                habana_utils.save_on_master((dataset, train_dir), cache_path)
        return dataset

    logging.info('Loading training data')
    st = time.time()
    dataset_train = _load_data(
        train_dir, args.cache_dataset, dataset_loader_train, loader_params)
    logging.info(f'Took {time.time() - st}')
    logging.info('Loading validation data')
    dataset_test = _load_data(
        val_dir, args.cache_dataset, dataset_loader_eval, loader_test_params)
    logging.info('Creating samplers')
    if args.distributed:
        train_sampler = (
            torch.utils.data.distributed.DistributedSampler(dataset_train))
        test_sampler = (
            torch.utils.data.distributed.DistributedSampler(dataset_test))
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    return dataset_train, dataset_test, train_sampler, test_sampler
