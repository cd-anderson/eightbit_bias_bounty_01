import gzip
import io
import math
import os

import torch
from semilearn import get_data_loader
from semilearn.datasets.augmentation import RandomResizedCropAndInterpolation, RandAugment
from semilearn.datasets.cv_datasets.imagenet import ImagenetDataset
from torchvision.transforms import transforms

from data.dataset import DS_MEAN, DS_STD, EightBitBiasBountySemilearnDataset
from trainers.trainer import EightBitBiasBountyTrainer


class EightBitModel:
    algorithm = None
    device = None
    config = None
    trainer = None
    transform_weak = None
    transform_strong = None
    transform_val = None

    def __init__(self):
        self.device = torch.device(f'cuda:{self.config.gpu}' if torch.cuda.is_available() else 'cpu')

    def create_transforms(self, mean=None, std=None):
        """
        Create the transforms used for the dataset
        :param mean:
        :param std:
        :return:
        """
        if mean is None:
            mean = DS_MEAN['bbc01']

        if std is None:
            std = DS_STD['bbc01']

        orig_size = int(math.floor(self.config.img_size / self.config.crop_ratio))

        self.transform_weak = transforms.Compose([
            transforms.Resize((orig_size, orig_size)),
            transforms.RandomCrop((self.config.img_size, self.config.img_size), padding='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(p=0.33),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.transform_strong = transforms.Compose([
            transforms.Resize((orig_size, orig_size)),
            RandomResizedCropAndInterpolation((self.config.img_size, self.config.img_size)),
            transforms.RandomHorizontalFlip(),
            RandAugment(3, 10, exclude_color_aug=False),
            transforms.RandomGrayscale(p=0.33),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.transform_val = transforms.Compose([
            transforms.Resize(orig_size),
            transforms.CenterCrop(self.config.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def create_trainer(self):
        """
        Create the trainer
        :return:
        """
        self.trainer = EightBitBiasBountyTrainer(self.config, self.algorithm)

    def load_model_to_device(self):
        """
        Load model checkpoint to the specified device
        :return:
        """
        if not os.path.isfile(self.config.load_path):
            self.algorithm.print_fn(f'model {self.config.load_path} not found')
            return

        checkpoint = torch.load(self.config.load_path, map_location='cpu')
        self.algorithm.model.load_state_dict(checkpoint['model'])
        self.algorithm.model.to(self.device)
        # self.algorithm.ema_model.load_state_dict(checkpoint['ema_model'])
        # self.algorithm.optimizer.load_state_dict(checkpoint['optimizer'])
        #
        # self.algorithm.scheduler.load_state_dict(checkpoint['scheduler'])
        # self.algorithm.loss_scaler.load_state_dict(checkpoint['loss_scaler'])
        # self.algorithm.it = checkpoint['it']
        # self.algorithm.best_it = checkpoint['best_it']
        # self.algorithm.best_eval_acc = checkpoint['best_eval_acc']
        self.algorithm.print_fn('model loaded')

        return checkpoint

    def get_bbc01_evaluation_dataloader(self, root_path, csv_path, shuffle=True):
        """
        Get a dataloader for an evaluation dataset from the 8-Bit Bias Bounty csv
        :param root_path:
        :param csv_path:
        :param shuffle:
        :return:
        """
        dataset = EightBitBiasBountySemilearnDataset(self.algorithm, root_path, csv_path, shuffle,
                                                     transform=self.transform_val,
                                                     seed=self.config.seed,
                                                     categories=[self.config.eightbit_category])

        return get_data_loader(self.config, dataset, self.config.eval_batch_size, data_sampler=None,
                               num_workers=2*self.config.num_workers, drop_last=False)

    def get_imagenet_evaluation_dataloader(self, root_path):
        """
        Get a dataloader for an evaluation dataset from the imagenet folder structure
        :param root_path:
        :return:
        """
        dataset = ImagenetDataset(root=root_path,
                                  transform=self.transform_val,
                                  ulb=False,
                                  alg=self.algorithm,
                                  strong_transform=None)

        return get_data_loader(self.config, dataset, self.config.eval_batch_size, data_sampler=None,
                               num_workers=self.config.num_workers, drop_last=False)

    def get_bbc01_training_labeled_dataloader(self, root_path, csv_path, shuffle=True):
        """
        Get a dataloader for a labeled training dataset from the 8-Bit Bias Bounty csv
        :param root_path:
        :param csv_path:
        :param shuffle:
        :return:
        """
        dataset = EightBitBiasBountySemilearnDataset(self.algorithm, root_path, csv_path, shuffle,
                                                     transform=self.transform_weak,
                                                     strong_transform=self.transform_strong,
                                                     num_labels=self.config.num_labels,
                                                     seed=self.config.seed,
                                                     categories=[self.config.eightbit_category])

        return get_data_loader(self.config, dataset, self.config.batch_size,
                               data_sampler=self.config.train_sampler,
                               num_iters=self.config.num_train_iter,
                               num_epochs=self.config.epoch,
                               num_workers=self.config.num_workers,
                               distributed=self.config.distributed)

    def get_imagenet_training_labeled_dataloader(self, root_path):
        """
        Get a dataloader for a labeled training dataset from the imagenet folder structure
        :param root_path:
        :return:
        """
        dataset = ImagenetDataset(root=root_path,
                                  transform=self.transform_weak,
                                  ulb=False,
                                  alg=self.algorithm,
                                  num_labels=self.config.num_labels,
                                  strong_transform=self.transform_strong)

        return get_data_loader(self.config, dataset, self.config.batch_size,
                               data_sampler=self.config.train_sampler,
                               num_iters=self.config.num_train_iter,
                               num_epochs=self.config.epoch,
                               num_workers=self.config.num_workers,
                               distributed=self.config.distributed)

    def get_bbc01_training_unlabeled_dataloader(self, root_path, csv_path, shuffle=True, include_lb_with_ulb=False):
        """
        Get a dataloader for a labeled training dataset from the 8-Bit Bias Bounty csv
        :param root_path:
        :param csv_path:
        :param shuffle:
        :param include_lb_with_ulb:
        :return:
        """
        dataset = EightBitBiasBountySemilearnDataset(self.algorithm, root_path, csv_path, shuffle,
                                                     transform=self.transform_weak,
                                                     strong_transform=self.transform_strong,
                                                     num_labels=self.config.ulb_num_labels,
                                                     seed=self.config.seed,
                                                     is_ulb=True,
                                                     include_lb_with_ulb=include_lb_with_ulb,
                                                     categories=[self.config.eightbit_category])

        return get_data_loader(self.config, dataset, int(self.config.batch_size * self.config.uratio),
                               data_sampler=self.config.train_sampler,
                               num_iters=self.config.num_train_iter,
                               num_epochs=self.config.epoch,
                               num_workers=self.config.num_workers,
                               distributed=self.config.distributed)

    def get_imagenet_training_unlabeled_dataloader(self, root_path):
        """
        Get a dataloader for a labeled training dataset from the imagenet folder structure
        :param root_path:
        :return:
        """
        dataset = ImagenetDataset(root=root_path,
                                  transform=self.transform_weak,
                                  ulb=True,
                                  alg=self.algorithm,
                                  num_labels=self.config.ulb_num_labels,
                                  strong_transform=self.transform_strong)

        return get_data_loader(self.config, dataset, int(self.config.batch_size * self.config.uratio),
                               data_sampler=self.config.train_sampler,
                               num_iters=self.config.num_train_iter,
                               num_epochs=self.config.epoch,
                               num_workers=self.config.num_workers,
                               distributed=self.config.distributed)
