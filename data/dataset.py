import itertools
import os
from operator import itemgetter

import PIL
import pandas as pd
from semilearn import BasicDataset
from semilearn.datasets.utils import get_onehot

DS_MEAN, DS_STD = {}, {}
DS_MEAN['bbc01'] = [0.47138864, 0.39088012, 0.34485637]
DS_STD['bbc01'] = [0.27880699, 0.25060371, 0.24824424]


class EightBitBiasBountySemilearnDataset(BasicDataset):
    """
    EightBitBiasBountyDataset returns a pair of image and labels from the specified csv and categories
    """

    default_classes = {'gender': ['female', 'male'],
                       'age': ['0_17', '18_30', '31_60', '61_100'],
                       'skin_tone': ['monk_1', 'monk_2', 'monk_3', 'monk_4', 'monk_5', 'monk_6', 'monk_7', 'monk_8',
                                     'monk_9', 'monk_10']}

    def __init__(self, alg, root_path, csv_path, data, shuffle=False, transform=None, strong_transform=None,
                 is_ulb=False, num_labels=-1, categories=None, seed=42, include_lb_with_ulb=False, *args, **kwargs):

        # todo: refactor
        super().__init__(alg, data, transform, is_ulb, strong_transform, *args, **kwargs)
        self.alg = alg
        self.is_ulb = is_ulb
        self.num_labels = num_labels
        self.transform = transform
        self.root = root_path
        self.csv_path = csv_path
        self.categories = categories or ['skin_tone', 'gender', 'age']
        self.data_frame = pd.read_csv(csv_path, usecols=['name'].extend(self.categories))
        self.include_lb_with_ulb = include_lb_with_ulb

        cats = sorted(itemgetter(*self.categories)(self.default_classes))
        if len(cats) > 1 and isinstance(cats[0], list):
            cats = sorted(list(itertools.product(*cats)))
        self.class_list = cats
        self.num_classes = len(self.class_list)
        self.onehot = False

        # remove unlabeled data from labeled dataset
        if not self.is_ulb:
            self.data_frame = self.data_frame.dropna()
        if self.is_ulb and not self.include_lb_with_ulb:
            self.data_frame = pd.merge(self.data_frame, self.data_frame.dropna(), indicator=True, how='outer').query(
                '_merge=="left_only"').drop('_merge', axis=1)

        # shuffle the data
        if shuffle:
            self.data_frame = self.data_frame.sample(frac=1.0, axis=1, random_state=seed).reset_index(drop=True)

        # set the transformations
        self.transform = transform
        self.strong_transform = strong_transform
        if self.strong_transform is None:
            if self.is_ulb:
                assert self.alg not in ['fullysupervised', 'supervised', 'pseudolabel', 'vat', 'pimodel', 'meanteacher',
                                        'mixmatch'], f"alg {self.alg} requires strong augmentation"

    def __len__(self):
        return len(self.data_frame)

    def __sample__(self, idx):
        # get the file path and build the target label
        file = self.data_frame.iloc[idx]['name']
        label = itemgetter(*self.categories)(self.data_frame.iloc[idx])
        target_ = None if self.is_ulb else self.class_list.index(label)
        target = target_ if not self.onehot else get_onehot(self.num_classes, target_)

        img_path = os.path.join(self.root, file)
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f).convert('RGB')

        return img, target
