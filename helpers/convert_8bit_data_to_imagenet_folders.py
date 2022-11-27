import argparse

import pandas as pd

from utils import create_labeled_dataset, create_unlabeled_dataset


def main(args):
    if args.verbose:
        print(f'Starting with arguments:\r\n{args}')
    df = pd.read_csv(args.csv, usecols=['name'].extend(args.cat))
    labeled_df = df.dropna()
    create_labeled_dataset(labeled_df, args.src, args.dst, target_attributes=args.cat, verbose=args.verbose)
    create_unlabeled_dataset(pd.merge(df, labeled_df, indicator=True, how='outer').query(
        '_merge=="left_only"').drop('_merge', axis=1), args.src, args.dst, verbose=args.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True, default='./data/data_bb1_img_recognition/train')
    parser.add_argument('--csv', type=str, required=True, default='./data/data_bb1_img_recognition/train/labels.csv')
    parser.add_argument('--dst', type=str, required=True, default='./data/semilearn-imagenet/train')
    parser.add_argument('--cat', type=str, nargs='+', required=False, default=['skin_tone', 'gender', 'age'])
    parser.add_argument('--combine', type=bool, required=False, default=False)
    parser.add_argument('--verbose', type=bool, required=False, default=True)
    main(parser.parse_args())
