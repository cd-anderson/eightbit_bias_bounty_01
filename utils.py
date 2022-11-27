import json
import os
import shutil

import PIL.Image
import numpy as np

DEFAULT_ATTRIBUTES = ['skin_tone', 'gender', 'age']


def disparity_score(cm):
    """
    Calculate the disparity score
    :param cm:
    :return:
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    all_acc = list(cm.diagonal())
    return max(all_acc) - min(all_acc)


def get_score(results):
    """
    Calculate the Bias Buccaneers Image Recognition Challenge score
    :param results:
    :return:
    """
    acc = results['accuracy']
    disp = results['disparity']
    ad = 2 * acc['gender'] * (1 - disp['gender']) + 4 * acc['age'] * (1 - disp['age'] ** 2) + 10 * acc[
        'skin_tone'] * (1 - disp['skin_tone'] ** 5)
    return ad


def eightbit_score(results, weight=2, power=1):
    """
    Calculate the Bias Buccaneers score
    :param results:
    :param weight:
    :param power:
    :return:
    """
    return weight * results['acc'] * (1 - results['disparity'] ** power)


def get_bb_01_submission(title, results):
    """
    Format the submission
    :param title:
    :param results:
    :return:
    """
    return {
        'submission_name': title,
        'score': get_score(results),
        'metrics': results
    }


def check_and_create_folder(path):
    """
    Checks for and creates a folder
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def dump_to_json(src, filename, path):
    """
    Dump the source to json
    :param src:
    :param filename:
    :param path:
    :return:
    """
    with open(os.path.join(path, filename), "w") as f:
        json.dump(src, f, indent=4)


def crop_center(pil_img: PIL.Image.Image, crop_width: int, crop_height: int) -> PIL.Image.Image:
    """
    Crop the center of an image
    :param pil_img:
    :param crop_width:
    :param crop_height:
    :return:
    """
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def crop_max_square(pil_img: PIL.Image.Image) -> PIL.Image.Image:
    """
    Find the maximum square of an image
    :param pil_img:
    :return:
    """
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))


def copy_dataset_item(src, dst, filename, verbose=False):
    """
    Create the destination folder and copies the item if it doesn't exist
    :param src:
    :param dst:
    :param filename:
    :param verbose:
    :return:
    """
    check_and_create_folder(dst)
    target = os.path.join(src, filename)
    final = os.path.join(dst, filename)
    if os.path.isfile(target) and not os.path.isfile(final):
        if verbose:
            print(f'Copying {filename} to {final}')
        shutil.copy(target, final)


def create_labeled_dataset(data_frame, src_path, dst_path, target_attributes=None, combine=False, verbose=False):
    """
    Parse the dataframe and copy to file to the correct folders
    :param data_frame:
    :param src_path:
    :param dst_path:
    :param target_attributes:
    :param combine:
    :param verbose:
    :return:
    """
    if target_attributes is None:
        target_attributes = DEFAULT_ATTRIBUTES

    for index, row in data_frame.iterrows():
        filename = row['name']
        for attribute in target_attributes:
            val = row[attribute]
            if val is None:
                if verbose:
                    print(f'Attribute {attribute} for {filename} is None, skipping...')
                continue
            label_path = os.path.join(dst_path, attribute, val)
            copy_dataset_item(src_path, label_path, filename, verbose)

        if combine:
            # TODO: Combine?
            pass


def create_unlabeled_dataset(data_frame, src_path, dst_path, verbose=False):
    """
    Copy unlabeled data
    :param data_frame:
    :param src_path:
    :param dst_path:
    :param verbose:
    :return:
    """
    for index, row in data_frame.iterrows():
        filename = row['name']
        label_path = os.path.join(dst_path, 'unlabeled', 'unlabeled')
        copy_dataset_item(src_path, label_path, filename)
