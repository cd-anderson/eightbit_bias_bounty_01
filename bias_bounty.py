import argparse
from datetime import datetime
from pprint import pprint

from semilearn.core.utils import get_logger
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

from models.age import EightBitAgeModel
from models.gender import EightBitGenderModel
from models.skintone import EightBitSkintoneModel
from utils import dump_to_json, disparity_score, get_bb_01_submission


class EightBitBiasBounty:

    def __init__(self, logfile='log.txt', logpath='.'):
        self.logger = get_logger(logfile, save_path=logpath, level="INFO")
        self.age_model = EightBitAgeModel()
        self.gender_model = EightBitGenderModel()
        self.skintone_model = EightBitSkintoneModel()

    def _evaluate_model(self, model, data_loader, use_ema_model=False):
        """
        Evaluate the data with the associated model
        :param model:
        :param data_loader:
        :param use_ema_model:
        :return:
        """
        y_predicted, y_logits, y_true = model.predict(data_loader, use_ema_model, return_gt=True)
        accuracy = accuracy_score(y_true, y_predicted)
        precision = precision_score(y_true, y_predicted, average='macro')
        recall = recall_score(y_true, y_predicted, average='macro')
        f1 = f1_score(y_true, y_predicted, average='macro')
        cm = confusion_matrix(y_true, y_predicted, normalize='true')
        disparity = disparity_score(cm)
        results = {'model': model.config.eightbit_category, 'accuracy': accuracy, 'precision': precision,
                   'recall': recall, 'f1': f1, 'disparity': disparity}

        for key, item in results.items():
            self.logger.info(f'{key}: {item}')

        return results

    def evaluate_bbc01_data(self, root_path, csv_path, shuffle=True):
        """
        Evaluate performance on an 8-Bit Bias Bounty dataset
        :param root_path:
        :param csv_path:
        :param shuffle:
        :return:
        """
        self.logger.info("Evaluating 8-Bit Bias Bounty dataset")

        age_results = self._evaluate_model(self.age_model,
                                           self.age_model.get_bbc01_evaluation_dataloader(root_path, csv_path, shuffle))

        gender_results = self._evaluate_model(self.gender_model,
                                              self.gender_model.get_bbc01_evaluation_dataloader(root_path, csv_path,
                                                                                                shuffle))
        skintone_results = self._evaluate_model(self.skintone_model,
                                                self.skintone_model.get_bbc01_evaluation_dataloader(root_path, csv_path,
                                                                                                    shuffle))

        return {'accuracy': {'skin_tone': skintone_results['accuracy'], 'gender': gender_results['accuracy'],
                             'age': age_results['accuracy']},
                'disparity': {'skin_tone': skintone_results['disparity'], 'gender': gender_results['disparity'],
                              'age': age_results['disparity']}}


def main(args):
    """
    8-Bit Bias Bounty (2022Q4 Sumission)
    :param args:
    :return:
    """
    bias_bounty = EightBitBiasBounty()
    results = bias_bounty.evaluate_bbc01_data(args.img_path, args.csv_path)
    submission = get_bb_01_submission(f'UNFiT', results)
    print("Creating submission. Yo ho, yo ho, a pirate's life for me.")
    dump_to_json(submission, 'submission.json', '.')
    pprint(submission)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=False, default='./data/data_bb1_img_recognition/test')
    parser.add_argument('--csv_path', type=str, required=False, default='./data/data_bb1_img_recognition/test/labels.csv')
    main(parser.parse_args())
