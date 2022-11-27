# 8-Bit Bias Bounty (2022Q4 Submission)
Avast ye! Here be a repository for submission to the [8-bit bias bounty](https://biasbounty.ai/8-bbb).

Based on [PyTorch](https://pytorch.org/) vision transformers and [semilearn](https://github.com/microsoft/Semi-supervised-learning) we applied Semi-Supervised Learning to this challenge.

## Prerequisites
```bash
pip install -r requirements.txt
```

### Checkpoints
Download the complete code with PyTorch checkpoints or just the checkpoint files from the GitHub [releases](https://github.com/cd-anderson/eightbit_bias_bounty_01/releases) page. 

> If downloading just the checkpoint files, extract them to the ``./save_models`` directory in the cloned repository.

### Evaluation Data
Extract the training and test folders from the data_bb1_img_recognition dataset into the ``./data/data_bb1_img_recognition directory`` or follow the usage guidelines below to specify the images and CSV file to evaluate.

## Usage
[bias_bounty.py](bias_bounty.py) accepts two optional command line arguments ``--img_path``, the path to the images and ``--csv_path`` which is the path to the CSV file.
```bash
python3 bias_bounty.py  --img_path ./data/data_bb1_img_recognition/test --csv_path ./data/data_bb1_img_recognition/test/labels.csv
```

## Docker
We also include the CUDA 11.7 enable Docker container used for testing and development.

### Build the docker image
```bash
docker build -t eightbit .
```

### Run the docker image
```bash
docker run -v "$(pwd):/eightbit_bias_bounty" --gpus=all --shm-size 16G -it eightbit bash
```

### Example Output
```bash
root@c9b54cf93449:/# cd eightbit_bias_bounty/
root@c9b54cf93449:/eightbit_bias_bounty# python3 bias_bounty.py
unlabeled data number: 1, labeled data number 1
Create train and test data loaders
[!] data loader keys: dict_keys(['train_lb', 'train_ulb', 'eval'])
Create optimizer and scheduler
model loaded
unlabeled data number: 1, labeled data number 1
Create train and test data loaders
[!] data loader keys: dict_keys(['train_lb', 'train_ulb', 'eval'])
Create optimizer and scheduler
model loaded
unlabeled data number: 1, labeled data number 1
Create train and test data loaders
[!] data loader keys: dict_keys(['train_lb', 'train_ulb', 'eval'])
Create optimizer and scheduler
model loaded
[2022-11-27 03:17:31,868 INFO] Evaluating 8-Bit Bias Bounty dataset
[2022-11-27 03:17:41,858 INFO] model: age
[2022-11-27 03:17:41,859 INFO] accuracy: 0.62
[2022-11-27 03:17:41,859 INFO] precision: 0.5906925495714132
[2022-11-27 03:17:41,860 INFO] recall: 0.5844506766034172
[2022-11-27 03:17:41,860 INFO] f1: 0.5837521955968761
[2022-11-27 03:17:41,860 INFO] disparity: 0.1706698174248178
[2022-11-27 03:17:49,365 INFO] model: gender
[2022-11-27 03:17:49,365 INFO] accuracy: 0.852
[2022-11-27 03:17:49,366 INFO] precision: 0.84423507606765
[2022-11-27 03:17:49,366 INFO] recall: 0.8543989957108484
[2022-11-27 03:17:49,367 INFO] f1: 0.8477015279241307
[2022-11-27 03:17:49,367 INFO] disparity: 0.022847578198556318
[2022-11-27 03:17:54,425 INFO] model: skin_tone
[2022-11-27 03:17:54,426 INFO] accuracy: 0.2946666666666667
[2022-11-27 03:17:54,426 INFO] precision: 0.29642222863735257
[2022-11-27 03:17:54,426 INFO] recall: 0.2600814417440575
[2022-11-27 03:17:54,427 INFO] f1: 0.26862604164730447
[2022-11-27 03:17:54,427 INFO] disparity: 0.21712538226299696
Creating submission. Yo ho, yo ho, a pirate's life for me.
{'metrics': {'accuracy': {'age': 0.62,
                          'gender': 0.852,
                          'skin_tone': 0.2946666666666667},
             'disparity': {'age': 0.1706698174248178,
                           'gender': 0.022847578198556318,
                           'skin_tone': 0.21712538226299696}},
 'score': 7.018074541795171,
 'submission_name': 'UNFiT'}
```
