from semilearn import get_algorithm, get_net_builder, get_config

from models.model import EightBitModel


class EightBitGenderModel(EightBitModel):
    config = get_config({
        'algorithm': 'fixmatch',
        'net': 'vit_small_patch16_224',
        'net_from_name': False,
        'use_pretrain': True,
        'pretrain_path': 'https://github.com/microsoft/Semi-supervised-learning/releases/download/v.0.0.0/vit_small_patch16_224_mlp_im_1k_224.pth',

        'save_dir': './saved_models',
        'save_name': 'bbc01_gender_vit_small_patch16_224',
        'load_path': './saved_models/bbc01_gender_vit_small_patch16_224/model_best_eightbit.pth',
        'resume': True,
        'use_tensorboard': True,
        'train_sampler': 'RandomSampler',

        # optimization configs
        'epoch': 75,  # set to 100
        'num_train_iter': 204800,  # set to 102400
        'num_log_iter': 512,  # set to 1024
        'num_eval_iter': 2048,  # set to 1024
        'batch_size': 16,
        'eval_batch_size': 32,
        'num_warmup_iter': 5120,
        'ema_m': 0.0,
        'p_cutoff': 0.95,
        'optim': 'AdamW',
        'lr': 0.0001,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'layer_decay': 0.75,
        'amp': False,
        'clip': 0.0,
        'use_cat': True,

        # dataset configs
        'dataset': 'imagenet',  # TODO: breaks with none, submit patch
        'num_labels': -1,
        'num_classes': 2,
        'ulb_num_labels': -1,
        'img_size': 224,
        'crop_ratio': 0.875,
        'data_dir': './data',

        # algorithm specific configs
        'hard_label': True,
        'uratio': 1,
        'ulb_loss_ratio': 1.0,

        # device configs
        'gpu': 0,
        'world_size': 1,
        'distributed': False,
        'multiprocessing_distributed': False,
        'num_workers': 4,
        'seed': 42,
        'eightbit_category': 'gender'
    })

    def __init__(self):
        super().__init__()
        self.algorithm = get_algorithm(self.config, get_net_builder(self.config.net, from_name=False), tb_log=None,
                                       logger=None)
        self.load_model_to_device()
        self.create_transforms()
        self.create_trainer()

    def evaluate(self, data_loader, use_ema_model=False):
        return self.trainer.evaluate(data_loader, use_ema_model)

    def predict(self, data_loader, use_ema_model=False, return_gt=False):
        return self.trainer.predict(data_loader, use_ema_model, return_gt)

    def fit(self, train_lb_loader, train_ulb_loader, eval_loader):
        return self.trainer.fit(train_lb_loader, train_ulb_loader, eval_loader)
