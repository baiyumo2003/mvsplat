from yacs.config import CfgNode as CN


class ConfigRttp:
    def __init__(self):
        self.cfg = CN()
        # self.cfg.name = ''
        # self.cfg.stage1_ckpt = None
        # self.cfg.restore_ckpt = None
        # self.cfg.lr = 0.0
        # self.cfg.wdecay = 0.0
        self.cfg.batch_size = 0
        # self.cfg.num_steps = 0
        # self.cfg.loss_scale = 1
        # self.cfg.use_wandb = False
        # self.cfg.use_lpips = False

        self.cfg.dataset = CN()
        self.cfg.dataset.data_root = ''
        self.cfg.dataset.source_cam_names = [['000228212212', "middle"],
                                             ['000655594512', "right"],
                                             ['000665414412', "up"],
                                             ['000738314412', "left"]]
        self.cfg.dataset.gt_cam_names = [['000597713512', "gt_left"],
                                         ['000907513512', "gt_right"]]
        self.cfg.dataset.downscale_factor = 2
        self.cfg.dataset.train_trial_names = []
        self.cfg.dataset.val_trial_names = []
        self.cfg.dataset.test_trial_names = []

        # gsussian render settings
        self.cfg.dataset.bg_color = [0, 0, 0]
        self.cfg.dataset.zfar = 100.0
        self.cfg.dataset.znear = 0.01
        self.cfg.dataset.trans = [0.0, 0.0, 0.0]
        self.cfg.dataset.scale = 1.0

        # self.cfg.raft = CN()
        # self.cfg.raft.mixed_precision = None
        # self.cfg.raft.train_iters = 0
        # self.cfg.raft.val_iters = 0
        # self.cfg.raft.corr_implementation = 'reg_cuda'  # or 'reg'
        # self.cfg.raft.corr_levels = 4
        # self.cfg.raft.corr_radius = 4
        # self.cfg.raft.n_downsample = 3
        # self.cfg.raft.n_gru_layers = 1
        # self.cfg.raft.slow_fast_gru = None
        # self.cfg.raft.encoder_dims = [64, 96, 128]
        # self.cfg.raft.hidden_dims = [128] * 3
        #
        # self.cfg.gsnet = CN()
        # self.cfg.gsnet.encoder_dims = None
        # self.cfg.gsnet.decoder_dims = None
        # self.cfg.gsnet.parm_head_dim = None
        #
        # self.cfg.record = CN()
        # self.cfg.record.ckpt_path = None
        # self.cfg.record.show_path = None
        # self.cfg.record.logs_path = None
        # self.cfg.record.file_path = None
        # self.cfg.record.loss_freq = 0
        # self.cfg.record.eval_freq = 0

    def get_cfg(self):
        return self.cfg.clone()

    def load(self, config_file):
        self.cfg.defrost()
        self.cfg.merge_from_file(config_file)
        self.cfg.freeze()