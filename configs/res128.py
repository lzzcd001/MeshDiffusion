"""Config file for reproducing the results of DDPM on bedrooms."""

from configs.default_configs import get_default_configs


def get_config():
  config = get_default_configs()

  # training
  training = config.training
  training.sde = 'vpsde'
  training.continuous = False
  training.reduce_mean = True
  training.batch_size = 8
  training.lip_scale = None
  training.iter_size = 4

  training.snapshot_freq_for_preemption = 1000

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'ancestral_sampling'
  sampling.corrector = 'none'

  # data
  data = config.data
  data.dataset = 'ShapeNet'
  data.centered = True
  data.image_size = 128
  data.num_channels = 4
  data.meta_path = "PLACEHOLDER" ### metadata for all dataset files
  data.filter_meta_path = "PLACEHOLDER" ### metadata for the list of training samples
  data.num_workers = 8
  data.aug = True


  # model
  model = config.model
  model.name = 'ddpm_res128_v2'
  model.scale_by_sigma = False
  model.num_scales = 1000
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 1, 2, 4, 4, 4)
  model.num_res_blocks_first = 2
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.dropout = 0.1

  # optim
  optim = config.optim
  optim.lr = 7e-5 / training.iter_size * 2.0

  config.eval.batch_size = 7
  config.seed = 42

  return config
