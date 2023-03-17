"""Training and evaluation"""

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

import lib.diffusion.trainer as trainer
import lib.diffusion.evaler as evaler


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "diffusion configs", lock_config=False)
flags.DEFINE_enum("mode", None, ["train", "uncond_gen", "cond_gen"], "Running mode")
flags.mark_flags_as_required(["config", "mode"])


def main(argv):
    if FLAGS.mode == 'train':
        trainer.train(FLAGS.config)
    elif FLAGS.mode == 'uncond_gen':
        evaler.uncond_gen(FLAGS.config)
    elif FLAGS.mode == 'cond_gen':
        evaler.cond_gen(FLAGS.config)

if __name__ == "__main__":
  app.run(main)
