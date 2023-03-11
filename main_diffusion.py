# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
