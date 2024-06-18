# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


try:
    pass
except ModuleNotFoundError:
    pass

from torchtitan.config_manager import JobConfig as BaseJobConfig


class JobConfig(BaseJobConfig):
    def __init__(self):
        super().__init__()

        # model configs
        self.parser.add_argument(
            "--model.tokenizer_type",
            type=str,
            default="sentencepiece",
            choices=["sentencepiece", "tiktoken"],
            help="Tokenizer type",
        )
