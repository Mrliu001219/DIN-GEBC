"""
Adapted from salesforce@LAVIS and Vision-CAIR@MiniGPT-4. Below is the original copyright:
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
os.environ['TORCH_HOME'] = '.cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '.cache'
os.environ['TRANSFORMERS_CACHE'] = '.cache'
import numpy as np
import video_llama.tasks as tasks
from video_llama.common.config import Config
from video_llama.common.registry import registry
from video_llama.common.utils import now

# imports modules for registration
from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.runners import *
from video_llama.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Feature Extraction")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    #获得运行类
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))
    #get函数，第一个参数是要寻找的k，如果没有找到，就返回第二个参数

    return runner_cls


def main():
    job_id = now()
    cfg = Config(parse_args())#合并之后的配置对象，runner_config, model_config, dataset_config, user_config
    print(cfg)#输出对象某些属性，Config类中有__str__()方法
    task = tasks.setup_task(cfg)#得到任务对象
    datasets = task.build_datasets(cfg)#获得处理视频对象
    model = task.build_model(cfg)#获得模型对象
    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets#传入处理视频对象
    )#获取运行类并且创建对象
    runner.start_extract()

if __name__ == "__main__":
    main()
