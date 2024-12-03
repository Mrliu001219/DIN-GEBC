"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from video_llama.common.registry import registry
from video_llama.tasks.base_task import BaseTask
from video_llama.tasks.image_text_pretrain import ImageTextPretrainTask
from video_llama.tasks.video_text_pretrain import VideoTextPretrainTask
from video_llama.tasks.video_feature_extraction import VideoFeatureExtractionTask
from video_llama.tasks.boundary_captioning import BoundaryCaptionTask
from video_llama.tasks.boundary_captioning_Qformer_and_RWKV import BoundaryCaption_Qformer_and_RWKV_Task
from video_llama.tasks.task_msvd_msrvtt import Task_MSVD_MSRVTT
from video_llama.tasks.comparison.task_single_branch import Task_Single_Branch
from video_llama.tasks.comparison.task_dual_branch import Task_Dual_Branch
from video_llama.tasks.comparison.task_single_branch_qformer_rwkv import Task_Single_Branch_Qformer_RWKV

def setup_task(cfg):
    assert "task" in cfg.run_cfg, "Task name must be provided."
    #task在run_cfg中，任务名称
    task_name = cfg.run_cfg.task#从参数里获取名称
    task = registry.get_task_class(task_name).setup_task(cfg=cfg)#得到一个传完参数的类
    assert task is not None, "Task {} not properly registered.".format(task_name)

    return task

#包括后面所有py文件
__all__ = [
    "BaseTask",
    "ImageTextPretrainTask",
    "VideoTextPretrainTask",
    "VideoFeatureExtractionTask",
    "BoundaryCaptionTask",
    "BoundaryCaption_Qformer_and_RWKV_Task",
    "Task_MSVD_MSRVTT",
    "Task_Single_Branch",
    "Task_Dual_Branch",
    "Task_Single_Branch_Qformer_RWKV"
]
