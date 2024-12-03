"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from video_llama.datasets.datasets.base_dataset import BaseDataset
from video_llama.datasets.datasets.caption_datasets import CaptionDataset
import pandas as pd
import decord
from decord import VideoReader
import random
import torch


        
class GEBCVideoDataset(BaseDataset):
    def __init__(self, vis_processor, vis_root):
        #创建对象时将所有视频地址存在video_paths
        """
        vis_root (string): Root directory of videos (e.g. coco/images/)
        """
        #vis_processor：从gebc_videos_builder.py文件的build(self)函数中传入
        super().__init__(vis_processor, text_processor=None, vis_root=vis_root, ann_paths=[])
        #video_paths列表中的每个元素现在都是一个完整的文件路径
        video_paths = os.listdir(vis_root)
        #视频id字典列表
        self.annotation = [{'video_id': video_name[0:11], 'video': video_name} for video_name in video_paths]

    def __getitem__(self, index):
        #获取到视频id
        ann = self.annotation[index]

        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname)#获取到此段视频的地址
                #decord库加载视频，并且形状变换
        video = self.vis_processor(video_path)

        return {
            "video": video,
            "video_id": ann["video_id"],
        }
        
    def collater(self, samples):
        videos = [v['video'] for v in samples]
        videos = torch.stack(videos, 0)
        video_ids = [v['video_id'] for v in samples]
        
        return {
            'video': videos,
            'video_id': video_ids 
        }