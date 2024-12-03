from video_llama.common.registry import registry
from video_llama.tasks.base_task import BaseTask
from video_llama.datasets.data_utils import prepare_sample
from tqdm import tqdm
import torch
import numpy as np
import os

@registry.register_task("video_feature_extraction")
class VideoFeatureExtractionTask(BaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        pass

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """


        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) == 1, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            #gebc_videos_builder.py,  但是build_datasets()在父类base_dataset_builder.py
            dataset = builder.build_datasets()#得到数据集对象
            break

        return dataset#处理视频对象
    #特征提取，被runner调用
    def feature_extraction(self, model, data_loader, save_dir, cuda_enabled=True):
        with torch.no_grad():
            #批量提取视频
            for sample in tqdm(data_loader):
                #move_to_cuda
                sample = prepare_sample(samples=sample, cuda_enabled=cuda_enabled)
                                #forward函数
                features = model.extract_feature(sample)
                self.save_feature(sample, features, save_dir=save_dir)
        
    def save_feature(self, samples, features, save_dir):
        #批量id
        video_names = samples['video_id']
        batch_output = [x.detach().cpu().numpy() for x in features]
        for video_name, feature in zip(video_names, batch_output):
            save_path = os.path.join(save_dir, video_name + '.npy')
            # print(save_path, feature.shape)
            np.save(save_path, feature)
