
from video_llama.common.registry import registry
from video_llama.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from video_llama.datasets.datasets.gebc_video_dataset import GEBCVideoDataset

@registry.register_builder("gebc_videos")
class GEBCVideosBuilder(BaseDatasetBuilder):
    #定义基类中的train_dataset_cls，在gebc_video_dataset.py中
    train_dataset_cls = GEBCVideoDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/gebc_videos/defaults.yaml"}
    
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass
    
    def build_processors(self):
        vis_proc_cfg = self.config.get("vis_processor")

        if vis_proc_cfg is not None:
            vis_eval_cfg = vis_proc_cfg.get("eval")
            #获取视频处理类，创建对象         父类 base_dataset_builder.py
            self.vis_processors["eval"] = self._build_proc_from_cfg(vis_eval_cfg)

    #创建对象
    def build(self):
        #创建视频处理对象self.vis_processors["eval"]
        self.build_processors()

        build_info = self.config.build_info
        dataset_cls = self.train_dataset_cls
        #初始化GEBCVideoDataset类，创建对象    gebc_video_deteset.py
        dataset = dataset_cls(
            #self.vis_processors['eval']，在self.build_processors()中创建
            vis_processor=self.vis_processors['eval'],#视频处理对象
            vis_root=build_info.videos_dir,#根路径
        )
        #返回数据集对象
        return dataset