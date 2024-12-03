import os
from video_llama.datasets.datasets.base_dataset import BaseDataset
import json
import numpy as np
import pickle
import torch
from scipy.interpolate import interp1d


def read_file(path, MEAN=0., VAR=1., data_norm=False):
    if os.path.exists(path):
        ext = path.split('.')[-1]
        if ext == 'npy':
            feats = np.load(path)
        elif ext == 'pkl':
            with open(path, 'rb') as f:
                feats = pickle.load(f)
        else:
            raise NotImplementedError

        padding = False
    else:
        raise FileNotFoundError('{} not exists'.format(path))
    if data_norm:
        feats = (feats - MEAN) / np.sqrt(VAR)
    return feats, padding


def get_feats(key, vf_type, vf_folder, data_norm=False):
    MEAN = VAR = 0
    if vf_type == 'q_former_tokens':
        feat_dim = 768
        path = os.path.join(vf_folder, key + '.npy')
    elif vf_type == 'intern_video':
        feat_dim = 768
        path = os.path.join(vf_folder, key[0:11] + '.pkl')
    elif vf_type == 'omni':
        feat_dim = 1536
        path = os.path.join(vf_folder, key[0:11] + '.pkl')
    elif vf_type == 'clip':
        feat_dim = 768
        path = os.path.join(vf_folder, key[0:11] + '.pkl')
    else:
        raise AssertionError('feature type error: {}'.format(vf_type))
    feats, padding = read_file(path, MEAN, VAR, data_norm)

    assert feats.shape[-1] == feat_dim, 'load {} error, got shape {}'.format(path, feats.shape)
    return feats


def resizeFeature(inputData, newSize, sample_method):
    # inputX: (temporal_length,feature_dimension) #
    originalSize = len(inputData)
    # print originalSize
    if originalSize == 1:
        inputData = np.reshape(inputData, [-1])
        return np.stack([inputData] * newSize)
    x = np.array(range(originalSize))
    f = interp1d(x, inputData, axis=0, kind=sample_method)
    x_new = [i * float(originalSize - 1) / (newSize - 1) for i in range(newSize)]
    y_new = f(x_new)
    return y_new

class MSVD_MSRVTT_Dataset(BaseDataset):
    def __init__(self, annotation_path, q_former_feature_folder, max_seq_len):
        self.q_former_feature_folder = q_former_feature_folder
        self.max_seq_len = max_seq_len
        super().__init__(vis_processor=None, text_processor=None, vis_root=None, ann_paths=[])
        self._load_annotations(annotation_path)

    def _load_annotations(self, annotation_path):
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        for key, video_caption in data.items():
            msvd_msrvtt_data = {
                'caption_id': key,
                'caption':video_caption,
            }
            self.annotation.append(msvd_msrvtt_data)

    def __getitem__(self, index):
        item_data = self.annotation[index]
        # Load caption
        caption = item_data['caption']
        # Load feature
        video_id = item_data['caption_id'].split('#')[0]
        #print(video_id)
        q_former_tokens = get_feats(video_id, 'q_former_tokens', self.q_former_feature_folder)
        q_former_tokens = torch.from_numpy(q_former_tokens)  # (t,q,h)

        return {
            'caption_id': item_data['caption_id'],
            'image_query_tokens': q_former_tokens,
            'prompt': 'This video describes the',
            'text_input': caption,
        }




class Eval_MSVD_MSRVTT_Dataset(BaseDataset):
    def __init__(self, annotation_path, q_former_feature_folder, max_seq_len):
        self.q_former_feature_folder = q_former_feature_folder
        self.max_seq_len = max_seq_len
        super().__init__(vis_processor=None, text_processor=None, vis_root=None, ann_paths=[])
        self._load_annotations(annotation_path)

    def _load_annotations(self, annotation_path):
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        for key, video_boundaries in data.items():
            val_msvd_msrvtt_data = {
                'video_id': key,
            }
            self.annotation.append(val_msvd_msrvtt_data)

    def __getitem__(self, index):
        item_data = self.annotation[index]
        # Load feature
        q_former_tokens = get_feats(item_data['video_id'], 'q_former_tokens', self.q_former_feature_folder)
        q_former_tokens = torch.from_numpy(q_former_tokens)
        return {
            'image_query_tokens': q_former_tokens,
            'prompt': 'This video describes the',
            'video_id': item_data['video_id'],
        }

