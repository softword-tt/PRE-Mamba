"""
EventRain-27K dataset (EVK4: Artifical & Real-world >> Scene 1-5)

Author: Ciyu Ruan (rcy23@mails.tsinghua.edu.cn)、Ruishan Guo (grs24@mails.tsinghua.edu.cn)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
import re
import torch

import random

from .builder import DATASETS
from .defaults import DefaultMultiScansDataset

def absoluteFilePaths(directory, selected_file_list=None):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            if selected_file_list is None or f.split('.')[0] in selected_file_list:
                yield os.path.abspath(os.path.join(dirpath, f))

@DATASETS.register_module()
class EVK4MultiScansDataset(DefaultMultiScansDataset):
    def __init__(
        self,
        split="train",
        data_root="data",
        event_size = [1280, 720],
        data_width = 1280,
        data_height = 720,
        gather_num=6,
        scan_modulation=False,
        transform=None,
        test_mode=False,
        test_cfg=None,
        windows_stride=None,
        loop=1,
        ignore_index=-1,
    ):
        self.gather_num = gather_num
        self.event_size = event_size
        self.data_width = data_width
        self.data_height = data_height
        self.scan_modulation = scan_modulation
        self.ignore_index = ignore_index
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
        super().__init__(
            split=split,
            data_root=data_root,
            gather_num=gather_num,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

        self.windows_stride = windows_stride

    def get_data_list(self):

        scene_split = {
        'scene1': {
            'train': [1],
            'val': [4],
            'test': [2, 3]
        },
        'scene2': { 
            'train': [3, 9], 
            'val': [4, 6],
            'test': [1, 2, 5, 10]
        },
        'scene3': {
            'train': [4, 10],
            'val': [6, 8],
            'test': [2,9]
        },
        'scene4': {
            'train': [1, 3],
            'val': [4, 9],
            'test': [2, 6, 13]
        },
        # 'scene5': {
        #     # 'train': [1],
        #     # 'val': [3],
        #     # 'test': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
        # },
    }

        data_list = []
        self.poses = {}

        # Get parent directory of data_root to access all scene folders
        parent_dir = os.path.dirname(self.data_root)

        # Get all scene folders
        scene_folders = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d)) and d.startswith('scene')]

        for scene in scene_folders:
            # Extract scene number and check if it's in our scene_split dictionary
            if scene not in scene_split:
                print(f"Skipping {scene} - no split configuration defined")
                continue
                
            # Get scene-specific sequences for current split
            if isinstance(self.split, str):
                seq_list = scene_split[scene][self.split]
            elif isinstance(self.split, list):
                seq_list = []
                for split in self.split:
                    seq_list += scene_split[scene][split]
            else:
                raise NotImplementedError
                
            scene_path = os.path.join(parent_dir, scene)
            
            # Process sequences for this scene
            for seq in seq_list:
                seq = f"rain_{seq}"
                seq_folder = os.path.join(scene_path, "merge_data")
                seq_path = os.path.join(seq_folder, seq)
                
                if os.path.exists(seq_path):
                    if self.split == "train":
                        data_list += absoluteFilePaths(seq_path)
                        print(f"Processing {scene} frames in Seq {seq} without filtering.")
                    else:
                        data_list += absoluteFilePaths(seq_path)

        return data_list

    def get_multi_data(self, idx):
        cur_data_path = self.data_list[idx % len(self.data_list)]

        gather_coord = []
        gather_strength = []
        gather_segment = []
        
        # Get scene and sequence info from path
        # Path format: /media/sdb2/grs/data/EVK4_derain/scene{x}/merge_data/rain_{y}/{z}.npz
        scene = cur_data_path.split('/')[-4]  # Get scene folder name
        seq = cur_data_path.split('/')[-2]    # Get rain_x folder name
        file_name = cur_data_path.split('/')[-1]  # Get file name
        scene_path = os.path.dirname(os.path.dirname(os.path.dirname(cur_data_path)))  # Get scene directory
        
        cur_scan_index = int(file_name.split('.')[0])

        tn = []
        modulation = 1
        if self.scan_modulation:
            modulation = random.randint(1, 3)

        for i in range(self.gather_num):
            last_scan_index = cur_scan_index - modulation * i
            last_scan_index = max(0, last_scan_index)
            
            # Construct scan path within current scene
            scan_path = os.path.join(
                scene_path,
                "merge_data",
                seq,
                f"{str(last_scan_index).zfill(10)}.npz"
            )

            if not os.path.exists(scan_path):
                print(f"File {scan_path} does not exist, skipping...")
                continue

            # Load scan data
            with np.load(scan_path) as data:
                x = data['x']
                y = data['y']
                t = data['t']
                p = data['p']
                if len(x) == 0:
                    print(f"No points in file: {scan_path}")
                    continue

            strength = p.reshape(-1, 1)
            
            label_path = os.path.join(
                scene_path,
                "labels",
                f"labels_{seq}",
                f"labels_{str(last_scan_index).zfill(10)}.npy"
            )

            if not os.path.exists(label_path):
                print(f"Label file {label_path} does not exist, skipping...")
                continue

            labels = np.load(label_path)
            t_normalized = (t - t.min()) / (t.max() - t.min())
            coord = np.column_stack((x, y, t_normalized))

            gather_coord.append(coord)
            gather_strength.append(strength)
            gather_segment.append(labels)
            tn.append(np.ones(len(coord)) * i)

        data_dict = dict(
            coord=np.concatenate(gather_coord),
            strength=np.concatenate(gather_strength),
            segment=np.concatenate(gather_segment),
            tn=np.expand_dims(np.concatenate(tn), axis=1)
        )

        torch.cuda.empty_cache()  # Add memory cleanup before return

        return data_dict

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]

        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = scan[:, -1].reshape([-1, 1])

        '''     
        label_file = data_path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.exists(label_file):
            with open(label_file, "rb") as a:
                segment = np.fromfile(a, dtype=np.int32).reshape(-1)
                segment = np.vectorize(self.learning_map.__getitem__)(
                    segment & 0xFFFF
                ).astype(np.int32)
        else:
            segment = np.zeros(scan.shape[0]).astype(np.int32)
        '''

        segment_path = data_path.replace("velodyne", "segments").replace(".bin", ".segment")
        if os.path.exists(segment_path):
            with open(segment_path, "rb") as a:
                segment = np.fromfile(a, dtype=np.float32).reshape(-1, 4)
        else:
            segment = np.zeros(scan.shape).astype(np.float32)

        data_dict = dict(coord=coord, strength=strength, segment=segment)
        return data_dict

    def get_data_name(self, idx):

        file_path = self.data_list[idx % len(self.data_list)] # /media/sdb2/grs/data/EVK4_derain/scene2/merge_data/rain_5/0000000104.npz
        # Extract components from path
        path_parts = file_path.split('/')
        scene = path_parts[-4]                  # scene2
        seq = path_parts[-2]                    # rain_5
        frame = path_parts[-1].split('.')[0]    # 0000000104
        data_name = f"{scene}/{seq}/{frame}"

        return data_name

    @staticmethod
    def get_learning_map_inv(ignore_index):
        learning_map_inv = {
            ignore_index: ignore_index,  # "unlabeled"
            0: 0,
            1: 1,
            2: 250,
            3: 251,
        }

        return learning_map_inv