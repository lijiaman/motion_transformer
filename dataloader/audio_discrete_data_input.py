import os
import json
import numpy as np
import random

import torch
import torch.utils.data as data


class PoseDiscreteData(data.Dataset):

    def __init__(self, pose3d_folder, discrete_pose3d_folder, mfcc_beat_json_folder, jsonfile, discrete_jsonfile, \
        debug, debug_index, max_timesteps, num_cls, zero_start, audio_vis=False):

        self.ids_dic = json.load(open(jsonfile, 'r'))

        if debug:
            # SMALL DATA TEST: keep a small set of data to test that stuff runs
            self.ids_dic = {str(k): self.ids_dic[str(k)] for k in range(debug_index, debug_index+1)}

        id_list = []
        for k in self.ids_dic.keys():
            id_list.append(int(k))

        self.ids = id_list

        self.debug = debug
        self.debug_index = debug_index

        self.discrete_data = self.ids_dic # They are the same

        self.max_timesteps = max_timesteps

        self.num_cls = num_cls

        self.pose3d_folder = pose3d_folder
        self.discrete_pose3d_folder = discrete_pose3d_folder
        self.mfcc_beat_json_folder = mfcc_beat_json_folder

        self.zero_start = zero_start

        self.audio_vis = audio_vis
      
    def __getitem__(self, index):
        if self.debug:
            index = self.debug_index

        index = str(index)

        data_per_sample = self.ids_dic[index]
        # Load pose3d coordinates
        f_name = data_per_sample['npy']

        # Load discrete data
        discrete_data_per_sample = self.discrete_data[index]
        discrete_pose3d_seq = np.load(os.path.join(self.discrete_pose3d_folder, f_name+".npy")) # T X 17 X 3
        discrete_pose3d_seq = torch.from_numpy(discrete_pose3d_seq).long()
        timesteps = discrete_pose3d_seq.size()[0]
        discrete_pose3d_seq = discrete_pose3d_seq.view(timesteps, -1) # T X (num_p*3)
       
        if self.zero_start:
            start_discrete_zeros = torch.zeros((1, 51)).long()
            start_discrete_zeros.fill_(self.num_cls)
            discrete_pose3d_input_seq = torch.cat((start_discrete_zeros, discrete_pose3d_seq[:-1, :])) # T X 51
            discrete_pose3d_gt_seq = discrete_pose3d_seq
        else:
            discrete_pose3d_input_seq = discrete_pose3d_seq[:-1, :] # T X 51
            discrete_pose3d_gt_seq = discrete_pose3d_seq[1:, :]
            timesteps -= 1

        if timesteps < self.max_timesteps:
            # Pad to max timesteps
            zero_paddings = torch.zeros((self.max_timesteps-timesteps, 51)).long()
            zero_paddings.fill_(self.num_cls)
            discrete_pose3d_input_seq = torch.cat((discrete_pose3d_input_seq, zero_paddings), dim=0) # max_t X 51
            discrete_pose3d_gt_seq = torch.cat((discrete_pose3d_gt_seq, zero_paddings), dim=0) # max_t X 51
        
        # Get timesteps for extracting mask padding
        f_num = timesteps
        f_num_tensor = torch.from_numpy(np.array([f_num])).long()
        mask = torch.arange(self.max_timesteps).expand(1, self.max_timesteps) < f_num_tensor.unsqueeze(1)
        # 1 X max_timesteps

        # Get position vec for position-wise embedding
        pos_vec = torch.arange(timesteps)+1 # timesteps
        pos_vec = pos_vec.unsqueeze(0) # 1 X timesteps
        pos_paddings = torch.zeros((1, self.max_timesteps-timesteps)).long()
        padding_pos_vec = torch.cat((pos_vec, pos_paddings), dim=1) # 1 X max_timesteps

        # Load MFCC and beat information
        mfcc_beat_json = os.path.join(self.mfcc_beat_json_folder, f_name+".json")
        mfcc_beat_data = json.load(open(mfcc_beat_json, 'r'))
        mfcc_data = np.array(mfcc_beat_data['mfcc']) # 39 X 480
        beat_data = mfcc_beat_data['beat'] # a list, each element represent time seconds

        # Pad mfcc features
        mfcc_data = torch.from_numpy(mfcc_data).float()
        mfcc_data = mfcc_data.transpose(0, 1) # 480 X 39
        mfcc_dims = mfcc_data.size()[1]
        if self.zero_start:
            mfcc_data = mfcc_data[:timesteps, :]
        else:
            mfcc_data = mfcc_data[1:timesteps+1, :]

        if timesteps < self.max_timesteps:
            zero_paddings = torch.zeros((self.max_timesteps-timesteps, mfcc_dims))
            mfcc_data = torch.cat((mfcc_data, zero_paddings), dim=0) # max_timesteps X 39

        # Get beat info vector
        block_idx = int(f_name.split('_')[-1])
        block_size = 480
        start = int(block_idx*block_size/2)
        end = int(block_idx*block_size/2)+block_size

        beat_len = len(beat_data)
        fps = 24
        beat_binary_vec = np.zeros(timesteps+1)
        for beat_idx in range(beat_len):
            time_frame = round(fps*beat_data[beat_idx])
            if time_frame >= start and time_frame <= end:
                if time_frame-start <= timesteps:
                    beat_binary_vec[time_frame-start] = 1

        if self.zero_start:
            beat_binary_vec = beat_binary_vec[:timesteps]
        else:
            beat_binary_vec = beat_binary_vec[1:timesteps+1]

        beat_binary_vec = torch.from_numpy(beat_binary_vec).float()
        if timesteps < self.max_timesteps:
            paddings = torch.zeros(self.max_timesteps-timesteps)
            paddings.fill_(2)
            beat_binary_vec = torch.cat((beat_binary_vec, paddings), dim=0) # max_timesteps

        if self.audio_vis:
            return mask, padding_pos_vec, discrete_pose3d_input_seq[:, 3:], discrete_pose3d_gt_seq[:, 3:], \
            mfcc_data[:, :26], beat_binary_vec, f_name
        else:
            return mask, padding_pos_vec, discrete_pose3d_input_seq[:, 3:], discrete_pose3d_gt_seq[:, 3:], \
            mfcc_data[:, :26], beat_binary_vec
        # Since our model does not predict trajectory, we drop root in model training. 
        # max_timesteps X 51(48) (shifted one step)
        # max_timesteps X 51(48)
        # 1 X max_timesteps
        # 1 X max_timesteps
        # max_timesteps X 51(48)
        # max_timesteps X 51(48)
        # mfcc: max_timesteps X 39/26/13
        # beat: max_timesteps

    def __len__(self):
        return len(self.ids)
