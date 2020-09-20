import sys
sys.path.append("../")

import pickle

import torch
import torch.utils.data

from dataloader.audio_discrete_data_input import PoseDiscreteData


def get_dataloader(args):
    # Define data files path.
    train_json_file = args.train_json_file
    val_json_file = args.val_json_file

    train_discrete_json_file = args.train_discrete_json_file
    val_discrete_json_file = args.val_discrete_json_file

    if not args.debug:
        train_dataset = PoseDiscreteData(args.pose3d_folder, args.discrete_pose3d_folder, args.mfcc_beat_json_folder, 
            train_json_file, train_discrete_json_file, 
            args.debug, args.debug_index, args.max_timesteps, args.num_cls, args.zero_start)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)


        if args.vis_results:
            val_dataset = PoseDiscreteData(args.pose3d_folder, args.discrete_pose3d_folder, args.mfcc_beat_json_folder,
            val_json_file, val_discrete_json_file, 
            args.debug, args.debug_index, args.max_timesteps, args.num_cls, args.zero_start, audio_vis=True)
        else:
            val_dataset = PoseDiscreteData(args.pose3d_folder, args.discrete_pose3d_folder, args.mfcc_beat_json_folder,
                val_json_file, val_discrete_json_file, 
                args.debug, args.debug_index, args.max_timesteps, args.num_cls, args.zero_start)

        if args.vis_results:
            val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
        else:
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
    else:
        train_dataset = PoseDiscreteData(args.pose3d_folder, args.discrete_pose3d_folder, args.mfcc_beat_json_folder,
            train_json_file, train_discrete_json_file, 
            args.debug, args.debug_index, args.max_timesteps, args.num_cls, args.zero_start)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        val_dataset = PoseDiscreteData(args.pose3d_folder, args.discrete_pose3d_folder, args.mfcc_beat_json_folder,
            train_json_file, train_discrete_json_file, 
            args.debug, args.debug_index, args.max_timesteps, args.num_cls, args.zero_start)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader
