import sys
sys.path.append("../")

import time
import numpy as np
import os
import json
from math import *
import pickle

import subprocess

from matplotlib import pyplot as plt
plt.switch_backend('agg')

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F


# Local imports
from utils.utils import AverageMeter

from common.camera import *


def train(args, train_loader, model, optimizer, ce_criterion, device, 
          epoch, curr_lr, model_writer, global_step):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (mask, pos_vec, pose3d_discrete_seq, pose3d_discrete_gt_seq, \
        mfcc_data, beat_data) in enumerate(train_loader):
        # BS X T X 48, BS X T X 48, BS X 1 X T, BS X 1 X T, BS X T X 48
        bs = pose3d_discrete_seq.size()[0]
        timesteps = pose3d_discrete_seq.size()[1]

        # measure data loading time
        data_time.update(time.time() - end)

        # Send to device
        pose3d_discrete_seq = pose3d_discrete_seq.to(device)
        pose3d_discrete_gt_seq = pose3d_discrete_gt_seq.to(device)

        mask = mask.to(device)
        pos_vec = pos_vec.to(device)

        if args.add_mfcc and args.add_beat:
            mfcc_data_input = mfcc_data.to(device)
            beat_data_input = beat_data.to(device).long()
            pred_out = model(pose3d_discrete_seq, mask, pos_vec, \
            mfcc_feats=mfcc_data_input, beat_feats=beat_data_input) # BS X T X 48 X N_cls
        elif args.add_mfcc:
            mfcc_data_input = mfcc_data.to(device)
            pred_out = model(pose3d_discrete_seq, mask, pos_vec, mfcc_feats=mfcc_data_input) # BS X T X 48 X N_cls
        elif args.add_beat:
            beat_data_input = beat_data.to(device).long()
            pred_out = model(pose3d_discrete_seq, mask, pos_vec, beat_feats=beat_data_input) # BS X T X 48 X N_cls
        else:
            pred_out = model(pose3d_discrete_seq, mask, pos_vec) # BS X T X 48 X N_cls
       	
        r_loss = ce_criterion(pred_out, pose3d_discrete_gt_seq, mask.squeeze(1).unsqueeze(2))

        total_loss = r_loss

        model_writer.add_scalar("Loss", np.array(total_loss.item()), global_step)

        total_losses.update(total_loss.item(), 1)
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if (i % args.print_freq == 0):
            print("\n\n")
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Total Loss {total_loss.val:.4f} ({total_loss.avg:.4f})\n'
                  'lr {learning_rate:.6f}\t'
                  .format(epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, 
                          total_loss=total_losses,
                          learning_rate=curr_lr))

        global_step += 1

    return global_step, total_losses.avg


def validate(args, val_loader, model, ce_criterion, device, 
          epoch, model_writer, global_step):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_losses = AverageMeter()

    # switch to train mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (mask, pos_vec, pose3d_discrete_seq, pose3d_discrete_gt_seq, \
            mfcc_data, beat_data) in enumerate(val_loader):
            # BS X T X 48, BS X T X 48, BS X 1 X T, BS X 1 X T, BS X T X 48
            bs = pose3d_discrete_seq.size()[0]
            timesteps = pose3d_discrete_seq.size()[1]

            # measure data loading time
            data_time.update(time.time() - end)

            # Send to device
            pose3d_discrete_seq = pose3d_discrete_seq.to(device)
            pose3d_discrete_gt_seq = pose3d_discrete_gt_seq.to(device)

            mask = mask.to(device)
            pos_vec = pos_vec.to(device)

            if args.add_mfcc and args.add_beat:
                mfcc_data_input = mfcc_data.to(device)
                beat_data_input = beat_data.to(device).long()
                pred_out = model(pose3d_discrete_seq, mask, pos_vec, \
                mfcc_feats=mfcc_data_input, beat_feats=beat_data_input) # BS X T X 48 X N_cls
            elif args.add_mfcc:
                mfcc_data_input = mfcc_data.to(device)
                pred_out = model(pose3d_discrete_seq, mask, pos_vec, mfcc_feats=mfcc_data_input) # BS X T X 48 X N_cls
            elif args.add_beat:
                beat_data_input = beat_data.to(device).long()
                pred_out = model(pose3d_discrete_seq, mask, pos_vec, beat_feats=beat_data_input) # BS X T X 48 X N_cls
            else:
                pred_out = model(pose3d_discrete_seq, mask, pos_vec) # BS X T X 48 X N_cls

            r_loss = ce_criterion(pred_out, pose3d_discrete_gt_seq, mask.squeeze(1).unsqueeze(2))

            total_loss = r_loss

            model_writer.add_scalar("VAL Loss", np.array(total_loss.item()), global_step)

            total_losses.update(total_loss.item(), 1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if (i % args.print_freq == 0):
                print("\n\n")
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Total Loss {total_loss.val:.4f} ({total_loss.avg:.4f})\n'
                      .format(epoch, i, len(val_loader), batch_time=batch_time,
                              data_time=data_time, 
                              total_loss=total_losses,
                              ))

    return total_losses.avg


def convert_discrete_to_coord(discrete_pose):
    # T X 17(16) X 3
    class_pkl = "/data/jiaman/github/cvpr20_dance/json_data/discrete300_class_dict.pkl"
    class_dict = pickle.load(open(class_pkl, 'rb'))
    timesteps, num_joints, coor_dim = discrete_pose.shape
    float_pose3d = np.zeros((timesteps, num_joints, coor_dim))
    for t_idx in range(timesteps):
        for j_idx in range(num_joints):
            int_x, int_y, int_z = discrete_pose[t_idx, j_idx, :]
            int_x = int(int_x)
            int_y = int(int_y)
            int_z = int(int_z)
            float_x = (class_dict[int_x][0]+class_dict[int_x][1])/float(2)
            float_y = (class_dict[int_y][0]+class_dict[int_y][1])/float(2)
            float_z = (class_dict[int_z][0]+class_dict[int_z][1])/float(2)
            float_pose3d[t_idx, j_idx, 0] = float_x
            float_pose3d[t_idx, j_idx, 1] = float_y
            float_pose3d[t_idx, j_idx, 2] = float_z

    return float_pose3d


def add_audio_given_start_inference_vis(args, val_loader, model, device, 
          epoch, model_writer, global_step):
    print('Loading dataset...')
    dataset_path = args.h36m_data

    from common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path)

    cam = dataset.cameras()['S1'][0]
    # cam = dataset.cameras()[args.viz_subject][args.viz_camera]
    rot = dataset.cameras()['S1'][0]['orientation']

    fps = args.fps

    # switch to train mode
    model.eval()

    if args.save_generation:
        gen_times = args.gen_num_for_eval
    else:
        gen_times = 20

    nums_per_mp3 = args.num_per_mp3
    mp3_folder = "/data/jiaman/github/cvpr20_dance/json_data/block_20s_train_val_data_discrete300/block_mp3_files"
    with torch.no_grad():
        for i, (mask, pos_vec, pose3d_discrete_seq, pose3d_discrete_gt_seq, \
            mfcc_data, beat_data, mp3_name) in enumerate(val_loader):
            # BS X T X 48, BS X T X 48, BS X 1 X T, BS X 1 X T, BS X T X 48
            bs = pose3d_discrete_seq.size()[0]
            timesteps = pose3d_discrete_seq.size()[1]

            # Send to device
            pose3d_discrete_seq = pose3d_discrete_seq.to(device)
            pose3d_discrete_gt_seq = pose3d_discrete_gt_seq.to(device)

            # Vis GT
            if not args.zero_start:
                first_step = pose3d_discrete_seq.squeeze(0)[0, :] # 48
                first_step = first_step.unsqueeze(0).view(1, -1, 3).data.cpu().float() # 1 X 16 X 3
                gt_pose3d = pose3d_discrete_gt_seq.squeeze(0).view(timesteps, -1, 3).data.cpu().float() # T X 16 X 3
                mask = mask.squeeze(0).squeeze(0).unsqueeze(1).unsqueeze(2).float()
                act_len = mask.sum() + 1
                gt_pose3d = torch.cat((first_step, gt_pose3d), dim=0) # (T+1) X 16 X 3
            else:
                gt_pose3d = pose3d_discrete_gt_seq.squeeze(0).view(timesteps, -1, 3).data.cpu().float() # T X 16 X 3
                mask = mask.squeeze(0).squeeze(0).unsqueeze(1).unsqueeze(2).float()
                act_len = mask.sum()
            
            act_len = int(act_len)
            gt_pose3d = gt_pose3d[:act_len, :, :] # act_T X 16 X 3

            root_zeros = torch.zeros(act_len, 1, 3).float()
            root_zeros.fill_(144) # Depends on how many classes for classification
            gt_pose3d = torch.cat((root_zeros, gt_pose3d), dim=1) # T X 17 X 3
            gt_pose3d = gt_pose3d.data.cpu()
            gt_pose3d = np.array(gt_pose3d, dtype="float32")

            gt_pose3d = convert_discrete_to_coord(gt_pose3d) # T X 17 X 3
            gt_pose3d = gt_pose3d - np.expand_dims(gt_pose3d[:, 0, :], axis=1) # Make root to zeros
            save_gt_pose3d = np.array(gt_pose3d, dtype="float32")
            
            gt_pose3d = camera_to_world(save_gt_pose3d, R=rot, t=0)
            # We don't have the trajectory, but at least we can rebase the height
            gt_pose3d[:, :, 2] -= np.min(gt_pose3d[:, :, 2], axis=1, keepdims=True)

            # Vis generation
            init_start_pose = pose3d_discrete_seq[:, :args.start_steps, :] # BS X T' X 48

            # Add audio info
            for gen_idx in range(nums_per_mp3):
                if args.add_mfcc and args.add_beat:
                    mfcc_data_input = mfcc_data.to(device) # bs(1) X T X 26
                    beat_data_input = beat_data.to(device).long() # bs(1) X T
                    cal_start_time = time.time()
                    gen_pose3d = model.given_start_inference(init_start_pose, act_len, device, \
                    mfcc_feats=mfcc_data_input, beat_feats=beat_data_input) # 1 X T X 48
                    cal_end_time = time.time()
                    print("Total time for whole seq:{0}".format(cal_end_time-cal_start_time))
                    print("Mean time for each frame among whole seq:{0}".format((cal_end_time-cal_start_time)/act_len))
                elif args.add_mfcc:
                    mfcc_data_input = mfcc_data.to(device)
                    gen_pose3d = model.given_start_inference(init_start_pose, act_len, device, mfcc_feats=mfcc_data_input) # 1 X T X 48
                elif args.add_beat:
                    beat_data_input = beat_data.to(device).long()
                    gen_pose3d = model.given_start_inference(init_start_pose, act_len, device, beat_feats=beat_data_input) # 1 X T X 48
                else:
                    gen_pose3d = model.given_start_inference(init_start_pose, act_len, device) # 1 X T X 48

                gen_pose3d = gen_pose3d.squeeze(0) # T X 48
                gen_pose3d = gen_pose3d.view(act_len, -1, 3) # T X 16 X 3
                
                root_zeros = torch.zeros(act_len, 1, 3).float()
                root_zeros.fill_(144) # Depends on how many classes for classification
                gen_pose3d = torch.cat((root_zeros, gen_pose3d), dim=1) # T X 17 X 3
                gen_pose3d = gen_pose3d.data.cpu()
                gen_pose3d = np.array(gen_pose3d, dtype=np.float32)

                gen_pose3d = convert_discrete_to_coord(gen_pose3d) # T X 17 X 3
                gen_pose3d = gen_pose3d - np.expand_dims(gen_pose3d[:, 0, :], axis=1) # Make root to zeros
                save_gen_pose3d = np.array(gen_pose3d, dtype="float32")

                if args.save_generation:
                    if not os.path.exists(args.gen_res_folder):
                        os.makedirs(args.gen_res_folder)
                    dest_gen_npy_path = os.path.join(args.gen_res_folder, str(i)+"_"+str(gen_idx)+"_gen.npy") # Notice that beat is 1 step delay!!!!!
                    np.save(dest_gen_npy_path, save_gen_pose3d) # T X 17 X 3
                    dest_gt_npy_path = os.path.join(args.gen_res_folder, str(i)+"_gt.npy")
                    np.save(dest_gt_npy_path, save_gt_pose3d) # T X 17 X 3
                    if args.add_beat or args.add_mfcc:
                        dest_beat_path = os.path.join(args.gen_res_folder, str(i)+"_beat_mp3.json")
                        data_dict = {}
                        data_dict['beat'] = np.array(beat_data.data.cpu())[0, :act_len-1].tolist() # T - 1
                        cropped_mp3_path = os.path.join(mp3_folder, str(mp3_name[0])+".mp3")
                        data_dict['mp3'] = cropped_mp3_path
                        json.dump(data_dict, open(dest_beat_path, 'w'))
                else:
                    gen_pose3d = camera_to_world(save_gen_pose3d, R=rot, t=0)
                    # We don't have the trajectory, but at least we can rebase the height
                    gen_pose3d[:, :, 2] -= np.min(gen_pose3d[:, :, 2], axis=1, keepdims=True)

                    anim_output = {'GT Pose3D': gt_pose3d,
                    'Generated Pose3D': gen_pose3d}
                    from common.visualization import render_animation
                    input_keypoints = gen_pose3d[:, :, :2] # Just for keeping vis codes unchanged
                    if not os.path.exists(args.viz_folder):
                        os.makedirs(args.viz_folder)

                    render_animation(input_keypoints, anim_output,
                                     dataset.skeleton(), 
                                     fps,
                                     3000, cam['azimuth'], os.path.join(args.viz_folder, str(i)+"_"+str(gen_idx)+".mp4"),
                                     limit=-1, downsample=1, size=5,
                                     input_video_path=None, viewport=(cam['res_w'], cam['res_h']),
                                     input_video_skip=0)


                    # Merge visual mp4 with mp3 file
                    mp4_path = os.path.join(args.viz_folder, str(i)+"_"+str(gen_idx)+".mp4")

                    dest_folder = os.path.join(args.viz_folder, "merged_mp4")
                    if not os.path.exists(dest_folder):
                        os.makedirs(dest_folder)
                    merged_mp4_file_path = os.path.join(dest_folder, str(i)+"_"+str(gen_idx)+".mp4")

                    cropped_mp3_path = os.path.join(mp3_folder, str(mp3_name[0])+".mp3")
                    merge_audio_cmd = "ffmpeg -i " + mp4_path +" -i "+ cropped_mp3_path + " -c:v copy -c:a aac -strict experimental " + merged_mp4_file_path

                    subprocess.call(merge_audio_cmd, shell=True)

                    os.remove(mp4_path)
            
            if i >= gen_times:
                break;

            

           
