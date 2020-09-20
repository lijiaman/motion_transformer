import os
import numpy as np
import pickle

import torch
import torch.optim

import torch.nn.functional as F

from scipy import sparse

# Local imports
import utils.utils as utils

# For data loader
import prep_dataset.prep_audio_discrete_data as prep_audio_discrete_data

# For model
from modules.two_stream_transformer_discrete import TransformerDiscreteDecoder

# For training and validation
import train_val.two_stream_transformer_discrete_model_play as two_stream_transformer_discrete_model_play

# For settings
import config
args = config.get_args()

# For visualization
from tensorboardX import SummaryWriter


def maskedCrossEntropy(probs, labels, mask):
    # probs: BS X T X J X n_cls, labels: BS X T X J, mask: BS X T X 1
    B, T, J, C = probs.size()
    prob_flat = probs.contiguous().view(-1, C)
    logp_flat = F.log_softmax(prob_flat, dim=1) # (BxT,C) log probabilities
    logp = logp_flat.view(B,T,J,C)
    mask = mask.repeat(1, 1, J) # BS X T X J
    labels = labels.float()*mask.float()
    logp = torch.gather(logp, 3, labels.unsqueeze(3).long()).squeeze(-1) # (B,T,J)
    negative_log_likelihood = -(logp*mask.float())

    # return negative_log_likelihood.mean()
    return torch.sum(negative_log_likelihood)/torch.sum(mask)

def main():
    global args
    print("config:{0}".format(args))

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda" if args.cuda else "cpu")

    global_step = 0
    min_loss = 999999
    min_loss_epoch = 0
    model_writer = SummaryWriter(log_dir=args.log_dir)

    # train_loader, val_loader = prep_discrete_ori_audio_pose_data.get_dataloader(args)
    train_loader, val_loader = prep_audio_discrete_data.get_dataloader(args)
    
    # Define Model
    model = TransformerDiscreteDecoder(n_dec_layers=args.n_dec_layers, n_head=args.n_head,
                           d_feats=args.feats_dim, d_model=args.d_model, d_k=args.d_k, 
                           d_v=args.d_v, d_out=args.d_out, num_cls=args.num_cls, max_timesteps=args.max_timesteps,
                           temperature=args.temperature, add_mfcc=args.add_mfcc, add_beat=args.add_beat, multi_stream=args.multi_stream).to(device)
   
    if args.test_model and args.vis_results:
        utils.load_pretrained_model(model, args.test_model)

        if args.add_mfcc or args.add_beat:
            two_stream_transformer_discrete_model_play.add_audio_given_start_inference_vis(args, val_loader, model,  \
                device, -1, model_writer, global_step)
            
        else:
            two_stream_transformer_discrete_model_play.given_start_inference_vis(args, val_loader, model,  \
                device, -1, model_writer, global_step)
            
        return

    # Define Optimizer
    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr, weight_decay=1e-5)

    ce_criterion = maskedCrossEntropy


    curr_exist_model = os.path.join(args.checkpoint_folder, "curr_checkpoint.pth.tar") 
    if os.path.exists(curr_exist_model):
        checkpoint = torch.load(curr_exist_model)
        args.start_epoch = checkpoint['epoch']+1
        min_loss = checkpoint['min_loss']
        utils.load_pretrained_model(model, curr_exist_model)
        optimizer.load_state_dict(checkpoint['optimizer'])
        curr_loss = two_stream_transformer_discrete_model_play.validate(args, val_loader, model, ce_criterion, \
                device, -1, model_writer, global_step)

    for epoch in range(args.start_epoch, args.epochs):

        curr_lr = utils.adjust_learning_rate(args.lr, args, optimizer, epoch, args.lr_steps)

        global_step, curr_loss = two_stream_transformer_discrete_model_play.train(args, train_loader, model, optimizer, \
            ce_criterion, device, epoch, curr_lr, model_writer, global_step)

        if not args.debug:
            curr_loss = two_stream_transformer_discrete_model_play.validate(args, val_loader, model, ce_criterion, \
                device, epoch, model_writer, global_step)

        if curr_loss < min_loss:
            min_loss = curr_loss
            min_loss_epoch = epoch
            utils.save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'min_loss': min_loss,
            }, epoch, min_loss, folder=args.checkpoint_folder)
            print("Min Loss is {0}, in {1} epoch.".format(min_loss, epoch))

        # Always save current one.
        utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'min_loss': min_loss,
        }, min_loss_epoch, min_loss, folder=args.checkpoint_folder, filename='curr_checkpoint.pth.tar')

    model_writer.close()


if __name__ == '__main__':
    main()
