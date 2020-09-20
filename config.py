import argparse

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch DanceTransformer Training')
    
    parser.add_argument('--checkpoint-folder', metavar='DIR',
                        help='path to checkpoint dir',
                        default='./checkpoint')
    parser.add_argument('--log-dir', metavar='DIR',
                        help='path to tensorboard dir',
                        default='./loss_vis')
    parser.add_argument('--viz-folder', metavar='DIR',
                        help='path to visualization videos dir',
                        default='./video_vis')
    parser.add_argument('--audio-folder', metavar='DIR',
                        help='path to audio dir',
                        default='')
    parser.add_argument('--pose3d-folder', metavar='DIR',
                        help='path to 3d coordinates dir',
                        default='')
    parser.add_argument('--discrete-pose3d-folder', metavar='DIR',
                        help='path to 3d discrete coordinates dir',
                        default='')
    parser.add_argument('--mfcc-beat-json-folder', metavar='DIR',
                        help='path to mfcc beat info dir',
                        default='')

    parser.add_argument('--seed', type=int, default=6666,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')

    parser.add_argument('--debug', action='store_true',
                        help='using a small set of data to overfit')
    parser.add_argument('--debug-index', default=0, type=int, metavar='N',
                        help='select specific data idx for debug, fitting one sample')

    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')
    
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('--lr-steps', default=[40], nargs="+",
                        metavar='LRSteps', help='epochs to decay learning rate by 10')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 1)')
    parser.add_argument('--lr', default=3e-5, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-6, type=float,
                        metavar='W', help='weight decay (default: 5e-6)')
    
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    
    parser.add_argument('--resume-model', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained-model', default='', type=str, metavar='PATH',
                        help='path to pretrained checkpoint (default: none)')
    parser.add_argument('--test-model', default='', type=str, metavar='PATH',
                        help='path to checkpoint used for visualization(default: none)')

    parser.add_argument('--train-json-file', default='', type=str, metavar='PATH',
                        help='train json file for training data (default: none)')
    parser.add_argument('--val-json-file', default='', type=str, metavar='PATH',
                        help='validation json file for validation data (default: none)')
    
    parser.add_argument('--train-discrete-json-file', default='', type=str, metavar='PATH',
                        help='train json file for training data (default: none)')
    parser.add_argument('--val-discrete-json-file', default='', type=str, metavar='PATH',
                        help='validation json file for validation data (default: none)')
    

    parser.add_argument('--h36m-data', default='/data/jiaman/github/dance_project/data_processing/VideoPose3D/data/data_3d_h36m.npz', type=str, metavar='PATH',
                        help='validation json file for validation data (default: none)')

    parser.add_argument('--class-pkl-file', default='/data/jiaman/github/cvpr20_dance/json_data/ik_fk_discrete300_class_dict.pkl', type=str, metavar='PATH',
                        help='validation json file for validation data (default: none)')

    parser.add_argument('--n-dec-layers', type=int, default=8,
                        help='the number of decoder layers')
    parser.add_argument('--n-head', type=int, default=8,
                        help='the number of multi-head attention')
    parser.add_argument('--d-model', type=int, default=256,
                        help='size of hidden state for decoder model')
    parser.add_argument('--d-k', type=int, default=256,
                        help='size of hidden state for key')
    parser.add_argument('--d-v', type=int, default=256,
                        help='size of hidden state for query and value')
    parser.add_argument('--d-hidden', type=int, default=256,
                        help='size of hidden state for query and value')
    parser.add_argument('--num-mixtures', type=int, default=20,
                        help='the number of gaussian mixtures for GMM model')

    parser.add_argument('--d-out', type=int, default=48,
                        help='the output dim before final classification')

    # For LSTM setting
    parser.add_argument('--feats-dim', type=int, default=48,
                        help='the number of gaussian mixtures for GMM model')
    parser.add_argument('--decoder-dim', type=int, default=1024,
                        help='the number of gaussian mixtures for GMM model')

    parser.add_argument('--max-timesteps', type=int, default=480,
                        help='the maximum timesteps for transformer model')

    parser.add_argument('--num-cls', type=int, default=50,
                        help='the number classes for discrete classes')

    parser.add_argument('--n-wavenet-layers', type=int, default=50,
                        help='the number layers for each block in wavenet')
    parser.add_argument('--n-wavenet-blocks', type=int, default=50,
                        help='the number of blocks in wavenet')

    parser.add_argument('--start-steps', type=int, default=1,
                        help='the number classes for discrete classes')

    parser.add_argument('--temperature', default=1, type=float, metavar='M',
                        help='used in sampling')

    parser.add_argument('--data-scale', type=int, default=1,
                        help='control value range for data input')
    
    parser.add_argument('--vis-results', action='store_true',
                        help='Visualize input and output 3d pose')

    parser.add_argument('--vis-debug', action='store_true',
                        help='debug for inference')

    parser.add_argument('--check-discrete-vis', action='store_true',
                        help='debug for inference')

    parser.add_argument('--given-start-inference', action='store_true',
                        help='Visualize input and output 3d pose')

    parser.add_argument('--zero-start', action='store_true',
                        help='input zeros for first timestep')

    parser.add_argument('--auto-condition', action='store_true',
                        help='use acLSTM scheduled sampling strategy')
    parser.add_argument('--auto-condition-step', type=int, default=1,
                        help='set the length for gt or pred training')

    parser.add_argument('--fps', type=int, default=24,
                        help='fps for generating visualization mp4')

    parser.add_argument('--vis-timesteps', type=int, default=480,
                        help='timesteps for visualization')

    parser.add_argument('--add-mfcc', action='store_true',
                        help='add mfcc feats for each timestep')
    parser.add_argument('--add-beat', action='store_true',
                        help='add beat feats for each timestep')


    parser.add_argument('--uncondition-generation', action='store_true',
                        help='uncondition generation for VAE')
    parser.add_argument('--condition-generation', action='store_true',
                        help='condition generation for VAE')
    parser.add_argument('--interpolation-generation', action='store_true',
                        help='interpolation-generation for VAE')

    parser.add_argument('--save-generation', action='store_true',
                        help='save generated pose sequence into numpy for further evaluation')
    parser.add_argument('--gen-res-folder', metavar='DIR',
                        help='path to generated pose seq results',
                        default='')
    parser.add_argument('--gen-num-for-eval', type=int, default=200,
                        help='the number of generations used for evaluation')
    parser.add_argument('--num-per-mp3', type=int, default=1,
                        help='the number of generations used for evaluation')

    parser.add_argument('--beta', default=1, type=float,
                        metavar='BT', help='weight for controlling VAE loss')

    parser.add_argument('--multi-stream', action='store_true',
                        help='split beat and mfcc into two different streams')

    parser.add_argument('--zero-start-inference', action='store_true',
                        help='split beat and mfcc into two different streams')

    parser.add_argument('--long-seq-gen-num', type=int, default=24,
                        help='fps for generating visualization mp4')
    parser.add_argument('--long-seq-generation', action='store_true',
                        help='split beat and mfcc into two different streams')

    args = parser.parse_args()

    return args
