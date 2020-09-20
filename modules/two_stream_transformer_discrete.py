import sys
sys.path.append("../")

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.distributions.multivariate_normal as dist_mn

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1) # b x ls x ls
    
    return subsequent_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, n_head*d_k)
        self.w_k = nn.Linear(d_model, n_head*d_k)
        self.w_v = nn.Linear(d_model, n_head*d_v)
        nn.init.normal_(self.w_q.weight, mean=0, std=np.sqrt(2.0/(d_model+d_k)))
        nn.init.normal_(self.w_k.weight, mean=0, std=np.sqrt(2.0/(d_model+d_k)))
        nn.init.normal_(self.w_v.weight, mean=0, std=np.sqrt(2.0/(d_model+d_v)))

        self.temperature = np.power(d_k, 0.5)
        self.attn_dropout = nn.Dropout(0.1)

        self.fc = nn.Linear(n_head*d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, q, k, v, mask=None):
        # q: BS X T X D, k: BS X T X D, v: BS X T X D, mask: BS X T X T 
        bs, n_q, _ = q.shape
        bs, n_k, _ = k.shape
        bs, n_v, _ = v.shape

        assert n_k == n_v

        residual = q

        q = self.w_q(q).view(bs, n_q, self.n_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_q, self.d_k)
        k = self.w_k(k).view(bs, n_k, self.n_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, self.d_k)
        v = self.w_v(v).view(bs, n_v, self.n_head, self.d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_v, self.d_v)

        attn = torch.bmm(q, k.transpose(1, 2)) # (n_head*bs) X n_q X n_k
        attn = attn / self.temperature

        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1) # (n_head*bs) x n_q x n_k 
            attn = attn.masked_fill(mask, -np.inf)

        attn = F.softmax(attn, dim=2) # (n_head*bs) X n_q X n_k
        attn = self.attn_dropout(attn)
        output = torch.bmm(attn, v) # (n_head*bs) X n_q X d_v

        output = output.view(self.n_head, bs, n_q, self.d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(bs, n_q, -1)
        # BS X n_q X (n_head*D)

        output = self.dropout(self.fc(output)) # BS X n_q X D
        output = self.layer_norm(output + residual) # BS X n_q X D

        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: BS X N X D
        residual = x
        output = x.transpose(1, 2) # BS X D X N
        output = self.w_2(F.relu(self.w_1(output))) # BS X D X N
        output = output.transpose(1, 2) # BS X N X D
        output = self.dropout(output)
        output = self.layer_norm(output + residual) # BS X N X D

        return output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_model)

    def forward(self, decoder_input, self_attn_time_mask, self_attn_padding_mask):
        # decode_input: BS X T X D
        # time_mask: BS X T X T (padding postion are ones)
        # padding_mask: BS X T (padding position are zeros, diff usage from above)
        bs, dec_len, dec_hidden = decoder_input.shape
        
        decoder_out, dec_self_attn = self.self_attn(decoder_input, decoder_input, decoder_input, \
                                mask=self_attn_time_mask)
        # BS X T X D, BS X T X T
        decoder_out *= self_attn_padding_mask.unsqueeze(-1).float()
        # BS X T X D

        decoder_out = self.pos_ffn(decoder_out) # BS X T X D
        decoder_out *= self_attn_padding_mask.unsqueeze(-1).float()

        return decoder_out, dec_self_attn
        # BS X T X D, BS X T X T


class Decoder(nn.Module):
    def __init__(
            self,
            d_feats, d_model,
            n_layers, n_head, d_k, d_v, max_timesteps):
        super(Decoder, self).__init__()

        self.start_conv = nn.Conv1d(d_feats, d_model, 1) # (input: 17*3)
        self.position_vec = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(max_timesteps+1, d_model, padding_idx=0),
            freeze=True)
        self.layer_stack = nn.ModuleList([DecoderLayer(d_model, n_head, d_k, d_v)
            for _ in range(n_layers)])

    def forward(self, decoder_input, padding_mask, decoder_pos_vec):
        # decoder_input: BS X D X T 
        # padding_mask: BS X 1 X T
        # decoder_pos_vec: BS X 1 X T

        dec_self_attn_list = []

        padding_mask = padding_mask.squeeze(1) # BS X T
        decoder_pos_vec = decoder_pos_vec.squeeze(1) # BS X T

        input_embedding = self.start_conv(decoder_input)  # BS X D X T
        input_embedding = input_embedding.transpose(1, 2) # BS X T X D
        pos_embedding = self.position_vec(decoder_pos_vec) # BS X T X D
        
        # Time mask is same for all blocks, while padding mask differ according to the position of block
        time_mask = get_subsequent_mask(decoder_pos_vec) 
        # BS X T X T (Prev steps are 0, later 1)
        
        dec_output = input_embedding + pos_embedding # BS X T X D
        for dec_layer in self.layer_stack:
            dec_output, dec_self_attn = dec_layer(
                dec_output, # BS X T X D
                self_attn_time_mask=time_mask, # BS X T X T
                self_attn_padding_mask=padding_mask) # BS X T

            dec_self_attn_list += [dec_self_attn]

        return dec_output, dec_self_attn_list
        # BS X T X D, list


class TransformerDiscreteDecoder(nn.Module):
    def __init__(
            self, n_dec_layers, n_head,
            d_feats, d_model, d_k, d_v, d_out, num_cls, max_timesteps, 
            temperature=1, add_mfcc=False, add_beat=False, multi_stream=False):
        super(TransformerDiscreteDecoder, self).__init__()

        self.d_feats = d_feats 
        self.d_model = d_model

        self.num_cls = num_cls

        self.max_timesteps = max_timesteps

        self.temperature = temperature

        self.add_mfcc = add_mfcc
        self.add_beat = add_beat

        self.multi_stream = multi_stream

        self.input_embed = nn.Embedding(num_cls+1, d_feats, padding_idx=num_cls) # +1 for padding, num_cls index is for padding embed

        # Input: BS X D X T (BS X (48*D) X T)
        # Output: BS X T X D' (BS X T X D')

        decoder_feats_dim = d_feats*48
        self.decoder = Decoder(d_feats=decoder_feats_dim, d_model=d_model, \
            n_layers=n_dec_layers, n_head=n_head, d_k=d_k, d_v=d_v, \
            max_timesteps=max_timesteps)

        # Add audio stream transformer
        if self.add_beat:
            self.beat_embedding = nn.Embedding(3, 30, padding_idx=2) # beat 1, no beat 0, padding 2

        if self.add_mfcc: # The data scale can be -400~400, need to do a embedding first
            self.mfcc_embedding = nn.Linear(26, 30)
            self.mfcc_tanh = nn.Tanh()

        audio_decoder_feats_dim = 0
        if self.add_mfcc:
            audio_decoder_feats_dim += 30
        if self.add_beat:
            audio_decoder_feats_dim += 30

        if self.multi_stream:
            self.beat_decoder = Decoder(d_feats=30, d_model=d_model//4, \
                n_layers=n_dec_layers//2, n_head=n_head//2, d_k=d_k//4, d_v=d_v//4, \
                max_timesteps=max_timesteps)
            self.mfcc_decoder = Decoder(d_feats=30, d_model=d_model//4, \
                n_layers=n_dec_layers//2, n_head=n_head//2, d_k=d_k//4, d_v=d_v//4, \
                max_timesteps=max_timesteps)

            self.final_mapping = nn.Linear(in_features=d_model+d_model//2, out_features=48*d_out)
        else:
            self.audio_decoder = Decoder(d_feats=audio_decoder_feats_dim, d_model=d_model//4, \
                n_layers=n_dec_layers//2, n_head=n_head//2, d_k=d_k//4, d_v=d_v//4, \
                max_timesteps=max_timesteps)

            self.final_mapping = nn.Linear(in_features=d_model+d_model//4, out_features=48*d_out)
        
        # self.final_relu = nn.ReLU() 
        self.final_cls = nn.Linear(in_features=d_out, out_features=num_cls) 
        # BS X T X D -> BS x T X (48*n_cls)

        # Beat constraint module
        # self.beat_conv = nn.Conv1d(num_channels*48, num_hidden, 1)

    def forward(self, decoder_input, padding_mask, pos_vec, mfcc_feats=None, beat_feats=None):
        # decoder_input: BS X T X D(51/48)
        # padding_mask: BS X 1 X T
        # pos_vec: BS X 1 X T
        # mfcc_feats: BS X T X 26
        # beat_feats: BS X T
        bs, max_len, n_joints = decoder_input.shape
        data_input = self.input_embed(decoder_input) # BS X T X 48 X D
        data_input = data_input.view(bs, max_len, -1) # BS X T X (48*D)

        data_input = data_input.transpose(1, 2)  # BS X (48*D) X T
        decoder_out, _ = self.decoder(data_input, padding_mask, pos_vec) 
        # BS X T X D'

        # Audio Transformer Part
        if self.multi_stream:
            mfcc_input = self.mfcc_embedding(mfcc_feats) # BS X T X 30
            mfcc_input = self.mfcc_tanh(mfcc_input) # BS X T X 30
            beat_input = self.beat_embedding(beat_feats) # BS X T X 30

            mfcc_data_input = mfcc_input.transpose(1, 2)
            mfcc_decoder_out, _ = self.mfcc_decoder(mfcc_data_input, padding_mask, pos_vec)

            beat_data_input = beat_input.transpose(1, 2)
            beat_decoder_out, _ = self.beat_decoder(beat_data_input, padding_mask, pos_vec)

            audio_decoder_out = torch.cat((beat_decoder_out, mfcc_decoder_out), dim=2)
        else:
            if self.add_mfcc and self.add_beat:
                mfcc_input = self.mfcc_embedding(mfcc_feats) # BS X T X 30
                mfcc_input = self.mfcc_tanh(mfcc_input) # BS X T X 30
                beat_input = self.beat_embedding(beat_feats) # BS X T X 30
                audio_data_input = torch.cat((mfcc_input, beat_input), dim=2) # BS X T X 60
            elif self.add_beat:
                beat_input = self.beat_embedding(beat_feats) # BS X T X 30
                audio_data_input = beat_input
            elif self.add_mfcc:
                mfcc_input = self.mfcc_embedding(mfcc_feats) # BS X T X 30
                mfcc_input = self.mfcc_tanh(mfcc_input) # BS X T X 30
                audio_data_input = mfcc_input

            audio_data_input = audio_data_input.transpose(1, 2)
            audio_decoder_out, _ = self.audio_decoder(audio_data_input, padding_mask, pos_vec)
            # BS X T X D

        decoder_fusion = torch.cat((decoder_out, audio_decoder_out), dim=2)

        feats_out = self.final_mapping(decoder_fusion) # BS X T X (48*d_out)
        # relu_out = self.final_relu(feats_out) # BS X T X (48*d_out)
        # relu_out = relu_out.view(bs, max_len, n_joints, -1) # BS X T X 48 X d_out
        # cls_out = self.final_cls(relu_out) # BS X T X 48 X n_cls

        feats_out = feats_out.view(bs, max_len, n_joints, -1) # BS X T X 48 X d_out
        cls_out = self.final_cls(feats_out) # BS X T X 48 X n_cls

        return cls_out

    def adjust_temp(self, prob):
        # pi_pdf: num_mixtures
        prob = torch.log(prob) / self.temperature
        prob -= prob.max()
        prob = torch.exp(prob)
        prob /= prob.sum()

        return prob

    def get_mask_pos(self, timesteps):
        # Get timesteps for extracting mask padding
        f_num = timesteps
        f_num_tensor = torch.from_numpy(np.array([f_num])).long()
        mask = torch.arange(self.max_timesteps).expand(1, self.max_timesteps) < f_num_tensor.unsqueeze(1)
        # 1 X max_timesteps
        bs = 1
        mask = mask.unsqueeze(0).repeat(bs, 1, 1) # BS X 1 X max_timesteps

        # Get position vec for position-wise embedding
        pos_vec = torch.arange(timesteps)+1 # timesteps
        pos_vec = pos_vec.unsqueeze(0) # 1 X timesteps
        pos_paddings = torch.zeros((1, self.max_timesteps-timesteps)).long()
        padding_pos_vec = torch.cat((pos_vec, pos_paddings), dim=1) # 1 X max_timesteps
        decoder_pos_vec = padding_pos_vec.unsqueeze(0).repeat(bs, 1, 1) # BS X 1 X max_timesteps 

        return mask, decoder_pos_vec

    def given_start_inference(self, init_pose_seq, timesteps, device, mfcc_feats=None, beat_feats=None):
        # init_pose_seq: bs X T' X 48
        # mfcc: bs x T X 26
        # beat: bs X T
        # Manual setting, timesteps should <= max_timesteps
        assert init_pose_seq.size()[0] == 1
        bs = 1
        self.device = device

        decoder_input = torch.zeros(bs, self.max_timesteps, 48) # BS X T X 51
        decoder_input.fill_(self.num_cls)
        
        given_timesteps = init_pose_seq.size()[1] # the number of given timesteps in start
        decoder_input[0, :given_timesteps, :] = init_pose_seq

        decoder_input = decoder_input.to(device).long()

        final_res = torch.zeros(bs, timesteps, 48)
        final_res[0, :given_timesteps, :] = init_pose_seq

        bs, max_len, n_joints = decoder_input.shape

        for t_idx in range(given_timesteps, timesteps):
            
            data_input = self.input_embed(decoder_input) # BS X T X 48 X D
            data_input = data_input.view(bs, max_len, -1) # BS X T X (48*D)

            data_input = data_input.transpose(1, 2)  # BS X (48*D) X T
            
            mask, decoder_pos_vec = self.get_mask_pos(t_idx)
            # BS X 1 X max_timesteps, BS X 1 X max_timesteps

            mask = mask.to(device)
            decoder_pos_vec = decoder_pos_vec.to(device)

            decoder_out, _ = self.decoder(data_input, mask, decoder_pos_vec) # BS X T X D

            # Audio Transformer Part
            if self.add_mfcc and self.add_beat:
                mfcc_input = self.mfcc_embedding(mfcc_feats) # BS X T X 30
                mfcc_input = self.mfcc_tanh(mfcc_input) # BS X T X 30
                beat_input = self.beat_embedding(beat_feats) # BS X T X 30
                audio_data_input = torch.cat((mfcc_input, beat_input), dim=2) # BS X T X 60
            elif self.add_beat:
                beat_input = self.beat_embedding(beat_feats) # BS X T X 30
                audio_data_input = beat_input
            elif self.add_mfcc:
                mfcc_input = self.mfcc_embedding(mfcc_feats) # BS X T X 30
                mfcc_input = self.mfcc_tanh(mfcc_input) # BS X T X 30
                audio_data_input = mfcc_input

            audio_data_input = audio_data_input.transpose(1, 2)
            audio_decoder_out, _ = self.audio_decoder(audio_data_input, mask, decoder_pos_vec)
            # BS X T X D

            decoder_fusion = torch.cat((decoder_out, audio_decoder_out), dim=2)

            feats_out = self.final_mapping(decoder_fusion) # BS X T X (48*d_out)
            feats_out = feats_out.view(bs, max_len, n_joints, -1) # BS X T X 48 X d_out
            cls_out = self.final_cls(feats_out) # BS X T X 48 X n_cls

            cls_out = F.softmax(cls_out, dim=3) # BS X T X 48 X n_cls
            cls_out = cls_out[0, t_idx-1, :, :] # 48 X n_cls, Notice the index diff from zero start!!!
            res_list = []
            for j_idx in range(n_joints):
                probs = cls_out[j_idx, :] # n_cls
                probs = self.adjust_temp(probs)
                dist = torch.distributions.Categorical(probs)
                res = dist.sample()
                res_list.append(res)
            pred_skeleton = torch.stack(res_list) # 48 

            final_res[0, t_idx, :] = pred_skeleton
        
            # Add sampled result to next position, update decoder_input
            
            decoder_input[0, t_idx, :] = pred_skeleton

        return final_res # BS X T X 48

    
    def unlimed_seq_generation(self, init_pose_seq, timesteps, device, mfcc_feats=None, beat_feats=None):
        # Make the generated seq aligh with gt actual length
        # init_pose_seq: bs X T' X 48
        # mfcc: bs x T X 26
        # beat: bs X T
        # In this case, timesteps can > max_timesteps, since we'd like to see if it can generate unlimited length
        assert init_pose_seq.size()[0] == 1
        bs = 1
        self.device = device

        given_timesteps = init_pose_seq.size()[1] # the number of given timesteps in start
        decoder_input = torch.zeros(bs, given_timesteps, 48) # BS X T X 51
        decoder_input[0, :given_timesteps, :] = init_pose_seq

        decoder_input = decoder_input.to(device).long()

        final_res = []

        bs, _, n_joints = decoder_input.shape

        ori_max_timesteps = self.timesteps
        for t_idx in range(given_timesteps, timesteps):
            
            data_input = self.input_embed(decoder_input) # BS X N X 48 X D

            curr_len = data_input.size()[1]
            data_input = data_input.view(bs, curr_len, -1) # BS X T X (48*D)
            data_input = data_input.transpose(1, 2) # BS X (48*D) X T

            self.max_timesteps = curr_len
            mask, decoder_pos_vec = self.get_mask_pos(curr_len) # No paddings actually since we use max_timesteps = t_idx
            # BS X 1 X max_timesteps, BS X 1 X max_timesteps

            mask = mask.to(device)
            decoder_pos_vec = decoder_pos_vec.to(device)

            decoder_out, _ = self.decoder(data_input, mask, decoder_pos_vec) # BS X Tn X D

            # Audio Transformer Part
            if self.add_mfcc and self.add_beat:
                mfcc_input = self.mfcc_embedding(mfcc_feats[:, t_idx-ori_max_timesteps:t_idx, :]) # BS X T X 30
                mfcc_input = self.mfcc_tanh(mfcc_input) # BS X T X 30
                beat_input = self.beat_embedding(beat_feats[:, t_idx-ori_max_timesteps:t_idx, :]) # BS X T X 30
                audio_data_input = torch.cat((mfcc_input, beat_input), dim=2) # BS X T X 60
            elif self.add_beat:
                beat_input = self.beat_embedding(beat_feats[:, t_idx-ori_max_timesteps:t_idx, :]) # BS X T X 30
                audio_data_input = beat_input
            elif self.add_mfcc:
                mfcc_input = self.mfcc_embedding(mfcc_feats[:, t_idx-ori_max_timesteps:t_idx, :]) # BS X T X 30
                mfcc_input = self.mfcc_tanh(mfcc_input) # BS X T X 30
                audio_data_input = mfcc_input

            audio_data_input = audio_data_input.transpose(1, 2)
            audio_decoder_out, _ = self.audio_decoder(audio_data_input, mask, decoder_pos_vec)
            # BS X T X D

            decoder_fusion = torch.cat((decoder_out, audio_decoder_out), dim=2)

            feats_out = self.final_mapping(decoder_fusion) # BS X T X (48*d_out)
            feats_out = feats_out.view(bs, curr_len, n_joints, -1) # BS X T X 48 X d_out
            cls_out = self.final_cls(feats_out) # BS X T X 48 X n_cls

            cls_out = F.softmax(cls_out, dim=3) # BS X T X 48 X n_cls
            cls_out = cls_out[0, -1, :, :] # 48 X n_cls, Notice the index diff from zero start!!!
            res_list = []
            for j_idx in range(n_joints):
                probs = cls_out[j_idx, :] # n_cls
                probs = self.adjust_temp(probs)
                dist = torch.distributions.Categorical(probs)
                res = dist.sample()
                res_list.append(res)
            pred_skeleton = torch.stack(res_list) # 48 

            final_res.append(pred_skeleton)
        
            # Add sampled result to next position, update decoder_input
            
            if t_idx >= ori_max_timesteps:
                decoder_input = torch.cat((decoder_input[:, 1:, :], pred_skeleton.unsqueeze(0).unsqueeze(0)), dim=1)
            else:
                decoder_input = torch.cat((decoder_input, pred_skeleton.unsqueeze(0).unsqueeze(0)), dim=1)
            
        final_res = torch.stack(final_res).unsqueeze(0) # 1 X T X 48
        final_res = torch.cat((init_pose_seq, final_res), dim=1) # 1 X T' X 48

        return final_res # BS X T X 48
