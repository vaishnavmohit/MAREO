import torch
import torch.nn as nn
import torch.nn.functional as F
from util import log
import numpy as np
import math
from modules import *
from skimage.util import random_noise
from kornia.geometry.transform import translate

# https://debuggercafe.com/adding-noise-to-image-data-for-deep-learning-data-augmentation/
def add_noise(img, device, mode):
	dtype = img.dtype
	noisy = img.cpu().numpy()
	for i in range(img.shape[0]):
		noisy[i,:,:,:] = random_noise(noisy[i,:,:,:], mode=mode, mean=0, var=0.05, clip=True)
	
	noisy_img = torch.tensor(noisy, dtype=dtype, device=device)

	return noisy_img

class TransformerEncoderLayer_qkv(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model=128, nhead=8, dim_feedforward=128, dropout=0.1, activation="relu", mlp = False):
        super(TransformerEncoderLayer_qkv, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.mlp = mlp
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        if mlp:
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout2 = nn.Dropout(dropout)
            self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer_qkv, self).__setstate__(state)

    def forward(self, src, k, v, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2, _ = self.self_attn(src, k, v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if self.mlp:
            src2 = self.activation(self.linear1(src))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        return src
    
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class Model(nn.Module):
	def __init__(self, task_gen, args):
		super(Model, self).__init__()
		# Encoder
		log.info('Building encoder...')
		self.encoder = Encoder_conv_deepfc() 
		
		# LSTM and output layers
		log.info('Building LSTM and output layers...')
		self.z_size = 128
		self.key_size = 256
		self.hidden_size = 512
		self.lstm = nn.LSTM(self.key_size , self.hidden_size, batch_first=True)
		self.lstm_in = nn.LSTM(self.z_size, self.hidden_size, batch_first=True)
		self.g_out_in = nn.Linear(self.hidden_size, 128)
		self.y_out = nn.Linear(self.key_size * 2, 2)
		
		# New addition
		self.query_size = 128
		self.query_w_out = nn.Linear(self.hidden_size, self.query_size)

		# time step:
		self.time_step = args.step

		# Transformer
		log.info('Building transformer encoder...')
		# self.n_heads = 8
		self.ga = TransformerEncoderLayer_qkv(d_model=128, nhead=4)

		# RN module terms:
		self.g_theta_hidden = nn.Linear((self.z_size+self.z_size), 512)
		self.g_theta_out = nn.Linear(512, 256)

		# Context normalization
		if args.norm_type == 'contextnorm' or args.norm_type == 'tasksegmented_contextnorm':
			self.contextnorm = True
			self.gamma1 = nn.Parameter(torch.ones(self.z_size))
			self.beta1 = nn.Parameter(torch.zeros(self.z_size))
		else:
			self.contextnorm = False

		# Nonlinearities
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.softmax = nn.Softmax(dim=1)
		# Initialize parameters
		for name, param in self.named_parameters():
			# Encoder parameters have already been initialized
			if not ('encoder' in name) and not ('confidence' in name):
				# Initialize all biases to 0
				if 'bias' in name:
					nn.init.constant_(param, 0.0)
				else:
					if 'lstm' in name:
						# Initialize gate weights (followed by sigmoid) using Xavier normal distribution
						nn.init.xavier_normal_(param[:self.hidden_size*2,:])
						nn.init.xavier_normal_(param[self.hidden_size*3:,:])
						# Initialize input->hidden and hidden->hidden weights (followed by tanh) using Xavier normal distribution with gain = 
						nn.init.xavier_normal_(param[self.hidden_size*2:self.hidden_size*3,:], gain=5.0/3.0)
					elif 'key_w' in name:
						# Initialize weights for key output layer (followed by ReLU) using Kaiming normal distribution
						nn.init.kaiming_normal_(param, nonlinearity='relu')
					elif 'query_w' in name:
						# Initialize weights for query output layer using Kaiming normal distribution
						nn.init.kaiming_normal_(param)
					elif 'g_out' in name:
						# Initialize weights for gate output layer (followed by sigmoid) using Xavier normal distribution
						nn.init.xavier_normal_(param)
					elif 'y_out' in name:
						# Initialize weights for multiple-choice output layer (followed by softmax) using Xavier normal distribution
						nn.init.xavier_normal_(param)
					elif 'transformer' in name:
						# Initialize attention weights using Xavier normal distribution
						if 'self_attn' in name:
							nn.init.xavier_normal_(param)
						# Initialize feedforward weights (followed by ReLU) using Kaiming normal distribution
						if 'linear' in name:
							nn.init.kaiming_normal_(param, nonlinearity='relu')

	def forward(self, x_seq, device):
		for t in range(x_seq.shape[0]):
			for seq_i in range(x_seq.shape[1]):
				x_coord = torch.randint(-5, 5, (1,))
				y_coord = torch.randint(-5, 5, (1,))
				translation = torch.tensor([[x_coord[0], y_coord[0]]]).to(device)
				x_seq[t,seq_i,:,:] = translate(x_seq[t,seq_i,:,:].unsqueeze(0), \
					translation.float(), padding_mode = 'border')

		y_pred_linear = []

		for m in [0, 2]:

			y_pred_linear.append(self.inner_loop(x_seq, device, m))

		y_pred_linear = torch.cat(y_pred_linear, 1)
		# import pdb; pdb.set_trace()
		y_pred_linear = self.y_out(y_pred_linear).squeeze()
		y_pred = y_pred_linear.argmax(1)
		
		return y_pred_linear, y_pred


	def inner_loop(self, x_seq, device, m):

		x_in1 = torch.cat((x_seq[:,0,:,:], x_seq[:,1,:,:]), 1).unsqueeze(1)
		x_in2 = torch.cat((x_seq[:,m+2,:,:], x_seq[:,m+3,:,:]), 1).unsqueeze(1)
		x_in = torch.cat((x_in1, x_in2), 3) # [32, 1, 96, 64]

		# x_in = torch.cat((x_seq[:,m,:,:], x_seq[:,m+1,:,:]), 1).unsqueeze(1)
		z_img = self.encoder(x_in) # B, 8, 128 each
		z_img = z_img.squeeze(1)
		
		self.task_seg = [np.arange(z_img.shape[1])]
		# import pdb; pdb.set_trace()
		# (Mohit addition)
		if self.contextnorm:
			# for keys:
			z_seq_all_seg = []
			for seg in range(len(self.task_seg)):
				z_seq_all_seg.append(self.apply_context_norm(z_img[:,self.task_seg[seg],:], self.gamma1, self.beta1))
			z_img = torch.cat(z_seq_all_seg, dim=1) # (M): cat --> stack		

		# Initialize hidden state
		hidden = torch.zeros(1, x_seq.shape[0], self.hidden_size).to(device)
		cell_state = torch.zeros(1, x_seq.shape[0], self.hidden_size).to(device)
		# Initialize retrieved key vector
		key_r = torch.zeros(x_seq.shape[0], 1, self.z_size).to(device)
		# torch.zeros(x_seq.shape[0], 1, self.z_size).to(device)

		# Memory model (extra time step to process key retrieved on final time step)
		for t in range(self.time_step):
			
			# Controller
			# LSTM
			lstm_out, (hidden, cell_state) = self.lstm_in(key_r, (hidden, cell_state))
			# Key & query output layers
			query_r = self.query_w_out(lstm_out)
			# Gates
			g = self.relu(self.g_out_in(lstm_out))
			w_z = self.ga(z_img, query_r, query_r).sum(1).unsqueeze(1) # [32, 8, 128]
			z_t = (z_img * w_z).sum(1).unsqueeze(1) 
			if t == 0:
				M_v = z_t
			else:
				M_v = torch.cat([M_v, z_t], dim=1)

			w_k = w_z.sum(dim=2) 
			key_r = g * (M_v * w_k.unsqueeze(2)).sum(1).unsqueeze(1)			

		# Task output layer
		all_g = []
		for z1 in range(M_v.shape[1]):
			for z2 in range(M_v.shape[1]):
				g_hidden = self.relu(self.g_theta_hidden(torch.cat([M_v[:,z1,:], M_v[:,z2,:]], dim=1)))
				g_out = self.relu(self.g_theta_out(g_hidden))
				all_g.append(g_out) # total length 4
		# Stack and sum all outputs from G_theta
		all_g = torch.stack(all_g, 1).sum(1) # B, 256
		
		return all_g
		

	def apply_context_norm(self, z_seq, gamma, beta):
		eps = 1e-8
		z_mu = z_seq.mean(1)
		z_sigma = (z_seq.var(1) + eps).sqrt()
		z_seq = (z_seq - z_mu.unsqueeze(1)) / z_sigma.unsqueeze(1)
		z_seq = (z_seq * gamma) + beta
		return z_seq