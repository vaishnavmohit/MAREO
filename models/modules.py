import torch
import torch.nn as nn
from util import log
from skimage.util import random_noise


def add_noise(img, device, mode):
	dtype = img.dtype
	noisy = img.cpu().numpy()
	for i in range(img.shape[0]):
		noisy[i,:,:,:] = random_noise(noisy[i,:,:,:], mode=mode, mean=0, var=0.05, clip=True)
	
	noisy_img = torch.tensor(noisy, dtype=dtype, device=device)

	return noisy_img

class DoubleConv(nn.Module):
	"""(convolution => [BN] => ReLU) * 2"""

	def __init__(self, in_channels, out_channels, mid_channels=None):
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(mid_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.double_conv(x)

class Down(nn.Module):
	"""Downscaling with maxpool then double conv"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(2),
			DoubleConv(in_channels, out_channels)
		)

	def forward(self, x):
		return self.maxpool_conv(x)

class Enc_Conv_Mq16_dist3fc(nn.Module):
	def __init__(self):
		super().__init__()

		self.value_size = 128 #args.value_size
		self.key_size = 128 #args.key_size

		self.inc = DoubleConv(1, 64) # 64x96x64
		self.down1 = Down(64, 64) # 64x48x32
		self.down2 = Down(64, 128) # 128x24x16
		self.down3 = Down(128, 128) # 128x12x8
		self.down4 = Down(128, 128) # 128x6x4
		self.down5 = Down(128, 128) # 128x3x2
		self.relu = nn.ReLU()

		self.conv_kv = nn.Conv2d(128, 128, 1, stride=1, padding=0)

	def forward(self, x):
		x = self.inc(x)
		
		x = self.down1(x)
		x = self.down2(x)
		x = self.down3(x)
		conv_out = self.down5(self.down4(x))
		z_kv = self.relu(self.conv_kv(conv_out))		# BxCxHxW 'relu' required 
		z_kv = torch.transpose(z_kv.view(x.shape[0], self.value_size, -1), 2, 1) # B, C, HW		

		return z_kv

class Encoder_conv_deepfc(nn.Module):
	def __init__(self,):
		super(Encoder_conv_deepfc, self).__init__()
		log.info('Building convolutional encoder...')
		# Convolutional layers
		log.info('Conv layers...')
		self.conv1 = DoubleConv(1, 64) # 64x64x32
		self.conv2 = Down(64, 64) # 64x32x16
		self.conv3 = Down(64, 128) # 128x16x8
		self.conv4 = Down(128, 128) # 128x8x4
		self.conv5 = Down(128, 128) # 128x4x2

		# Nonlinearities
		self.relu = nn.ReLU()

	def forward(self, x):
		# Convolutional layers
		conv_out = self.conv1(x)
		conv_out = self.conv2(conv_out)
		conv_out = self.conv3(conv_out)
		conv_out = self.conv4(conv_out)

		# Output
		z = self.conv5(conv_out)
		return torch.transpose(z.view(x.shape[0], 128, -1), 2, 1) # bringing it to dimension B, 8, 128

class Encoder_conv_com(nn.Module):
	def __init__(self, ):
		super(Encoder_conv_com, self).__init__()
		log.info('Building convolutional encoder...')
		# Convolutional layers
		log.info('Conv layers...')
		self.conv1 = Down(1, 64) # 16x16
		self.conv2 = Down(64, 64) # 8x8
		self.conv3 = Down(64, 128) # 4x4
		self.conv4 = Down(128, 128) # 2x2
		self.conv5 = Down(128, 128) # 1x1

		# Nonlinearities
		self.relu = nn.ReLU()
		# Initialize parameters
		for name, param in self.named_parameters():
			# Initialize all biases to 0
			if 'bias' in name:
				nn.init.constant_(param, 0.0)
			# Initialize all pre-ReLU weights using Kaiming normal distribution
			# elif 'weight' in name:
			# 	nn.init.kaiming_normal_(param, nonlinearity='relu')
	def forward(self, x):
		# Convolutional layers
		conv1_out = self.conv1(x)
		conv2_out = self.conv2(conv1_out)
		conv3_out = self.conv3(conv2_out)
		conv3_out = self.conv4(conv3_out)
		z = self.conv5(conv3_out)

		return z.squeeze() 

class Encoder_conv(nn.Module):
	def __init__(self, args):
		super(Encoder_conv, self).__init__()
		log.info('Building convolutional encoder...')
		# Convolutional layers
		log.info('Conv layers...')
		self.conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
		# Fully-connected layers
		log.info('FC layers...')
		self.fc1 = nn.Linear(4*32, 4096)
		self.fc2 = nn.Linear(4096, 128)
		# Nonlinearities
		self.relu = nn.ReLU()
		# Initialize parameters
		for name, param in self.named_parameters():
			# Initialize all biases to 0
			if 'bias' in name:
				nn.init.constant_(param, 0.0)
			# Initialize all pre-ReLU weights using Kaiming normal distribution
			elif 'weight' in name:
				nn.init.kaiming_normal_(param, nonlinearity='relu')
	def forward(self, x):
		# Convolutional layers
		conv1_out = self.relu(self.conv1(x))
		conv2_out = self.relu(self.conv2(conv1_out))
		conv3_out = self.relu(self.conv3(conv2_out))
		conv3_out = self.relu(self.conv4(conv3_out))
		# Flatten output of conv. net
		conv3_out_flat = torch.flatten(conv3_out, 1) # [32, 128]
		# Fully-connected layers
		fc1_out = self.relu(self.fc1(conv3_out_flat)) # [32, 128]
		fc2_out = self.relu(self.fc2(fc1_out))
		# Output
		z = fc2_out
		return z