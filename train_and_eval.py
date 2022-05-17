import argparse
import os
import sys
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pdb
from csv import writer, DictWriter

# seed_everything
def seed_everything(seed: int):
	import random, os
	import numpy as np
	import torch
	import torch.nn as nn
	import torch.optim as optim
	from torch.utils.data import Dataset, DataLoader
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True

# Prevent python from saving out .pyc files
sys.dont_write_bytecode = True
# Add models and tasks to path
sys.path.insert(0, './models')
sys.path.insert(0, './tasks')
# Logging utility
from util import log

# Method for creating directory if it doesn't exist yet
def check_path(path):
	if not os.path.exists(path):
		os.mkdir(path)

class seq_dataset(Dataset):
	def __init__(self, dset, args):
		self.seq_ind = dset['seq_ind']
		self.y = dset['y']
		self.len = self.seq_ind.shape[0]
	def __len__(self):
		return self.len
	def __getitem__(self, idx):
		seq_ind = self.seq_ind[idx]
		y = self.y[idx]
		return seq_ind, y

def train(args, model, device, optimizer, epoch, all_imgs, train_loader, run_neptune):
	# Create file for saving training progress
	train_prog_dir = './train_prog/'
	check_path(train_prog_dir)
	task_dir = train_prog_dir + args.task + '/'
	check_path(task_dir)
	gen_dir = task_dir + 'm' + str(args.m_holdout) + '/'
	check_path(gen_dir)
	model_dir = gen_dir + args.model_name + '/'
	check_path(model_dir)
	run_dir = model_dir + 'run' + args.run + '/'
	check_path(run_dir)
	train_prog_fname = run_dir + 'epoch_' + str(epoch) + '.txt'
	train_prog_f = open(train_prog_fname, 'w')
	train_prog_f.write('batch loss acc\n')
	# Set to training mode 
	model.train()
	# Iterate over batches
	for batch_idx, (seq_ind, y) in enumerate(train_loader):
		# Batch start time
		start_time = time.time()
		# Use sequence indices to slice corresponding images
		x_seq = all_imgs[seq_ind,:,:]
		# Load data to device
		x_seq = x_seq.to(device)
		y = y.to(device)
		# Zero out gradients for optimizer 
		optimizer.zero_grad()
		# Run model 
		if 'MNM' in args.model_name:
			y_pred_linear, y_pred, const_loss = model(x_seq, device)
		else:
			y_pred_linear, y_pred, = model(x_seq, device) # reco, x_in 

		loss_fn = nn.CrossEntropyLoss()
		loss = loss_fn(y_pred_linear, y)

		if 'MNM' in args.model_name:
			loss += const_loss
		# Update model
		loss.backward()
		optimizer.step()
		# Batch duration
		end_time = time.time()
		batch_dur = end_time - start_time
		# Report prgoress
		if batch_idx % args.log_interval == 0:
			# Accuracy
			acc = torch.eq(y_pred, y).float().mean().item() * 100.0
			# Report 	
			log.info('[Epoch: ' + str(epoch) + '] ' + \
					 '[Batch: ' + str(batch_idx) + ' of ' + str(len(train_loader)) + '] ' + \
					 '[Loss = ' + '{:.4f}'.format(loss.item()) + '] ' + \
					 '[Accuracy = ' + '{:.2f}'.format(acc) + '] ' + \
					 '[' + '{:.3f}'.format(batch_dur) + ' sec/batch]')
			# Save progress to file
			train_prog_f.write(str(batch_idx) + ' ' +\
							   '{:.4f}'.format(loss.item()) + ' ' + \
							   '{:.2f}'.format(acc) + '\n')

	train_prog_f.close()

def test(args, model, device, all_imgs, test_loader, run_neptune):
	log.info('Evaluating on test set...')
	# Set to eval mode
	model.eval()
	# Iterate over batches
	all_acc = []
	all_loss = []
	for batch_idx, (seq_ind, y) in enumerate(test_loader):
		# Use sequence indices to slice corresponding images
		x_seq = all_imgs[seq_ind,:,:]
		# Load data to device
		x_seq = x_seq.to(device)
		y = y.to(device)
		# Run model 
		if 'MNM' in args.model_name:
			y_pred_linear, y_pred, const_loss = model(x_seq, device)
		else:
			y_pred_linear, y_pred = model(x_seq, device)[:2]
		# Loss
		loss_fn = nn.CrossEntropyLoss()
		loss = loss_fn(y_pred_linear, y)
		if 'MNM' in args.model_name:
			loss += const_loss
		all_loss.append(loss.item())
		# Accuracy
		acc = torch.eq(y_pred, y).float().mean().item() * 100.0
		all_acc.append(acc)
		# Report progress
		log.info('[Batch: ' + str(batch_idx) + ' of ' + str(len(test_loader)) + ']')
	# Report overall test performance
	avg_loss = np.mean(all_loss)
	avg_acc = np.mean(all_acc)
	log.info('[Summary] ' + \
			 '[Loss = ' + '{:.4f}'.format(avg_loss) + '] ' + \
			 '[Accuracy = ' + '{:.2f}'.format(avg_acc) + ']')
	# Save performance
	test_dir = './test/'
	check_path(test_dir)
	task_dir = test_dir + args.task + '/'
	check_path(task_dir)
	gen_dir = task_dir + 'm' + str(args.m_holdout) + '/'
	check_path(gen_dir)
	model_dir = gen_dir + args.model_name + '/'
	check_path(model_dir)
	test_fname = model_dir + 'run' + args.run + '.txt'
	test_f = open(test_fname, 'w')
	test_f.write('loss acc\n')
	test_f.write('{:.4f}'.format(avg_loss) + ' ' + \
				 '{:.2f}'.format(avg_acc))
	test_f.close()								  
	
def main():

	# Settings
	parser = argparse.ArgumentParser()
	# Model settings
	parser.add_argument('--model_name', type=str, default='ESBN_v1', help="{'ESBN', \
		'ESBN_noise', 'Transformer', 'RN', 'MAREO', 'MAREO_dist3', 'MAREO_noise', 'MAREO_RMTS', \
			'MAREO_RN', 'MAREO_SA'}")
	parser.add_argument('--norm_type', type=str, default='contextnorm', help="{'contextnorm'}")
	parser.add_argument('--encoder', type=str, default='conv', help="{'conv'}")
	# Task settings
	parser.add_argument('--task', type=str, default='same_diff', help="{'same_diff', 'RMTS', 'dist3', 'identity_rules'}")
	parser.add_argument('--train_gen_method', type=str, default='full_space', help="{'full_space', 'subsample'}")
	parser.add_argument('--test_gen_method', type=str, default='full_space', help="{'full_space', 'subsample'}")
	parser.add_argument('--n_shapes', type=int, default=100, help="n = total number of shapes available for training and testing")
	parser.add_argument('--m_holdout', type=int, default=0, help="m = number of objects (out of n) withheld during training")
	# Training settings
	parser.add_argument('--train_batch_size', type=int, default=32)
	parser.add_argument('--train_set_size', type=int, default=10000)
	parser.add_argument('--train_proportion', type=float, default=0.95)
	parser.add_argument('--lr', type=float, default=5e-4)
	parser.add_argument('--epochs', type=int, default=50)
	parser.add_argument('--log_interval', type=int, default=10)
	# Test settings
	parser.add_argument('--test_batch_size', type=int, default=100)
	parser.add_argument('--test_set_size', type=int, default=10000)
	# Device settings
	parser.add_argument('--no-cuda', action='store_true', default=False)
	parser.add_argument('--device', type=int, default=0)
	# Run number
	parser.add_argument('--run', type=str, default='1')
	# time steps
	parser.add_argument('--step', type=int, default=16)
	# Neptune
	parser.add_argument('--neptune', type=bool, default=False)
	args = parser.parse_args()

	run_neptune = None

	log.info('args: ')
	log.info(args)

	# seed_everything
	seed_everything(int(args.run)) #42

	# Set up cuda	
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda:" + str(args.device) if use_cuda else "cpu")
	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

	# Randomly assign objects to training or test set
	all_shapes = np.arange(args.n_shapes)
	np.random.shuffle(all_shapes)
	if args.m_holdout > 0:
		train_shapes = all_shapes[args.m_holdout:]
		test_shapes = all_shapes[:args.m_holdout]
	else:
		train_shapes = all_shapes
		test_shapes = all_shapes
	# Generate training and test sets
	task_gen = __import__(args.task)
	log.info('Generating task: ' + args.task + '...')
	args, train_set, test_set = task_gen.create_task(args, train_shapes, test_shapes)
	# Convert to PyTorch DataLoaders
	train_set = seq_dataset(train_set, args)
	train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True)
	test_set = seq_dataset(test_set, args)
	test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True)

	# Load images
	all_imgs = []
	for i in range(args.n_shapes):
		img_fname = './imgs/' + str(i) + '.png'
		img = torch.Tensor(np.array(Image.open(img_fname))) / 255.
		all_imgs.append(img)
	all_imgs = torch.stack(all_imgs, 0)

	# Create model
	model_class = __import__(args.model_name)
	model = model_class.Model(task_gen, args).to(device)
	log.info('Model is...')
	log.info(model)

	# Append relevant hyperparameter values to model name
	# args.model_name = args.model_name + '_' + args.norm_type + '_lr' + str(args.lr)

	# Create optimizer
	log.info('Setting up optimizer...')
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	# Train
	log.info('Training begins...')
	for epoch in range(1, args.epochs + 1):
		# Training loop
		train(args, model, device, optimizer, epoch, all_imgs, train_loader, run_neptune)
	
	# save the model:
	dir = '/users/mvaishn1/data/data/mvaishn1/esbn/weight/'
	save_weight = dir + args.model_name + '_' + str(args.m_holdout) + '.pt'
	# torch.save(model.state_dict(), save_weight)

	# Test model
	test_acc = test(args, model, device, all_imgs, test_loader, run_neptune)

	# Dictionary
	dict = vars(args)
	dict['test_acc'] = test_acc

	# list of column names 	
	field_names = dict.keys()
	
	## Open your CSV file in append mode
	## Create a file object for this file
	with open('result_sd.csv', 'a') as f_object:
		
		# Pass the file object and a list 
		# of column names to DictWriter()
		# You will get a object of DictWriter
		dictwriter_object = DictWriter(f_object, fieldnames=field_names)
	
		#Pass the dictionary as an argument to the Writerow()
		dictwriter_object.writerow(dict)
	
		#Close the file object
		f_object.close()

if __name__ == '__main__':
	main()