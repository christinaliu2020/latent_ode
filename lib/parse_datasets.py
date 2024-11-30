###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import os
import numpy as np

import torch
import torch.nn as nn

import lib.utils as utils
from lib.diffeq_solver import DiffeqSolver
from generate_timeseries import Periodic_1d
from torch.distributions import uniform
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from embeddings_processing import MouseVideoEmbeddings, variable_time_collate_fn_embeddings, variable_time_collate_fn_embeddings_keypoints
from sklearn import model_selection
import random

#####################################################################################################
def parse_datasets(args, device):
	

	def basic_collate_fn(batch, time_steps, args = args, device = device, data_type = "train"):
		#batch = torch.stack(batch)
		batch_vals = torch.stack([item['vals'] for item in batch])
		if 'frame_ids' in batch[0]:
			batch_frame_ids = torch.stack([item['frame_ids'] for item in batch])
		else:
			batch_frame_ids = None
		data_dict = {
			"data": batch_vals,
			"time_steps": time_steps}
		if batch_frame_ids is not None:
			data_dict["frame_ids"] = batch_frame_ids
		data_dict = utils.split_and_subsample_batch(data_dict, args, data_type = data_type)
		return data_dict


	dataset_name = args.dataset

	n_total_tp = args.timepoints + args.extrap
	max_t_extrap = args.max_t / args.timepoints * n_total_tp

	##################################################################

	##############################
	#mouse embeddings dataset

	# noinspection PyPackageRequirements
	if dataset_name == "mouse_embeddings":
		#embeddings_file = '/root/SSL_behavior/data/mae_embeddings/train_embeddings_mae.npy'
		train_embeddings = args.train_embeddings.split(',')
		train_labels = args.train_labels.split(',')
		test_embeddings = args.test_embeddings
		test_labels = args.test_labels
		#labels_file = 'data/calms21 embeddings/loaded_train_behaviors.npy'
		#keypoints_file = '/root/SSL_behavior/data/keypoints/loaded_train_kps.npy'
		keypoints_file = args.keypoints
		# train_embeddings_file = '/root/SSL_behavior/data/mae_embeddings/train_embeddings_mae.npy'
		# train_labels_file = 'data/calms21 embeddings/loaded_train_behaviors.npy'
		# test_embeddings_file = '/root/SSL_behavior/data/mae_embeddings/test_embeddings_mae.npy'
		# test_labels_file = 'data/calms21 embeddings/loaded_test_behaviors.npy'
		train_dataset = MouseVideoEmbeddings(train_embeddings, train_labels,
											 split_sequences=True, do_pca=True,
											 normalize=True, num_splits=40,
											 device=device)
		test_dataset = MouseVideoEmbeddings([test_embeddings], [test_labels],
											split_sequences=True, do_pca=True,
											normalize=True, num_splits=40,
											device=device)

		batch_size = min(min(len(train_labels) + len(test_labels), args.batch_size), args.n)

		# if args.keypoints:
		# 	train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
		# 								  collate_fn=lambda batch: variable_time_collate_fn_embeddings_keypoints(batch, args, device,
		# 																							   data_type="train"))
		# 	test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False,
		# 								 collate_fn=lambda batch: variable_time_collate_fn_embeddings_keypoints(batch, args, device,
		# 																							  data_type="test"))
		#
		# 	embed_dim = train_data[0]['vals'].size(-1)
		# 	keypoint_dim = train_data[0]['keypoints'].size(-1) * train_data[0]['keypoints'].size(-2)
		# 	input_dim = embed_dim + keypoint_dim
		# 	labels = np.load(labels_file)
		# 	n_labels = len(np.unique(labels))
		# 	#
		# 	# train_labels = np.load(train_labels_file)
		# 	# test_labels = np.load(test_labels_file)
		# 	# all_labels = np.concatenate([train_labels, test_labels])
		# 	# n_labels = len(np.unique(all_labels))
		# 	data_objects = {
		# 		"dataset_obj": dataset_obj,
		# 		"train_dataloader": utils.inf_generator(train_dataloader),
		# 		"test_dataloader": utils.inf_generator(test_dataloader),
		# 		"input_dim": input_dim,
		# 		"n_train_batches": len(train_dataloader),
		# 		"n_test_batches": len(test_dataloader),
		# 		"classif_per_tp": True,
		# 		"n_labels": n_labels,  # Placeholder, adjust if needed
		# 		"keypoint_window": 101
		# 	}
		#
		# 	return data_objects
		train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
									  collate_fn=lambda batch: variable_time_collate_fn_embeddings(batch, args,
																								   device,
																								   data_type="train"))
		test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False,
									 collate_fn=lambda batch: variable_time_collate_fn_embeddings(batch, args,
																								  device,
																								  data_type="test"))
		input_dim = train_dataset[0]['vals'].size(-1)
		n_labels = args.num_classes
		#
		# train_labels = np.load(train_labels_file)
		# test_labels = np.load(test_labels_file)
		# all_labels = np.concatenate([train_labels, test_labels])
		# n_labels = len(np.unique(all_labels))
		data_objects = {
			"train_dataloader": utils.inf_generator(train_dataloader),
			"test_dataloader": utils.inf_generator(test_dataloader),
			"input_dim": input_dim,
			"n_train_batches": len(train_dataloader),
			"n_test_batches": len(test_dataloader),
			"classif_per_tp": True,
			"n_labels": n_labels  # Placeholder, adjust if needed
		}

		return data_objects

	if dataset_name == "mouse_keypoints":
		embeddings_file = 'data/keypoints/loaded_train_kps.npy'
		labels_file = 'data/calms21 embeddings/train_labels.npy'

		dataset_obj = MouseVideoEmbeddings(embeddings_file, labels_file, split_sequences=True, normalize = False, num_splits=20,  device = device)

		# Split into train and test
		train_size = int(0.8 * len(dataset_obj))
		test_size = len(dataset_obj) - train_size
		train_data, test_data = torch.utils.data.random_split(dataset_obj, [train_size, test_size])

		batch_size = min(min(len(dataset_obj), args.batch_size), args.n)

		train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
									  collate_fn=lambda batch: variable_time_collate_fn_embeddings(batch, args, device,
																								   data_type="train"))
		test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False,
									 collate_fn=lambda batch: variable_time_collate_fn_embeddings(batch, args, device,
																								  data_type="test"))

		input_dim = train_data[0]['vals'].size(-1)
		labels = np.load(labels_file)
		n_labels= len(np.unique(labels))
		data_objects = {
			"dataset_obj": dataset_obj,
			"train_dataloader": utils.inf_generator(train_dataloader),
			"test_dataloader": utils.inf_generator(test_dataloader),
			"input_dim": input_dim,
			"n_train_batches": len(train_dataloader),
			"n_test_batches": len(test_dataloader),
			"classif_per_tp": True,
			"n_labels":n_labels
		}

		return data_objects
	########### 1d datasets ###########

	# Sampling args.timepoints time points in the interval [0, args.max_t]
	# Sample points for both training sequence and explapolation (test)
	distribution = uniform.Uniform(torch.Tensor([0.0]),torch.Tensor([max_t_extrap]))
	time_steps_extrap =  distribution.sample(torch.Size([n_total_tp-1]))[:,0]
	time_steps_extrap = torch.cat((torch.Tensor([0.0]), time_steps_extrap))
	time_steps_extrap = torch.sort(time_steps_extrap)[0]

	dataset_obj = None
	##################################################################
	# Sample a periodic function
	if dataset_name == "periodic":
		dataset_obj = Periodic_1d(
			init_freq = None, init_amplitude = 1.,
			final_amplitude = 1., final_freq = None, 
			z0 = 1.)

	##################################################################

	if dataset_obj is None:
		raise Exception("Unknown dataset: {}".format(dataset_name))

	dataset = dataset_obj.sample_traj(time_steps_extrap, n_samples = args.n, 
		noise_weight = args.noise_weight)

	# Process small datasets
	dataset = dataset.to(device)
	time_steps_extrap = time_steps_extrap.to(device)

	train_y, test_y = utils.split_train_test(dataset, train_fraq = 0.8)

	n_samples = len(dataset)
	input_dim = dataset.size(-1)

	batch_size = min(args.batch_size, args.n)
	train_dataloader = DataLoader(train_y, batch_size = batch_size, shuffle=False,
		collate_fn= lambda batch: basic_collate_fn(batch, time_steps_extrap, data_type = "train"))
	test_dataloader = DataLoader(test_y, batch_size = args.n, shuffle=False,
		collate_fn= lambda batch: basic_collate_fn(batch, time_steps_extrap, data_type = "test"))
	
	data_objects = {#"dataset_obj": dataset_obj, 
				"train_dataloader": utils.inf_generator(train_dataloader), 
				"test_dataloader": utils.inf_generator(test_dataloader),
				"input_dim": input_dim,
				"n_train_batches": len(train_dataloader),
				"n_test_batches": len(test_dataloader)}

	return data_objects


