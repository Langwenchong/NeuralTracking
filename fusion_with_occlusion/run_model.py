# The code in the this file creates a class to run deformnet using torch
import os
import sys
import torch
import numpy as np

# Modules (make sure modules are visible in sys.path)
from model.model import DeformNet


import utils.utils as utils
import utils.nnutils as nnutils
import options as opt

class Deformnet_runner():
	"""
		Runs deformnet to outputs result
	"""
	def __init__(self):

		#####################################################################################################
		# Options
		#####################################################################################################

		# We will overwrite the default value in options.py / settings.py
		opt.use_mask = True
		
		#####################################################################################################
		# Load model
		#####################################################################################################

		saved_model = opt.saved_model

		assert os.path.isfile(saved_model), f"Model {saved_model} does not exist."
		pretrained_dict = torch.load(saved_model)

		# Construct model
		self.model = DeformNet().cuda()

		if "chairs_things" in saved_model:
			self.model.flow_net.load_state_dict(pretrained_dict)
		else:
			if opt.model_module_to_load == "full_model":
				# Load completely model            
				self.model.load_state_dict(pretrained_dict)
			elif opt.model_module_to_load == "only_flow_net":
				# Load only optical flow part
				model_dict = self.model.state_dict()
				# 1. filter out unnecessary keys
				pretrained_dict = {k: v for k, v in pretrained_dict.items() if "flow_net" in k}
				# 2. overwrite entries in the existing state dict
				model_dict.update(pretrained_dict) 
				# 3. load the new state dict
				self.model.load_state_dict(model_dict)
			else:
				print(opt.model_module_to_load, "is not a valid argument (A: 'full_model', B: 'only_flow_net')")
				exit()

		self.model.eval()

	def __call__(self,source_data,target_data,graph_data,skin_data):
		"""
			Main Module to run the Neural Tracking estimator 
		"""

		# Move to device and unsqueeze in the batch dimension (to have batch size 1)

		source_cuda               = torch.from_numpy(source_data["im"]).cuda().unsqueeze(0)
		target_cuda               = torch.from_numpy(target_data["im"]).cuda().unsqueeze(0)
		target_boundary_mask_cuda = torch.from_numpy(target_data["target_boundary_mask"]).cuda().unsqueeze(0)
		graph_nodes_cuda          = torch.from_numpy(graph_data["graph_nodes"]).cuda().unsqueeze(0)
		graph_edges_cuda          = torch.from_numpy(graph_data["graph_edges"]).cuda().unsqueeze(0)
		graph_edges_weights_cuda  = torch.from_numpy(graph_data["graph_edges_weights"]).cuda().unsqueeze(0)
		graph_clusters_cuda       = torch.from_numpy(graph_data["graph_clusters"]).cuda().unsqueeze(0)
		pixel_anchors_cuda        = torch.from_numpy(skin_data["pixel_anchors"]).cuda().unsqueeze(0)
		pixel_weights_cuda        = torch.from_numpy(skin_data["pixel_weights"]).cuda().unsqueeze(0)
		intrinsics_cuda           = torch.from_numpy(source_data["intrinsics"]).cuda().unsqueeze(0)

		num_nodes_cuda            = torch.from_numpy(graph_data["num_nodes"]).cuda().unsqueeze(0)
		prev_rot = None
		prev_trans = None
		# return self.run()

	# def run(self,source_cuda,target_cuda,target_boundary_mask,intrinsics,\
	# 		graph_nodes,graph_edges,graph_edges_weights,graph_clusters,\
	# 		pixel_weights,pixel_anchors,\
	# 		num_nodes):
		
		#####################################################################################################
		# Predict deformation
		#####################################################################################################

		# Run Neural Non Rigid tracking and obtain results
		with torch.no_grad():
			model_data = self.model(
				source_cuda, target_cuda, 
				graph_nodes_cuda, graph_edges_cuda, graph_edges_weights_cuda, graph_clusters_cuda, 
				pixel_anchors_cuda, pixel_weights_cuda, 
				num_nodes_cuda, intrinsics_cuda, 
				evaluate=True, split="test",
				prev_rot=prev_rot,
				prev_trans=prev_trans,
			)	

		# Post Process output   
		model_data["node_rotations"]    = model_data["node_rotations"].view(-1, 3, 3).cpu().numpy()
		model_data["node_translations"] = model_data["node_translations"].view(-1, 3).cpu().numpy()
		
		assert model_data["mask_pred"] is not None, "Make sure use_mask=True in options.py"
		model_data["mask_pred"] = model_data["mask_pred"].view(-1, opt.image_height, opt.image_width).cpu().numpy()

		# Correspondence info
		xy_coords_warped,\
		source_points, valid_source_points,\
		target_matches, valid_target_matches,\
		valid_correspondences, deformed_points_idxs, deformed_points_subsampled = model_data["correspondence_info"].values()

		model_data["target_matches"]        = target_matches.view(-1, opt.image_height, opt.image_width).cpu().numpy()
		model_data["valid_source_points"]   = valid_source_points.view(-1, opt.image_height, opt.image_width).cpu().numpy()
		# model_data["valid_target_matches"]  = valid_target_matches.view(-1, opt.image_height, opt.image_width).cpu().numpy()
		model_data["valid_correspondences"] = valid_correspondences.view(-1, opt.image_height, opt.image_width).cpu().numpy()
		model_data["deformed_graph_nodes"] = graph_data["graph_nodes"] + model_data["node_translations"]

		# TODO: ? Might be important later. 
		del model_data["flow_data"]
		
		model_data = self.dict_to_numpy(model_data)

		return model_data

	
	def dict_to_numpy(self,model_data):	
		# Convert every torch tensor to np.array to save results 
		for k in model_data:
			if type(model_data[k]) == torch.Tensor:
				model_data[k] = model_data[k].cpu().data.numpy()
				# print(k,model_data[k].shape)
			elif type(model_data[k]) == list:
				for i,r in enumerate(model_data[k]): # Numpy does not handle variable length, this will produce error 
					if type(r) == torch.Tensor:
						model_data[k][i] = model_data[k][i].cpu().data.numpy()
						# print(k,i,model_data[k][i].shape)
			elif type(model_data[k]) == dict:
				for r in model_data[k]:
					if type(model_data[k][r]) == torch.Tensor:
						model_data[k][r] = model_data[k][r].cpu().data.numpy()
						# print(k,r,model_data[k][r].shape)

		return model_data


	def run_arap(self,reduced_graph_dict,model_data, graph,warpfield):
		"""
			ARAP(as-rigid-as-possible) is used to find transformations 

		"""
		
		visible_nodes = reduced_graph_dict["visible_nodes"]		
		assert len(visible_nodes) == len(graph.nodes)
		
		N = len(graph.nodes)

		graph_nodes = warpfield.deformed_graph_nodes

		R_current = np.tile(np.eye(3)[None],(N,1,1))
		T_current = np.zeros((N,3))

		R_current[visible_nodes] = model_data["node_rotations"]
		T_current[visible_nodes] = model_data["node_translations"]

		# Initialize based on 
		non_visible_nodes = np.where(~visible_nodes)[0]
		print("Non Visible nodes:",non_visible_nodes)
		# Sort them based on number of neigbours in visible nodes
		neigbours_visibility = np.sum(visible_nodes[graph.edges[non_visible_nodes]],axis=1) # Kx8
		print("Neigbours visibility",neigbours_visibility)

		sorted_indices = np.argsort(neigbours_visibility)
		print("Sort:",non_visible_nodes[sorted_indices])

		updated_nodes = visible_nodes.copy()
		for ind in sorted_indices[::-1]: # Use in descending order

			num_edges = graph.edges.shape[1] - np.sum(graph.edges[ind]==-1) # -1 values denote no edge 

			updated_neighbours = graph.edges[ind,:num_edges]
			updated_neighbours = np.where(updated_nodes[updated_neighbours])[0]
			# Find the closest neighbour
			closest_neighbour = np.argmin(graph.edges_weights[ind,updated_neighbours])
			closest_neighbour = graph.edges[ind,closest_neighbour]

			R_current[ind] = R_current[closest_neighbour]
			T_current[ind] = R_current[ind]@(graph_nodes[ind]- graph_nodes[closest_neighbour])\
								+ T_current[closest_neighbour]\
								+ graph_nodes[closest_neighbour] - graph_nodes[ind]
 
			updated_nodes[ind] = True

		init_node_position_cuda   = torch.from_numpy(reduced_graph_dict["graph_nodes"]).cuda()		
		deformed_node_position_cuda= torch.from_numpy(model_data["deformed_graph_nodes"]).cuda()		

		visible_nodes_indices_cuda= torch.from_numpy(np.where(visible_nodes)[0]).cuda()

		graph_nodes_cuda          = torch.from_numpy(graph_nodes).cuda()
		graph_edges_cuda          = torch.from_numpy(graph.edges).cuda()
		graph_edges_weights_cuda  = torch.from_numpy(graph.edges_weights).cuda()
		graph_clusters_cuda       = torch.from_numpy(graph.clusters).cuda().unsqueeze(0)

		R_current_cuda 			  =	torch.from_numpy(R_current).cuda()
		T_current_cuda 			  =	torch.from_numpy(T_current).cuda()
		# Update parameters of complete graph using as rigid as possible similar to embedded deformation 
		arap_data = self.model.arap(visible_nodes_indices_cuda,init_node_position_cuda,deformed_node_position_cuda,\
			graph_nodes_cuda,graph_edges_cuda,graph_edges_weights_cuda,graph_clusters_cuda,
			R_current_cuda,T_current_cuda)

		arap_data = self.dict_to_numpy(arap_data)
		arap_data["deformed_graph_nodes"] = graph_nodes + arap_data["node_translations"]

		return arap_data