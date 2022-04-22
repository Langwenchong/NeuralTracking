# The code in the this file creates a class to run deformnet using torch
import os
import sys
import torch
import numpy as np
import logging
# Modules (make sure modules are visible in sys.path)
from model.model import DeformNet

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

		# Set logging level 
		self.log = logging.getLogger(__name__)

	def __call__(self,source_data,target_data,graph_data,skin_data):
		"""
			Main Module to run the Neural Tracking estimator 
		"""

		# Move to device and unsqueeze in the batch dimension (to have batch size 1)

		source_cuda               = torch.from_numpy(source_data["im"]).cuda().unsqueeze(0)
		target_cuda               = torch.from_numpy(target_data["im"]).cuda().unsqueeze(0)
		target_boundary_mask_cuda = torch.from_numpy(target_data["target_boundary_mask"]).cuda().unsqueeze(0)

		# Send the canonical position to calculate arap loss, Not sure why doesn't work. Sending deformed nodes for now  
		# canonical_cuda 			  = torch.from_numpy(self.graph.nodes[graph_data["valid_nodes_mask"]]).cuda().unsqueeze(0)
		canonical_cuda 			  = torch.from_numpy(graph_data["valid_nodes_at_source"]).cuda().unsqueeze(0)

		graph_nodes_cuda          = torch.from_numpy(graph_data["valid_nodes_at_source"]).cuda().unsqueeze(0)
		graph_edges_cuda          = torch.from_numpy(graph_data["graph_edges"]).cuda().unsqueeze(0)
		graph_edges_weights_cuda  = torch.from_numpy(graph_data["graph_edges_weights"]).cuda().unsqueeze(0)
		graph_clusters_cuda       = torch.from_numpy(graph_data["graph_clusters"]).cuda().unsqueeze(0)
		pixel_anchors_cuda        = torch.from_numpy(skin_data["pixel_anchors"]).cuda().unsqueeze(0)
		pixel_weights_cuda        = torch.from_numpy(skin_data["pixel_weights"]).cuda().unsqueeze(0)
		intrinsics_cuda           = torch.from_numpy(source_data["intrinsics"]).cuda().unsqueeze(0)

		num_nodes_cuda            = torch.from_numpy(graph_data["num_nodes"]).cuda().unsqueeze(0)
		prev_rot = None
		prev_trans = None

		
		# Check all objects map have same number of nodes
		print([canonical_cuda.shape[1],graph_nodes_cuda.shape[1],graph_edges_cuda.shape[1],graph_edges_weights_cuda.shape[1]])
		assert len(set([canonical_cuda.shape[1],graph_nodes_cuda.shape[1],\
			graph_edges_cuda.shape[1],graph_edges_weights_cuda.shape[1]])) == 1,"Not all input maps to correct graph node"

		#####################################################################################################
		# Predict deformation
		#####################################################################################################

		# Run Neural Non Rigid tracking and obtain results
		with torch.no_grad():
			model_data = self.model(
				source_cuda, target_cuda, 
				canonical_cuda,
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
		model_data["deformed_nodes_to_target"] = graph_data["valid_nodes_at_source"] + model_data["node_translations"]

		model_data["source_frame_id"] = source_data["id"]
		model_data["target_frame_id"] = target_data["id"]

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
			ARAP(as-rigid-as-possible) is used to find transformations of invalid nodes. 
			Invalid nodes could be nodes not vivible during aligment or newly added nodes to the graph 
			Essentially ARAP will estiamate rigid transformations of invalid nodes based on its neigbours

			@params: 
				reduced_graph_dict: Valid nodes estimated 
					all_nodes_at_source: np.ndarray(float32) (Nx3): Graph nodes at deformed using source frame transformations
					valid_nodes_at_source: np.ndarray(float32) (Mx3): Valid Graph nodes at deformed using source frame transformations
					valid_nodes_mask: np.ndarry(bool) (Nx1): Whether nodes deformation is known or not
				model_data: 
					node_rotations: np.ndarray(float32), (Mx3x3):  Rotations of valid graph nodes to transform from source to target
					node_translations: np.ndarray(float32), (Mx3) Translations of valid graph nodes to transform from source to target
					deformed_nodes_to_target: np.ndarray(float32), (Mx3): Valid deformed graph nodes at target

			@returns: 
				arap_data: 
					node_rotations: np.ndarray(float32), (Nx3x3):  Rotations of all graph nodes to transform from source to target
            		node_translations:  np.ndarray(float32), (Nx3) Translations of all graph nodes to transform from source to target
					deformed_nodes_to_target: np.ndarray(float32), (Nx3), All deformed graph nodes  at target
					source_frame_id: int: Source frame id
					target_frame_id: int: target frame id 
		            valid_solve: valid_solve, Nodes for which solution was possible, hardcoded to true for all 
            		convergence_info: Loss numbers calculated
		"""

		# TODO Assert correct type of reduced graph dict
		# for k in reduced_graph_dict:
		# 	if "mask" in k:
		# 		assert reduced_graph_dict[k].dtype == np.bool, f"reduced_graph_dict[{k}] not bool but {reduced_graph_dict[k].dtype}"	
		# 	else:
		# 		assert reduced_graph_dict[k].dtype == np.float32,f"reduced_graph_dict[{k}] not np.float32 but {reduced_graph_dict[k].dtype}"	

		# Assert correct type of model data
		for k in ["node_rotations","node_translations","deformed_nodes_to_target"]:
			assert model_data[k].dtype == np.float32,f"model_data[{k}] not np.float32 but {model_data[k].dtype}"	


		# Assert valid node mask of correct graph structure		
		valid_nodes_mask = reduced_graph_dict["valid_nodes_mask"]		
		assert len(valid_nodes_mask) == len(self.graph.nodes), f"Valid nodes not from current graph. Expcted:{len(graph.nodes)} got:{len(valid_nodes_mask)}"


		N = len(self.graph.nodes)

		# Initialize transformations 
		R_current = np.tile(np.eye(3,dtype=np.float32)[None],(N,1,1))
		T_current = np.zeros((N,3),dtype=np.float32)

		R_current[valid_nodes_mask] = model_data["node_rotations"]
		T_current[valid_nodes_mask] = model_data["node_translations"]

		# Initialize based transformations of invalid nodes based on closest neighbour in valid nodes
		non_valid_nodes = np.where(~valid_nodes_mask)[0]
		# self.log.debug(f"Non Visible nodes:{non_valid_nodes}")

		# Sort them based on number of neigbours in visible nodes
		num_edges = np.sum(graph.edges!=-1,axis=1)
		neigbours_visibility = [ np.sum(valid_nodes_mask[graph.edges[n,:num_edges[n]]]) for n in non_valid_nodes] # Kx8
		# self.log.debug(f"Neigbours visibility:{neigbours_visibility}")

		# TODO check if mapping to correct indices
		sorted_invalid_indices = np.argsort(neigbours_visibility)
		sorted_invalid_indices = list(non_valid_nodes[sorted_invalid_indices[::-1]]) # Use in descending order 
		self.log.debug(f"Sorted Invalid Nodes:{sorted_invalid_indices}")

		assert np.all(np.sort(sorted_invalid_indices) == non_valid_nodes), f"sorted_invalid_indices:{sorted_invalid_indices} != non_valid_nodes:{non_valid_nodes}" 

		# Position of nodes at source frame
		# For non-rigid alignment: source_frame = t-1
		# For adding new nodes: source_frame = canonical frame 
		graph_nodes = reduced_graph_dict["all_nodes_at_source"]

		updated_nodes = valid_nodes_mask.copy()

		while np.sum(updated_nodes) < N: # Some nodes might have no neighbours common, update them later
			L = len(sorted_invalid_indices)
			for i in range(L):
				ind = sorted_invalid_indices[i]
				updated_neighbours = graph.edges[ind,:num_edges[ind]]
				# print("Possible Neigbours:",updated_neighbours)
				updated_neighbours = np.where(updated_nodes[updated_neighbours])[0]
				# print("Possible Neigbours indices:",updated_neighbours)
				# If no closest neighbour, append to que and continue 
				if len(updated_neighbours) == 0:
					self.log.debug(f"Found no neigbours for:{ind}")
					sorted_invalid_indices.append(ind)
					continue
					
				# Find the closest neighbour
				closest_neighbour = np.argmax(graph.edges_weights[ind,updated_neighbours]) 
				closest_neighbour = updated_neighbours[closest_neighbour]
				# print("closest neighbour index",closest_neighbour)
				closest_neighbour = graph.edges[ind,closest_neighbour]
				# print("closest neighbour:",closest_neighbour)



				R_current[ind] = R_current[closest_neighbour]
				T_current[ind] = R_current[ind]@(graph_nodes[ind]- graph_nodes[closest_neighbour])\
									+ T_current[closest_neighbour]+ graph_nodes[closest_neighbour]\
									- graph_nodes[ind]
	 							# Difference due to rotations
	 							# Difference due to tranlations
	 							# w.r.t graph_nodes

				updated_nodes[ind] = True
				self.log.debug(f"Updated transformations for:{ind},{closest_neighbour } ")
				# print(R_current[ind])
				# print(R_current[closest_neighbour])


			# Remove indices already checked
			del sorted_invalid_indices[:L]

			self.log.debug(f"Remaining node:{len(sorted_invalid_indices)}")

			# If all members of the queue have updated neighbours
			if len(sorted_invalid_indices) == 0: 
				break

			# If a seperate cluster has formed during adding nodes phase
			# or no node could be updated. 
			elif len(sorted_invalid_indices) == L:
				break	

		
		# To test without running ARAP and just the initialized nodes. Uncomment below section 
		# print("Valid rotations:",R_current[valid_nodes_mask][0:4])
		# print("Initialized invalid rotationss:",R_current[non_valid_nodes])
		# arap_data = {}
		# arap_data["node_rotations"] = R_current
		# arap_data["node_translations"] = T_current
		# arap_data["deformed_nodes_to_target"] = reduced_graph_dict["all_nodes_at_source"] + arap_data["node_translations"]
		# arap_data["source_frame_id"] = model_data["source_frame_id"]
		# arap_data["target_frame_id"] = model_data["target_frame_id"]		
		# return arap_data	

		# For ARAP make sure to send torch tensors
		source_all_nodes_cuda = torch.from_numpy(reduced_graph_dict["all_nodes_at_source"]).cuda() #  # Position of all graph nodes at source frame				
		source_node_position_cuda   = torch.from_numpy(reduced_graph_dict["valid_nodes_at_source"]).cuda()	 # Position of valid graph nodes at source frame	
		target_node_position_cuda= torch.from_numpy(model_data["deformed_nodes_to_target"]).cuda() # Position of valid nodes at target frame		

		valid_nodes_mask_cuda = torch.from_numpy(valid_nodes_mask).cuda()
		
		graph_edges_cuda          = torch.from_numpy(graph.edges).cuda()
		graph_edges_weights_cuda  = torch.from_numpy(graph.edges_weights).cuda()
		graph_clusters_cuda       = torch.from_numpy(graph.clusters).cuda().unsqueeze(0)

		R_current_cuda 			  =	torch.from_numpy(R_current).cuda()
		T_current_cuda 			  =	torch.from_numpy(T_current).cuda()


		# Send arap graph in original position for calculating arap.  
		# canonical_all_node_cuda = torch.from_numpy(graph.nodes).cuda() # Not sure why but this doens't work. Maybe the displacement changed between the current and deformed position causes problems 
		
		canonical_all_node_cuda = torch.from_numpy(reduced_graph_dict["all_nodes_at_source"]).cuda()
		

		assert source_node_position_cuda.shape[0] == target_node_position_cuda.shape[0], f"Source != Target. shapes:{source_node_position_cuda.shape} {target_node_position_cuda.shape}"

		# Check all objects map have same number of nodes
		assert len(set([canonical_all_node_cuda.shape[0],valid_nodes_mask_cuda.shape[0], 
			graph_edges_cuda.shape[0],graph_edges_weights_cuda.shape[0],
			R_current_cuda.shape[0], T_current.shape[0]])) == 1, "Not all input maps to correct graph node"

		# Update parameters of complete graph using as rigid as possible similar to embedded deformation 
		arap_data = self.model.arap(source_all_nodes_cuda,source_node_position_cuda,target_node_position_cuda,
			valid_nodes_mask_cuda,
			canonical_all_node_cuda,
			graph_edges_cuda,graph_edges_weights_cuda,graph_clusters_cuda,
			R_current_cuda,T_current_cuda)

		arap_data = self.dict_to_numpy(arap_data)
		arap_data["deformed_nodes_to_target"] = reduced_graph_dict["all_nodes_at_source"] + arap_data["node_translations"]
		arap_data["source_frame_id"] = model_data["source_frame_id"]
		arap_data["target_frame_id"] = model_data["target_frame_id"]
 
		return arap_data