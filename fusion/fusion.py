# The code in this file performs experiment using different methods of fusion. 
# Fusion in a process to merge RGBD images at various frames to generate a canonical pose & warp field at each timestep 
# The method described by the paper is to use DynamicFusion by Richard Newcombe et al 2015 CVPR 


# Library imports
import cv2
import os
import sys
import time
import open3d as o3d
import numpy as np
import copy
from scipy.io import savemat 

import matplotlib.pyplot as plt

sys.path.append("../")  # Making it easier to load modules

# Modules
from utils import image_proc
import utils.viz_utils as viz_utils
import utils.line_mesh as line_mesh_utils




# Modules for Fusion
from frame_loader import RGBDVideoLoader
from run_model import Deformnet_runner
from tsdf import TSDFVolume
import create_graph_data_using_depth
from NeuralNRT._C import erode_mesh as erode_mesh_c
from Generate_mask import generate_mask_depth


class DynamicFusion:
	def __init__(self, seqpath):
		"""
			Initialize the class which includes, loading Deformnet, loading dataloader for frames
		
			@params:
				seqpath => Location of RGBD sequences (must contain a color dir, depth dir, and interincs.txt)
				source_frame => Source frame over which the canonical pose is generated
		"""
		self.dataloader = RGBDVideoLoader(seqpath)
		self.model = Deformnet_runner()  # Class to run deformnet on data

	@staticmethod
	def plot_frame(source_data, target_data, graph_data, model_data):

		# Params for visualization correspondence info
		weight_thr = 0.3
		weight_scale = 1

		# Source
		source_pcd = viz_utils.get_pcd(source_data["source"])

		# keep only object using the mask
		valid_source_mask = np.moveaxis(model_data["valid_source_points"], 0, -1).reshape(-1).astype(np.bool)
		source_object_pcd = source_pcd.select_by_index(np.where(valid_source_mask)[0])

		# Source warped
		warped_deform_pred_3d_np = image_proc.warp_deform_3d(
			source_data["source"], graph_data["pixel_anchors"], graph_data["pixel_weights"], graph_data["graph_nodes"],
			model_data["node_rotations"], model_data["node_translations"]
		)
		source_warped = np.copy(source_data["source"])
		source_warped[3:, :, :] = warped_deform_pred_3d_np
		warped_pcd = viz_utils.get_pcd(source_warped).select_by_index(np.where(valid_source_mask)[0])
		warped_pcd.paint_uniform_color([1, 0.706, 0])

		# TARGET
		target_pcd = viz_utils.get_pcd(target_data["target"])
		# o3d.visualization.draw_geometries([source_pcd])
		# o3d.visualization.draw_geometries([source_object_pcd])
		# o3d.visualization.draw_geometries([warped_pcd])
		# o3d.visualization.draw_geometries([target_pcd])

		####################################
		# GRAPH #
		####################################
		rendered_graph = viz_utils.create_open3d_graph(
			viz_utils.transform_pointcloud_to_opengl_coords(graph_data["graph_nodes"] + model_data["node_translations"]), graph_data["graph_edges"])

		# Correspondences
		# Mask
		mask_pred_flat = model_data["mask_pred"].reshape(-1)
		valid_correspondences = model_data["valid_correspondences"].reshape(-1).astype(np.bool)
		# target matches
		target_matches = np.moveaxis(model_data["target_matches"], 0, -1).reshape(-1, 3)
		target_matches = viz_utils.transform_pointcloud_to_opengl_coords(target_matches)

		# "Good" matches
		good_mask = valid_correspondences & (mask_pred_flat >= weight_thr)
		good_matches_set, good_weighted_matches_set = viz_utils.create_matches_lines(good_mask, np.array([0.0, 0.8, 0]),
																					 np.array([0.8, 0, 0.0]),
																					 source_pcd, target_matches,
																					 mask_pred_flat, weight_thr,
																					 weight_scale)

		bad_mask = valid_correspondences & (mask_pred_flat < weight_thr)
		bad_matches_set, bad_weighted_matches_set = viz_utils.create_matches_lines(bad_mask, np.array([0.0, 0.8, 0]),
																				   np.array([0.8, 0, 0.0]), source_pcd,
																				   target_matches, mask_pred_flat,
																				   weight_thr, weight_scale)

		####################################
		# Generate info for aligning source to target (by interpolating between source and warped source)
		####################################
		warped_points = np.asarray(warped_pcd.points)
		valid_source_points = np.asarray(source_object_pcd.points)
		assert warped_points.shape[0] == np.asarray(source_object_pcd.points).shape[
			0], f"Warp points:{warped_points.shape} Valid Source Points:{valid_source_points.shape}"
		line_segments = warped_points - valid_source_points
		line_segments_unit, line_lengths = line_mesh_utils.normalized(line_segments)
		line_lengths = line_lengths[:, np.newaxis]
		line_lengths = np.repeat(line_lengths, 3, axis=1)

		####################################
		# Draw
		####################################

		geometry_dict = {
			"source_pcd": source_pcd,
			"source_obj": source_object_pcd,
			"target_pcd": target_pcd,
			"graph": rendered_graph,
			# "deformed_graph":    rendered_deformed_graph
		}

		alignment_dict = {
			"valid_source_points": valid_source_points,
			"line_segments_unit": line_segments_unit,
			"line_lengths": line_lengths
		}

		matches_dict = {
			"good_matches_set": good_matches_set,
			"good_weighted_matches_set": good_weighted_matches_set,
			"bad_matches_set": bad_matches_set,
			"bad_weighted_matches_set": bad_weighted_matches_set
		}

		#####################################################################################################
		# Open viewer
		#####################################################################################################
		manager = viz_utils.CustomDrawGeometryWithKeyCallback(
			geometry_dict, alignment_dict, matches_dict
		)
		manager.custom_draw_geometry_with_key_callback()

	@staticmethod
	def plot_tsdf(vis, graph_data, graph_deformation_data, target_data, canonical_mesh, dmesh, bbox):
		"""
			For visualizing the tsdf integration plot: 
			1. Canonical Pose + Graph           2. Target RGBD as Point Cloud 
			3. Deformed Pose                    4. Deformed graph   
		"""
		vis.reset_view_point(True)
		vis.clear_geometries()

		vis.add_geometry(canonical_mesh)

		# Motion Graph
		rendered_graph = viz_utils.create_open3d_graph(viz_utils.transform_pointcloud_to_opengl_coords(
			graph_data["graph_nodes"]), graph_data["graph_edges"])
		vis.add_geometry(rendered_graph[0])
		vis.add_geometry(rendered_graph[1])

		# Target
		trans = np.array([1.5, 0, 0]) * bbox
		target_pcd = viz_utils.get_pcd(target_data["target"])
		# Add boundary mask
		# boundary_points = np.where(target_data["target_boundary_mask"].reshape(-1) > 0)[0]
		points_color = np.asarray(target_pcd.colors)
		# points_color[boundary_points, 0] = 1.0
		target_pcd.colors = o3d.utility.Vector3dVector(points_color)  # Mark boundary points in read
		target_pcd.translate(trans)
		vis.add_geometry(target_pcd)

		# Deformed Pose + Motion Graph
		trans = np.array([0, -1.5, 0]) * bbox
		dmesh.translate(trans)
		vis.add_geometry(dmesh)

		trans2 = np.array([1.5, -1.5, 0]) * bbox
		rendered_deformed_graph = viz_utils.create_open3d_graph(viz_utils.transform_pointcloud_to_opengl_coords(
			graph_deformation_data["deformed_graph_nodes"]) + trans2, graph_data["graph_edges"])
		vis.add_geometry(rendered_deformed_graph[0])
		vis.add_geometry(rendered_deformed_graph[1])

		vis.poll_events()
		vis.update_renderer()

	def estimate_warp_field_parameters(self, source_data, target_data, graph_data, prev_graph_deformation_data,show_frame=False):
		"""
			Run deformnet to get correspondence
		"""

		model_data = self.model(source_data["source"], target_data["target"], target_data["target_boundary_mask"],
								source_data["intrinsics"],
								# graph_nodes, graph_data["graph_edges"], graph_data["graph_edges_weights"],
								graph_data["graph_nodes"], graph_data["graph_edges"], graph_data["graph_edges_weights"],
								graph_data["graph_clusters"],
								graph_data["pixel_weights"], graph_data["pixel_anchors"],
								graph_data["num_nodes"],
								prev_rot = None if prev_graph_deformation_data is None else prev_graph_deformation_data["node_rotations"],
								prev_trans = None if prev_graph_deformation_data is None else prev_graph_deformation_data["node_translations"])

		if show_frame:
			self.plot_frame(source_data, target_data, graph_data, model_data)

		return model_data

	def update_warpfield(self, updated_mesh_verts, tsdf, live_frame,graph_data,plot_update=False):

		dist, nearest_node = tsdf.warpfield.kdtree.query(updated_mesh_verts, k=1)
		unreachable_verts = np.where(dist > graph_data["node_coverage"])[0]

		if len(unreachable_verts) > 0:  # Ball-Radius search to find new nodes and eliminate already affected nodes
			graph_data = create_graph_data_using_depth.update_graph(updated_mesh_verts[unreachable_verts], graph_data,plot_update=plot_update)			
			
			# Update pixel anchors and weights
			mask = np.any(graph_data["pixel_anchors"] >= 0, axis=2)
			graph_data = create_graph_data_using_depth.update_live_frame_graph(live_frame,graph_data,mask)	

			tsdf.warpfield.update(graph_data)


		return graph_data

	def update_warpfield_deformation(self,graph_data,graph_deformation_data):

		# Update Node Rotation and Translation as the weighted sum of deformation of old graph nodes
		# new_node_idx = np.arange(len(graph_deformation_data["deformed_graph_nodes"]),
		# 						 len(graph_data["graph_nodes"]))
		# nn_neighbours = graph_data["graph_edges"][new_node_idx]
		# new_node_translation = np.sum(
		# 	graph_data["graph_edges_weights"][new_node_idx, :, None] * graph_deformation_data["node_translations"][
		# 															   nn_neighbours, :], axis=1)
		# new_node_rotation = np.sum(
		# 	graph_data["graph_edges_weights"][new_node_idx, :, None, None] * graph_deformation_data[
		# 																		 "node_rotations"][nn_neighbours, :,
		# 																	 :], axis=1)

		# graph_deformation_data["node_translations"] = np.concatenate(
		# 	[graph_deformation_data["node_translations"], new_node_translation], axis=0)
		# graph_deformation_data["node_rotations"] = np.concatenate(
		# 	[graph_deformation_data["node_rotations"], new_node_rotation], axis=0)
		# graph_deformation_data["deformed_graph_nodes"] = np.concatenate(
		# 	[graph_deformation_data["deformed_graph_nodes"],
		# 	 graph_data["graph_nodes"][new_node_idx] + new_node_translation], axis=0)

		# Update Rotation and Translation of new nodes as identity
		new_node_num = len(graph_data["graph_nodes"]) - len(graph_deformation_data["deformed_graph_nodes"])
		if new_node_num > 0:
			graph_deformation_data["node_translations"] = np.concatenate(
				[graph_deformation_data["node_translations"], 
				np.zeros((new_node_num,3))], axis=0)
			graph_deformation_data["node_rotations"] = np.concatenate(
				[graph_deformation_data["node_rotations"], 
				np.tile(np.eye(3).reshape((1,3,3)),(new_node_num,1,1))], axis=0)
		
			graph_deformation_data["deformed_graph_nodes"] = np.concatenate(
				[graph_deformation_data["deformed_graph_nodes"],
				 graph_data["graph_nodes"][-new_node_num:]], axis=0)		

		return graph_deformation_data

	def run(self, source_frame=0, skip=1,keyframe_interval=50,voxel_size=0.01):
		"""
			Run dynamic fusion using NNRT
		"""

		source_data = self.dataloader.get_source_data(source_frame)  # Get source data
		graph_data = self.dataloader.get_graph(source_frame, source_data["cropper"])
		graph_deformation_data = None

		# Create a new tsdf volume
		max_depth = np.max(source_data["source"][-1])  # Needs to be updated

		tsdf = TSDFVolume(max_depth+1, voxel_size, source_data["intrinsics"], graph_data, use_gpu=False)

		tsdf_input = source_data["source"].copy()
		
		mask = np.any(graph_data["pixel_anchors"] >= 0, axis=2)
		tsdf_input[:, mask < 1] = 0
		tsdf.integrate(tsdf_input, None)


		# Initialize
		vis = o3d.visualization.Visualizer()
		vis.create_window(width=1280, height=960)

		# Source
		source_pcd = viz_utils.get_pcd(source_data["source"])
		vis.add_geometry(source_pcd)
		# Pose
		bbox = (source_pcd.get_max_bound() - source_pcd.get_min_bound())
		verts, face, normals, colors = tsdf.get_mesh()  # Extract the new canonical pose using marching cubes

		mesh = o3d.geometry.TriangleMesh(
			o3d.utility.Vector3dVector(viz_utils.transform_pointcloud_to_opengl_coords(verts)),
			o3d.utility.Vector3iVector(face))
		mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype('float64') / 255)
		mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
		o3d.io.write_triangle_mesh(os.path.join(self.dataloader.savepath, "canonical_model",
										 f"{source_frame}.ply"), mesh)
		trans = np.array([-1, 0, 0]) * bbox
		mesh.translate(trans)
		vis.add_geometry(mesh)
		# vis.run()

		# Motion Graph
		rendered_graph = viz_utils.create_open3d_graph(
			viz_utils.transform_pointcloud_to_opengl_coords(graph_data["graph_nodes"]) + trans,
			graph_data["graph_edges"])
		vis.add_geometry(rendered_graph[0])
		vis.add_geometry(rendered_graph[1])

		vis.poll_events()
		vis.update_renderer()

		updated_graph = False

		# Run from source_frame + 1 to T
		for i in range(source_frame + skip, self.dataloader.get_video_length(), skip):
			
			# If image plotted that means the computation for the frame has been completed
			image_path = os.path.join(self.dataloader.savepath, "images", f"{i}.png")
			if os.path.isfile(image_path):
				updated_graph = False
				continue

			if i != source_frame:
				
				target_data = self.dataloader.get_target_data(i, source_data["cropper"])  # Get input data

				deformation_path = os.path.join(self.dataloader.savepath, "deformation", f"{target_data['target_index']}.mat")
				# Extract result
				if os.path.isfile(deformation_path):
					graph_deformation_data = self.dataloader.load_dict(deformation_path)
				else:

					if updated_graph == False:			
						prev_graph_path = os.path.join(self.dataloader.savepath, "updated_graph",f"{target_data['target_index']-1}.mat")
						if os.path.isfile(prev_graph_path):
							graph_data = self.dataloader.load_dict(prev_graph_path)

					graph_deformation_data = self.estimate_warp_field_parameters(source_data, target_data, graph_data,graph_deformation_data,
												show_frame=False)
					print(
						f"Estimated Warpfield Parameters for Frame:{i} Info: {graph_deformation_data['convergence_info']}")
					savemat(deformation_path,graph_deformation_data) # Save results
				
				# Perform surface fusion with the live frame
				tsdf_path = os.path.join(self.dataloader.savepath, "tsdf", f"{target_data['target_index']}.pkl")
				if os.path.isfile(tsdf_path):
					tsdf.load_volume(tsdf_path)
				else:
					st = time.time()
					tsdf.integrate(target_data["target"], graph_deformation_data)
					en = time.time()
					print(f"TSDF warped to live Frame:{i} time:{en - st}")
					tsdf.save_volume(tsdf_path)

				# Get Canonical Mesh
				mesh_path = os.path.join(self.dataloader.savepath, "canonical_model",
										 f"{target_data['target_index']}.ply")
				if os.path.isfile(mesh_path):
					canonical_mesh = o3d.io.read_triangle_mesh(mesh_path)
					verts = viz_utils.transform_pointcloud_to_opengl_coords(np.asarray(canonical_mesh.vertices))
				else:
					verts, face, normals, colors = tsdf.get_mesh()  # Extract the new canonical pose using marching cubes
					canonical_mesh = o3d.geometry.TriangleMesh(
						o3d.utility.Vector3dVector(viz_utils.transform_pointcloud_to_opengl_coords(verts)),
						o3d.utility.Vector3iVector(face))
					canonical_mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype('float64') / 255)
					canonical_mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
					o3d.io.write_triangle_mesh(mesh_path, canonical_mesh)

				# Get Deformed Mesh
				dmesh_path = os.path.join(self.dataloader.savepath, "deformed_model",
										  f"{target_data['target_index']}.ply")
				if os.path.isfile(dmesh_path):
					dmesh = o3d.io.read_triangle_mesh(dmesh_path)
				else:
					dverts,_ = tsdf.warpfield.deform_tsdf(graph_deformation_data["node_rotations"],
														graph_deformation_data["node_translations"],
														points=verts)
					dmesh = o3d.geometry.TriangleMesh(
						o3d.utility.Vector3dVector(viz_utils.transform_pointcloud_to_opengl_coords(dverts)),
						canonical_mesh.triangles)
					dmesh.vertex_colors = canonical_mesh.vertex_colors
					o3d.io.write_triangle_mesh(dmesh_path, dmesh)
				# Plot results
				# vis = o3d.visualization.Visualizer()
				# vis.create_window(width=1280,height=960)
				self.plot_tsdf(vis, graph_data, graph_deformation_data, target_data, canonical_mesh, dmesh, bbox)
				# vis.run()

				# Save image
				vis.capture_screen_image(image_path) # TODO: Returns segfault

				# vis.close()

				#### Post Processing ###############
				updated_graph_path = os.path.join(self.dataloader.savepath, "updated_graph",
										 f"{target_data['target_index']}.mat")
				if os.path.isfile(updated_graph_path):
					graph_data = self.dataloader.load_dict(updated_graph_path)
				else:

					# fig = plt.figure()
					# ax1 = fig.add_subplot(131)

					graph_data = self.update_warpfield(verts.astype(np.float32),
													   tsdf,source_data["source"],
													   graph_data,plot_update=False)
					old_anchors = graph_data["pixel_anchors"].copy()
					# ax1.imshow(old_anchors)

					
					graph_deformation_data = self.update_warpfield_deformation(graph_data,graph_deformation_data)
					
					# graph_data,graph_deformation_data = create_graph_data_using_depth.remove_unwanted_nodes(graph_data,graph_deformation_data)
					# graph_data = create_graph_data_using_depth.update_live_frame_graph(source_data["source"],graph_data,mask)	
					# ax2 = fig.add_subplot(132)
					# ax2.imshow(graph_data["pixel_anchors"])

					# ax3 = fig.add_subplot(132)
					# ax3.imshow(old_anchors - graph_data["pixel_anchors"])

					# plt.show()

				updated_graph = True
				savemat(updated_graph_path,graph_data)
				
				# # Update on key volume
				if (i - source_frame) % keyframe_interval == 0:
					# Check if mask exist
					source_data = self.dataloader.get_source_data(i)  # Update source data

					# Update pixel anchors and weights
					_,new_depth_path,new_mask_path = self.dataloader.get_frame_path(i)
					if os.path.isfile(new_mask_path):
						mask = cv2.imread(new_mask_path,0)
						mask = source_data["cropper"](mask).astype(bool)
					else:	
						mask = generate_mask_depth(new_depth_path,cropper=source_data["cropper"])
					

					print(mask.shape,source_data["source"].shape)
					graph_data["graph_nodes"] = graph_deformation_data["deformed_graph_nodes"]

					graph_data = create_graph_data_using_depth.update_live_frame_graph(source_data["source"],graph_data,mask)	


					max_depth = np.max(source_data["source"][-1])  # Needs to be updated

					tsdf = TSDFVolume(max_depth+1, voxel_size, source_data["intrinsics"], graph_data, use_gpu=False)

					tsdf_input = source_data["source"].copy()
					tsdf_input[:, mask < 1] = 0
					tsdf.integrate(tsdf_input, None)


		vis.destroy_window()


# Run the module
if __name__ == "__main__":

	seq_path = None
	source_frame = 0
	skip = 1
	keyframe_interval = 50
	voxel_size = 0.01
	if len(sys.argv) <= 1:
		raise IndexError(
			"Usage python3 fusion.py <path to data> <source frame | optional (default=0)> <skip frame | optional (default=1)> <key frame | optional (default=50)> <voxel size | optional (default=0.01)>")
	if len(sys.argv) > 1:
		seq_path = sys.argv[1]
	if len(sys.argv) > 2:
		source_frame = int(sys.argv[2])
	if len(sys.argv) > 3:
		skip = int(sys.argv[3])
	if len(sys.argv) > 4:
		keyframe_interval = float(sys.argv[4])
	if len(sys.argv) > 5:
		voxel_size = float(sys.argv[5])

	method = DynamicFusion(seq_path)
	method.run(source_frame=source_frame, skip=skip, keyframe_interval=keyframe_interval,voxel_size=voxel_size)

# method.generate_video()
