# Python Imports
import os
import numpy as np
import open3d as o3d

# Nueral Tracking Modules
from utils import image_proc
import utils.viz_utils as viz_utils
import utils.line_mesh as line_mesh_utils

# Fusion Modules
from .visualizer import Visualizer

class VisualizeOpen3D(Visualizer):
	def __init__(self,opt):
		super().__init__(opt)


	######################################
	# Helper modules 					 #	
	######################################	
	def plot(self,object_list,title,debug):
		"""
			Main Function which takes all open3d objects ans plots them
			
			@params: 
				object_list: List of open3D objects need to be plotted
				title: Title of the plot
				debug: Whether to stop program when visualizing results
		"""	



		print(f"::{title} Debug:{debug}")	
		if debug: 
			self.vis_debug = o3d.visualization.Visualizer()
			self.vis_debug.create_window(width=1280, height=960,window_name="Fusion Pipeline")
			
			for o in object_list:
				self.vis_debug.add_geometry(o)

			self.vis_debug.run() # Plot and halt the program
			self.vis_debug.destroy_window()
			self.vis_debug.close()

		else:
			if hasattr(self,'vis'):
				self.vis.clear_geometries() # Clear previous dataq
			else:	
				# Create visualization object
				self.vis = o3d.visualization.Visualizer()
				self.vis.create_window(width=1280, height=960,window_name="Fusion Pipeline")

			for o in object_list:
				self.vis.add_geometry(o)


			self.vis.poll_events()
			self.vis.update_renderer()

	@staticmethod		
	def get_mesh(verts,faces,trans=np.zeros((3,1)),color=None,normals=None):
		"""
			Create Open3D Mesh  
		"""		
		canonical_mesh = o3d.geometry.TriangleMesh(
			o3d.utility.Vector3dVector(viz_utils.transform_pointcloud_to_opengl_coords(verts)),
			o3d.utility.Vector3iVector(faces))
		if color is not None:
			color = Visualizer.get_color(color)
			canonical_mesh.vertex_colors = o3d.utility.Vector3dVector(color)
		if normals is not None: 
			canonical_mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

		canonical_mesh.translate(trans)
			
		return canonical_mesh

	def get_model_from_tsdf(self,trans=np.zeros((3,1))): 
		"""
			Create open3D object to visualize tsdf 
		"""
		assert hasattr(self,'tsdf'),  "TSDF not defined. Add tsdf as attribute to visualizer first." 
		verts, faces, normals, colors = self.tsdf.get_mesh()  # Extract the new canonical pose using marching cubes
		return self.get_mesh(verts,faces,trans=trans,color=colors,normals=normals)	

	def get_deformed_model_from_tsdf(self,trans=np.zeros((3,1))):
		assert hasattr(self,'tsdf'),  "TSDF not defined. Add tsdf as attribute to visualizer first." 

		verts,faces,normals,colors = self.tsdf.get_deformed_model()
		return self.get_mesh(verts,faces,trans=trans,color=colors,normals=normals)

	@staticmethod	
	def get_rendered_graph(nodes,edges,color=None,trans=np.zeros((1,3))):
		"""
			Get graph in a graph structure that can be plotted using Open3D
			@params:
				color: Color of nodes (could be a label, rgb color)
				trans: np.ndarray(1,3): Global Translation of the graph for plotting 
		"""
		color = Visualizer.get_color(color) # Get color

		# Motion Graph
		rendered_graph = viz_utils.create_open3d_graph(
			viz_utils.transform_pointcloud_to_opengl_coords(nodes) + trans,
			edges,
			color=color)
		
		return rendered_graph

	def get_rendered_reduced_graph(self,trans=np.zeros((1,3))):
		"""
			Get graph in a graph structure that can be plotted using Open3D
			@params:
				color: Color of nodes (could be a label, rgb color)
				trans: np.ndarray(1,3): Global Translation of the graph for plotting 
		"""
		assert hasattr(self,'tsdf'),  "TSDF not defined. Add tsdf as attribute to visualizer first." 
		assert hasattr(self.tsdf,'reduced_graph_dict'),  "Visible nodes not calculated. Can't show reduded graph" 

		nodes = self.tsdf.reduced_graph_dict["valid_nodes_at_source"]
		edges = self.tsdf.reduced_graph_dict["graph_edges"]
		
		rendered_graph = self.get_rendered_graph(nodes,edges,trans=trans)
		return rendered_graph

	def get_rendered_deformed_graph(self,trans=np.zeros((1,3))):
		assert hasattr(self,'tsdf'),  "TSDF not defined. Add tsdf as attribute to visualizer first." 
		assert hasattr(self,'warpfield'),  "Warpfield not defined. Add warpfield as attribute to visualizer first." 
		
		nodes = self.warpfield.get_deformed_nodes()
		edges = self.warpfield.graph.edges

		color = self.tsdf.reduced_graph_dict["valid_nodes_mask"]

		return self.get_rendered_graph(nodes,edges,color=color,trans=trans)


	def get_source_RGBD(self,trans=np.zeros((3,1))):
		assert hasattr(self,'warpfield'),  "Warpfield not defined. Add warpfield as attribute to visualizer first." 

		if hasattr(self.warpfield,'source_im'):
			source_pcd = viz_utils.get_pcd(self.warpfield.source_im) # Get point cloud with max 10000 points
		elif hasattr(self.tsdf,'im'):
			source_pcd = viz_utils.get_pcd(self.tsdf.im) # Get point cloud with max 10000 points

		source_pcd.translate(trans)

		return source_pcd

	def get_target_RGBD(self,trans=np.zeros((3,1))):
		"""
			Get Target image from TSDF
			Plot as point cloud  
		"""	
		assert hasattr(self,'tsdf'),  "TSDF not defined. Add tsdf as attribute to visualizer first." 
		assert hasattr(self.tsdf,'im'),  "Target image not defined. Update/integrate target image to tsdf first." 

		target_pcd = viz_utils.get_pcd(self.tsdf.im) # Get point cloud with max 10000 points

		# Update boundary mask color
		# boundary_points = np.where(target_data["target_boundary_mask"].reshape(-1) > 0)[0]
		# points_color = np.asarray(target_pcd.colors)
		# points_color[boundary_points, 0] = 1.0
		# target_pcd.colors = o3d.utility.Vector3dVector(points_color)  # Mark boundary points in read


		target_pcd.translate(trans)

		return target_pcd


	# Make functions defined in sub-classes based on method used 
	def plot_skinned_model(self,debug=True):
		"""
			Plot the skinning of the mesh 
		"""	

		color_list = self.get_color(np.arange(self.graph.nodes.shape[0]+1)) # Common color for graph and mesh 
		color_list[-1] = 0. # Last/Background color is black
		print(color_list)
		rendered_graph = self.get_rendered_graph(self.graph.nodes,self.graph.edges,color=color_list,trans=np.array([0,0,0.01]))
		
		verts, faces, normals, _ = self.tsdf.get_mesh()  # Extract the new canonical pose using marching cubes
		vert_anchors,vert_weights,valid_verts = self.warpfield.skin(verts)
		print(valid_verts)
		mesh_colors = np.array([vert_weights[i,:]@color_list[vert_anchors[i,:],:] for i in range(verts.shape[0])])
		
		reshape_gpu_vol = [verts.shape[0],1,1]        
		deformed_vertices = self.warpfield.deform(verts,vert_anchors,vert_weights,reshape_gpu_vol,valid_verts)    

		mesh = self.get_mesh(deformed_vertices,faces,color=mesh_colors,normals=normals)	




		self.plot([mesh] + rendered_graph,"Skinned Object",debug=debug)


	def plot_graph(self,color,title="Embedded Graph",debug=False):
		"""
			@parama:
				color: Color of nodes (could be a None,label, rgb color)
				debug: bool: Stop program to show the plot
		"""
		
		assert hasattr(self,'graph'),  "Graph not defined. Add graph as attribute to visualizer first." 
		rendered_graph_nodes,rendered_graph_edges = self.get_rendered_graph(self.graph.nodes,self.graph.edges,color)
		self.plot([rendered_graph_nodes,rendered_graph_edges],title,debug)

		# source_pcd = self.get_source_RGBD()
		# rendered_reduced_graph_nodes,rendered_reduced_graph_edges = self.get_rendered_reduced_graph()

		# self.plot([source_pcd,rendered_reduced_graph_nodes,rendered_reduced_graph_edges],"Showing reduced graph",debug)


	def plot_deformed_graph(self,debug=False):
		
		init_graph = self.get_rendered_graph(self.graph.nodes,self.graph.edges) # Initial graph 
		if not hasattr(self,'bbox'):
			self.bbox = (init_graph[0].get_max_bound() - init_graph[0].get_min_bound()) # Compute bounding box using nodes of init graph

		bbox = self.bbox

		deformed_graph = self.get_rendered_deformed_graph(trans=np.array([1,0,0])*bbox)

		self.plot(init_graph + deformed_graph,"Deformed Graph",debug)	

	def init_plot(self,debug=False):	
		"""
			Plot the initial TSDF and graph used for registration
		"""
		title = "Initization"
		canonical_mesh = self.get_model_from_tsdf()
		rendered_graph_nodes,rendered_graph_edges = self.get_rendered_graph(self.graph.nodes,self.graph.edges)
		self.plot([canonical_mesh,rendered_graph_nodes,rendered_graph_edges],title,debug)
		
	def plot_alignment(self,source_frame_data,\
			target_frame_data,graph_data,skin_data,\
			model_data):
		"""
			Plot Alignment similiar to neural tracking

		"""

		# Params for visualization correspondence info
		weight_thr = 0.3
		weight_scale = 1

		# Source
		source_pcd = self.get_source_RGBD()

		# keep only object using the mask
		valid_source_mask = np.moveaxis(model_data["valid_source_points"], 0, -1).reshape(-1).astype(bool)
		source_object_pcd = source_pcd.select_by_index(np.where(valid_source_mask)[0])

		# Source warped
		warped_deform_pred_3d_np = image_proc.warp_deform_3d(
			source_frame_data["im"], skin_data["pixel_anchors"], skin_data["pixel_weights"], graph_data["valid_nodes_at_source"],
			model_data["node_rotations"], model_data["node_translations"]
		)

		source_warped = np.copy(source_frame_data["im"])
		source_warped[3:, :, :] = warped_deform_pred_3d_np
		warped_pcd = viz_utils.get_pcd(source_warped).select_by_index(np.where(valid_source_mask)[0])
		warped_pcd.paint_uniform_color([1, 0.706, 0])

		# TARGET
		target_pcd = self.get_target_RGBD()


		####################################
		# GRAPH #
		####################################
		rendered_graph = viz_utils.create_open3d_graph(
			viz_utils.transform_pointcloud_to_opengl_coords(graph_data["valid_nodes_at_source"] + model_data["node_translations"]), graph_data["graph_edges"])

		# Correspondences
		# Mask
		mask_pred_flat = model_data["mask_pred"].reshape(-1)
		valid_correspondences = model_data["valid_correspondences"].reshape(-1).astype(bool)
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

	def show(self,matches=None,debug=True):
		"""
			For visualizing the tsdf integration: 
			1. Source RGBD + Graph(visible nodes)   2. Target RGBD as Point Cloud 
			3. Canonical Model + Graph   			3. Deformed Model   
		"""

		# Top left
		source_pcd = self.get_source_RGBD()

		# Create bounding box for later use 
		if not hasattr(self,'bbox'):
			self.bbox = (source_pcd.get_max_bound() - source_pcd.get_min_bound())

		bbox = self.bbox
		rendered_reduced_graph_nodes,rendered_reduced_graph_edges = self.get_rendered_reduced_graph(trans=np.array([0.0, 0, 0.03]) * bbox)

		# Top right 
		target_pcd = self.get_target_RGBD(trans=np.array([1.0, 0, 0]) * bbox)

		# Bottom left
		canonical_mesh = self.get_model_from_tsdf(trans=np.array([0, -1.0, 0]) * bbox)
		rendered_graph_nodes,rendered_graph_edges = self.get_rendered_graph(self.graph.nodes,self.graph.edges,trans=np.array([0, -1.0, 0.01]) * bbox)

		# Bottom right
		deformed_mesh = self.get_deformed_model_from_tsdf(trans=np.array([1.0, -1.0, 0]) * bbox)
		# rendered_reduced_graph_nodes2,rendered_reduced_graph_edges2 = self.get_rendered_reduced_graph(trans=np.array([1.5, -1.5, 0.01]) * bbox)
		# rendered_deformed_nodes,rendered_deformed_edges = self.get_rendered_graph(self.warpfield.get_deformed_nodes(),self.graph.edges,trans=np.array([1.5, -1.5, 0.01]) * bbox)

		# Add matches
		# print(matches)
		# trans = np.array([0, 0, 0]) * bbox
		# matches[0].translate(trans)
		# matches[1].translate(trans)
		# if matches is not None:
			# vis.add_geometry(matches[0])
			# vis.add_geometry(matches[1])

		self.plot([source_pcd,rendered_reduced_graph_nodes,rendered_reduced_graph_edges,\
			target_pcd,\
			canonical_mesh,rendered_graph_nodes,rendered_graph_edges,\
			deformed_mesh],"Showing frame",debug)

		image_path = os.path.join(self.savepath,"images",f"{self.tsdf.frame_id}.png")
		if not os.path.isfile(image_path):
			self.vis.capture_screen_image(image_path) # TODO: Returns segfault