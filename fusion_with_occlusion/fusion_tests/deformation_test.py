#  The file contains tests on the warpfield class 

import os
import numpy as np 
from scipy.spatial.transform import Rotation as R

# Neural Tracking modules 
import options as opt

# Import Fusion Modules 
from tsdf import TSDFVolume # Create main TSDF module where the 3D volume is stored
from embedded_deformation_graph import EDGraph # Create ED graph from mesh, depth image, tsdf  
from warpfield import WarpField # Connects ED Graph and TSDF/Mesh/Whatever needs to be deformed  
from vis import get_visualizer # Visualizer 
# Test imports 
from .test_utils import Dict2Class

def create_graph_from_dict(graph_dict):
	pass

def create_tsdf(max_depth, voxel_size, cam_intr, graph_dict, use_gpu=True):
	pass

# Simple Test cases for the module
def test1(use_gpu=True,debug=False):
	"""
		Case 1: 
			Voxel Grid = 3x3 Cube
			Deformable Graph = 1-single node

			Deformation: Rotating and translating in all axis

	"""

	opt.image_width = 3
	opt.image_height = 3
	max_depth = 3

	fopt = Dict2Class({"voxel_size":1,"source_frame":0,"gpu":use_gpu,"visualizer":"matplotlib","datadir":"/tmp","skip_rate":3})
	
	vis = get_visualizer(fopt)


	cam_intr = [3, 3, 1.5, 1.5]
	graph = Dict2Class({
		'nodes' : np.array([[-0.5, -0.5, 1.0]], dtype=np.float32),
		'graph_generation_parameters' : {'node_coverage' : 1.0,'graph_neighbours': 1}
	})

	# Initialize tsdf
	tsdf = TSDFVolume(max_depth,cam_intr, fopt,vis)
	tsdf.frame_id = fopt.source_frame # Since integrate is not being called update frame id normally  
	
	warpfield = WarpField(graph,tsdf,vis)

	# Add elements to vis 
	vis.tsdf = tsdf
	vis.graph = graph
	vis.warpfield = warpfield


	vis.create_fig3D(azim=0,elev=-15,title="Deforming Cube")

	# Deformation 3
	for i, deg in enumerate(range(0, 180, fopt.skip_rate)):
		r =  R.from_rotvec([0.4*deg*np.pi/180, 0.4*deg*np.pi/180, 0.4*deg*np.pi/180])
		
		if i < 10:
			graph_deformation_data = {
				'node_rotations' : np.array([r.as_matrix()]),
				# 'node_rotations' : np.eye(3)[None],
				'node_translations' : 3*i*np.ones((1,3))/10,
			}
		else: 	
			graph_deformation_data = {
				'node_rotations' : -np.array([r.as_matrix()]),
				'node_translations' : 3*i*np.zeros((1,3))/10,
			}

		graph_deformation_data['target_frame_id'] =  i
		graph_deformation_data["node_translations"][:,[2]] = 0
		graph_deformation_data['deformed_nodes_to_target'] = warpfield.deformed_nodes + graph_deformation_data['node_translations']


		warpfield.update_transformations(graph_deformation_data)

		tsdf.frame_id = graph_deformation_data['target_frame_id'] # Update tsdf manually instead of integrate 

		vis.plot_tsdf_deformation()	
		vis.set_ax_lim(lim_min=-10,lim_max=10)
		vis.draw(pause=0.1)


def test2(use_gpu=True,debug=False):

	"""
		Case 2: 
			Voxel Grid = L-shape 
			Deformable Graph = 3-single node

			Deformation: 
				1. Rotating the graph node center node by similar to elbow movement to create a straight voxel
				2. Translating end graph nodes to extend the cube

	"""

	opt.image_width = 3
	opt.image_height = 3
	max_depth = 3
	fopt = Dict2Class({"voxel_size":1,"source_frame":0,"gpu":use_gpu,"visualizer":"matplotlib","datadir":"/tmp","skip_rate":3})

	vis = get_visualizer(fopt)


	cam_intr = [3, 3, 1.5, 1.5]
	graph = Dict2Class({
		'nodes' : np.array([[-0.5, -0.5, 1.0],
								[-3.5, -0.5, 1.0],
								[-6.5, -0.5, 1.0],
								[-0.5, -3.5, 1.0]], dtype=np.float32),
		'graph_generation_parameters' : {'node_coverage' : 1.5,'graph_neighbours': 4},
	})

	# Initialize tsdf
	tsdf = TSDFVolume(max_depth,cam_intr, fopt,vis)
	tsdf.frame_id = fopt.source_frame # Since integrate is not being called update frame id normally  
	
	tsdf.world_pts = np.concatenate([	tsdf.world_pts,
										tsdf.world_pts + np.array([-3, 0, 0]),
										tsdf.world_pts + np.array([-6, 0, 0]),
										tsdf.world_pts + np.array([0, -3, 0])], axis=0).astype(np.float32)
	tsdf._vol_dim = (12,3,3)

	warpfield = WarpField(graph,tsdf,vis)

	# Add elements to vis 
	vis.tsdf = tsdf
	vis.graph = graph
	vis.warpfield = warpfield

	vis.create_fig3D(azim=0,elev=90,title="Deforming L-shaped object")
	# Deformation 1
	for i, deg in enumerate(range(0, 90, 1)):
		r = R.from_rotvec([0, 0, -np.pi/180])
		graph_deformation_data = {
			'node_rotations': np.array([r.as_matrix(), r.as_matrix(), r.as_matrix(), np.eye(3)]),
			'node_translations': np.array(	[[0, 0, 0],
											[3/90, 3/90, 0],
											[6/90, 6/90, 0],
											[0, 0, 0]])
		}


		graph_deformation_data['target_frame_id'] = i
		graph_deformation_data['deformed_nodes_to_target'] = warpfield.deformed_nodes + graph_deformation_data['node_translations']


		warpfield.update_transformations(graph_deformation_data)

		tsdf.frame_id = graph_deformation_data['target_frame_id'] # Update tsdf manually instead of integrate 

		vis.plot_tsdf_deformation()	
		vis.set_ax_lim(lim_min=-10,lim_max=10)

		if debug:
			vis.show()
		else: 
			vis.draw(pause=(i+1)/90)

