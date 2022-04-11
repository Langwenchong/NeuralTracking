#  The file contains tests on the arap module
# Python Imports
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# Import Fusion Modules 
from embedded_deformation_graph import EDGraph # Create ED graph from mesh, depth image, tsdf  
from warpfield import WarpField # Connects ED Graph and TSDF/Mesh/Whatever needs to be deformed  
from vis import get_visualizer # Visualizer 
from run_model import Deformnet_runner # Neural Tracking + ARAP Moudle 

# Test imports 
from .test_utils import Dict2Class

# Base TSDF class
class TestMesh:
	def __init__(self,fopt,mesh):
		self.fopt = fopt
		self.mesh = mesh

		# Estimate normals for future use
		self.mesh.compute_vertex_normals(normalized=True)
		self.frame_id = fopt.source_frame
	def get_mesh(self):
		vertices = np.asarray(self.mesh.vertices)
		faces = np.asarray(self.mesh.triangles)			

		return vertices,faces,None,None



def test1(use_gpu=True):
	"""
		Rotating and translating sphere.

		1. Random nodes of the graph have pose estimation. Hence will use ARAP to calculate them 
		2. First 10% nodes have no pose estimation

	"""  

	fopt = Dict2Class({"source_frame":0,\
		"gpu":use_gpu,"visualizer":"open3d",\
		"datadir":"/media/srialien/Elements/AT-Datasets/DeepDeform/new_test/sphere",\
		"skip_rate":1})
	vis = get_visualizer(fopt)

	# Create sphere
	mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1)
	mesh_sphere.translate([1,-10,-2])
	mesh_sphere.paint_uniform_color([1.0,0,0]) # Red color sphere

	# Create fusion modules

	tsdf = TestMesh(fopt,mesh_sphere)
	graph = EDGraph(tsdf,vis)
	warpfield = WarpField(graph,tsdf,vis)
	model = Deformnet_runner()	

	# Add modules to vis
	vis.tsdf = tsdf
	vis.graph = graph
	vis.warpfield = warpfield

	model.graph = graph
	model.warpfield = warpfield

	# Create sphere
	# vis.plot_graph(None,title="Embedded Graph",debug=True)

	for i,invisible_node_percentage in enumerate(range(0,100,10)):
		
		visible_nodes = np.ones(graph.num_nodes,dtype=np.bool)	
		num_invisible_nodes = graph.num_nodes*invisible_node_percentage//100
		invisible_nodes = np.random.choice(graph.num_nodes,size=num_invisible_nodes,replace=False)
		visible_nodes[invisible_nodes] = False


		# Create random rotations 
		rotmat = R.from_rotvec(np.random.random(3)*np.pi).as_matrix()		
		tsdf.mesh.rotate(rotmat)
		tsdf.mesh.translate(np.random.random(3))

		# Get deformed graph positions  
		deformed_vertices = np.asarray(tsdf.mesh.vertices,dtype=np.float32)
		deformed_graph_nodes = deformed_vertices[graph.node_indices]



		# Create reduce graph dict  
		reduced_graph_dict = graph.get_reduced_graph(visible_nodes) # Get reduced graph at initial frame
		reduced_graph_dict["graph_nodes"] = warpfield.deformed_graph_nodes[visible_nodes] # update reduced graph at timestep t-1
		tsdf.reduced_graph_dict = reduced_graph_dict 	

		# warpfield.deformed_graph_nodes[visible_nodes] = deformed_graph_nodes[visible_nodes]
		# vis.plot_deformed_graph(debug=True)

		print(f"Test:{i} => percentage:{invisible_node_percentage} num_invisible_nodes:{num_invisible_nodes}/{graph.num_nodes}")

		# Get transformations to update tsdf from t-1 to t
		graph_deformation_data = {}
		graph_deformation_data['source_frame_id'] =  0
		graph_deformation_data['target_frame_id'] =  i

		graph_deformation_data["deformed_graph_nodes"] = deformed_graph_nodes[visible_nodes]
		graph_deformation_data["node_translations"] = deformed_graph_nodes[visible_nodes] - warpfield.deformed_graph_nodes[visible_nodes]
		graph_deformation_data['node_rotations'] = np.tile(rotmat[None],(reduced_graph_dict["num_nodes"],1,1))

		# Run as rigid as possible 
		estimated_complete_graph_parameters = model.run_arap(\
			reduced_graph_dict,
			graph_deformation_data,
			graph,warpfield)

		# Update warpfield parameters, warpfield maps intial frme to frame t 
		warpfield.update_transformations(estimated_complete_graph_parameters)
		# print(warpfield.rotations)
		# print(warpfield.translations)		
		# TSDF gets updated last
		tsdf.frame_id = i		

		# Evaluate results 
		print("Reconstructon Error:",np.linalg.norm(warpfield.deformed_graph_nodes[invisible_nodes]-deformed_vertices[graph.node_indices[invisible_nodes]]))

		# Plot deformed graph with different color 
		vis.plot_deformed_graph(debug=True)
