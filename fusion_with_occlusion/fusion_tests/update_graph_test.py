#  The file contains tests on the graph class, its update and other things. 
# Python Imports
import numpy as np
import open3d as o3d

# Import Fusion Modules 
from embedded_deformation_graph import EDGraph # Create ED graph from mesh, depth image, tsdf  
from warpfield import WarpField # Connects ED Graph and TSDF/Mesh/Whatever needs to be deformed  
from vis import get_visualizer # Visualizer 

# Test imports 
from .test_utils import Dict2Class

# Base TSDF class
class TestMesh:
	def __init__(self,fopt,mesh):
		self.fopt = fopt
		self.mesh = mesh
		self.visible_verts_percentage = 50

		# Estimate normals for future use
		self.mesh.compute_vertex_normals(normalized=True)
		self.frame_id = fopt.source_frame
		
		self.vertices = np.asarray(self.mesh.vertices,dtype=np.float32)
		self.faces = np.asarray(self.mesh.triangles)
		self.normals = np.asarray(self.mesh.vertex_normals)			

	@property
	def world_pts(self):
		num_nodes = self.visible_verts_percentage*self.vertices.shape[0]//100
		return self.vertices[:num_nodes]
		
	@staticmethod	
	def subsample_mesh(vertices,faces,normals,visible_verts_percentage):	
		num_nodes = visible_verts_percentage*vertices.shape[0]//100
		# print("Number of nodes:",num_nodes)
		svertices = vertices[:num_nodes]
		snormals  =  normals[:num_nodes] 
		sfaces 	  = []
		for f in faces:
			if np.any(f >= num_nodes): continue
			sfaces.append(f)
		sfaces = np.array(sfaces)
		
		return svertices,sfaces,snormals

	def set_visisble_node_percentage(self,new_visible_verts_percentage):
		if new_visible_verts_percentage <= self.visible_verts_percentage: return False
		self.visible_verts_percentage = new_visible_verts_percentage
		return True	

	def get_mesh(self):
		sub_vertices,sub_faces,sub_normals = self.subsample_mesh(self.vertices,self.faces,self.normals,self.visible_verts_percentage)
		return sub_vertices,sub_faces,sub_normals,None

	def get_canonical_model(self):
		return self.get_mesh()	


class Model:
	def __init__(self):
		pass
	def run_arap(self,graph_data,transformation_parameters,graph,warpfield):
		arap_data = {'deformed_nodes_to_target':graph.nodes,
					'node_rotations': np.tile(np.eye(3)[None],(graph.nodes.shape[0],1,1)),
					'node_translations': np.zeros((graph.nodes.shape[0],3))}
		return arap_data			


def test1():
	"""
		Creating a sphere. 
		Initially showing only a small number of indices and gradually increasing them. 
	"""  

	fopt = Dict2Class({"source_frame":0,\
		"gpu":False,"visualizer":"open3d",\
		"datadir":"/media/srialien/Elements/AT-Datasets/DeepDeform/new_test/sphere",\
		"skip_rate":1})
	vis = get_visualizer(fopt)

	# Create sphere
	mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1)
	mesh_sphere.paint_uniform_color([1.0,0,0]) # Red color sphere

	# Create fusion modules
	tsdf = TestMesh(fopt,mesh_sphere)
	graph = EDGraph(tsdf,vis)
	warpfield = WarpField(graph,tsdf,vis)
	model = Model()
	
	# Add modules to WarpField
	warpfield.model = model

	# Add modules to vis
	vis.tsdf = tsdf
	vis.graph = graph
	vis.warpfield = warpfield

	# Create sphere
	vis.plot_graph(None,title="Embedded Graph",debug=True)

	for i,visible_verts_percentage in enumerate(range(10,100,10)):

		# Find valid vertices 
		tsdf.set_visisble_node_percentage(visible_verts_percentage)
		print(f"Test:{i} => percentage:{visible_verts_percentage}")

		# Update warpfield parameters, warpfield maps intial frme to frame t 
		old_num_nodes = graph.nodes.shape[0]
		warpfield.update_graph()
		new_num_nodes = graph.nodes.shape[0]
		print("Updated Graph. New number of nodes:",new_num_nodes)
		# TSDF gets updated last
		tsdf.frame_id = i		


		canonical_mesh = vis.get_model_from_tsdf(trans=np.array([0, 0, 0]))
		graph_color = np.zeros(new_num_nodes)
		graph_color[:old_num_nodes] = 1	
		rendered_graph_nodes,rendered_graph_edges = vis.get_rendered_graph(vis.graph.nodes,vis.graph.edges,color=graph_color,trans=np.array([0, 0, 0.01]))

		# Plot deformed graph with different color 
		vis.plot([canonical_mesh,rendered_graph_nodes,rendered_graph_edges],title="Gradually increasing speher",debug=True)
