# Python Imports
import numpy as np
import open3d as o3d

# Nueral Tracking Modules
import utils.viz_utils as viz_utils

# Fusion Modules
from .visualizer import Visualizer

class VisualizeOpen3D(Visualizer):
	def __init__(self,opt):
		super().__init__(opt)

	def get_mesh_from_tsdf(self): 

		assert hasattr(self,'tsdf'),  "TSDF not defined. Add tsdf as attribute to visualizer first." 

		verts, face, normals, colors = self.tsdf.get_mesh()  # Extract the new canonical pose using marching cubes
		canonical_mesh = o3d.geometry.TriangleMesh(
			o3d.utility.Vector3dVector(viz_utils.transform_pointcloud_to_opengl_coords(verts)),
			o3d.utility.Vector3iVector(face))
		canonical_mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype('float64') / 255)
		canonical_mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
		
		return canonical_mesh	

	def get_rendered_graph(self,trans=np.zeros((1,3))):
		"""
			Get graph in a graph structure that can be plotted using Open3D
			@params:
				trans: np.ndarray(1,3): How to translat the graph for plotting 
		"""
		assert hasattr(self,'graph'),  "Graph not defined. Add graph as attribute to visualizer first." 

		# Motion Graph
		rendered_graph = viz_utils.create_open3d_graph(
			viz_utils.transform_pointcloud_to_opengl_coords(self.graph.nodes) + trans,
			self.graph.edges)
		
		return rendered_graph

	def init_plot(self):	

		vis = o3d.visualization.Visualizer()
		vis.create_window(width=1280, height=960)
		
		canonical_mesh = self.get_mesh_from_tsdf()
		rendered_graph_nodes,rendered_graph_edges = self.get_rendered_graph()

		vis.add_geometry(canonical_mesh)
		vis.add_geometry(rendered_graph_nodes)
		vis.add_geometry(rendered_graph_edges)

		# vis.run() # Plot and halt the program

		#
		vis.poll_events()
		vis.update_renderer()