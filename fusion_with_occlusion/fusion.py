# fusion.py is the main file to connect everything. 
# Run python fusion.py --datadir <path-to-RGBD-frames>

import sys
import argparse # To parse arguments 
import logging # To log info 




sys.path.append("../")  # Making it easier to load modules

# Neural Tracking Modules
from utils import image_proc
import utils.viz_utils as viz_utils
import utils.line_mesh as line_mesh_utils

# Fusion Modules 
from frame_loader import RGBDVideoLoader
from tsdf import TSDFVolume # Create main TSDF module where the 3D volume is stored
from embedded_deformation_graph import EDGraph # Create ED graph from mesh, depth image, tsdf 
from log import get_visualizer # Visualizer 
from run_model import Deformnet_runner # Neural Tracking Moudle 
from warpfield import WarpField # Connects ED Graph and TSDF/Mesh/Whatever needs to be deformed  


class DynamicFusion:
	def __init__(self,opt):
		self.vis = get_visualizer(opt) # Visualizer / Logger
		self.frameloader = RGBDVideoLoader(opt.datadir)
		self.opt = opt 
		
		self.model = Deformnet_runner()	
	
	def create_tsdf(self):
		# Need to load initial frame 
		self.target_frame = self.opt.source_frame
		source_data = self.frameloader.get_source_data(self.opt.source_frame)  # Get source data

		# TODO: Needs to be updated (Iterate over all frames and find max depth only of the semgmented object)
		max_depth = source_data["im"][-1].max()

		# Create a new tsdf volume
		self.tsdf = TSDFVolume(max_depth+1, source_data["intrinsics"], self.opt)

		# Add TSDF to visualizer
		self.vis.tsdf = self.tsdf


		# Integrate source frame 
		assert "mask" in source_data, "Source frame must contain segmented object for graph generation"
		mask = source_data["mask"]
		source_data["im"][:, mask == 0] = 0
		self.tsdf.integrate(source_data)


	def create_graph(self):
		# Assert TSDF already initialized  
		assert hasattr(self,'tsdf'), "TSDF not defined. Run create_tsdf first." 
	
		# Initialize graph 
		self.graph = EDGraph(self.tsdf)
		self.tsdf.graph = self.graph 		# Add graph to tsdf
		self.vis.graph  = self.graph 		# Add graph to visualizer 

		assert hasattr(self,'graph'),  "Graph not defined. Run create_graph first." 
		self.warpfield = WarpField(self.graph,self.tsdf)
		self.tsdf.warpfield = self.warpfield



	def register_new_frame(self): 

		# Check next frame can be registered
		if self.target_frame + self.opt.skip_rate > len(self.frameloader):
			return False,"Registration Completed"

		success = self.register_frame(self.target_frame,self.target_frame + self.opt.skip_rate)
		
		# Update frame number 
		self.target_frame = self.target_frame + self.opt.skip_rate

		return success

	def register_frame(self,source_frame,target_frame):
		"""
			Main step of the algorithm. Register new target frame 
			Args: 
				source_frame(int): which source frame id to integrate  
				target_frame(int): which target frame id to integrate 

			Returns:
				success(bool): Return whether sucess or failed in registering   
		"""

		source_frame_data = self.frameloader.get_source_data(source_frame)	
		target_frame_data = self.frameloader.get_target_data(target_frame,source_frame_data["cropper"])	
		
		# Obtain reduced graph based on visibility of graph nodes in source frame. Graph nodes are already deformed 
		reduced_graph_dict = self.tsdf.get_reduced_graph() # Assuming previous frame was used as source 
		skin_data		   = self.warpfield.skin_image(reduced_graph_dict["graph_nodes"],source_frame_data)

		# Compute optical flow and estimate transformation for graph using neural tracking

		estimated_reduced_graph_parameters = self.model(source_frame_data,\
			target_frame_data,reduced_graph_dict,skin_data)

		# good_match,bad_match = self.vis.plot_frame(source_data, target_data, graph_data, estimated_reduced_graph_parameters)

		# USE ARAP to extend to complete graph 
		# TODO add tests to visualize registerd transformations 
		estimated_complete_graph_parameters = self.model.run_arap(\
			reduced_graph_dict,
			estimated_reduced_graph_parameters,
			self.graph,self.warpfield)

		# Update parameters 
		self.warpfield.update_transformations(estimated_complete_graph_parameters)

		# Register TSDF 
		self.warpfield.deform_tsdf() # Deform world pts to target frame space and then register
		self.tsdf.integrate(target_frame_data)

		# Add new nodes to warpfield and graph if any
		update = self.warpfield.update_graph() 
		if update: 
			##################################    
			# Update deformation parameters  #
			##################################

			graph_data = self.graph.get_reduced_graph(np.ones(self.graph.shape[0],dtype=np.bool)) # Get all nodes 
			transformation_data = self.warpfield.get_transformation_wrt_graph_node() # Get their transformations     
			for k in graph_data:
				try: 
					print(k,graph_data[k].shape)
				except: 
					continue 

				
			estimated_new_graph_parameters = self.run_arap(graph_data,transformation_data,self.graph,self.warpfield)

			# TODO: Needs refactoring  
			self.warpfield.update_transformations(estimated_complete_graph_parameters)


		# Return whether sucess or failed in registering 
		return True

	def clear_frame_data(self):
		if hasattr(self,'tsdf'):  self.tsdf.clear() # Clear image information, remove deformed model 
		# if hasattr(self,'graph'): self.graph.clear()   # Remove estimated deformation from previous timesteps 	

	def __call__(self):

		# Initialize
		self.create_tsdf()
		self.create_graph()

		self.vis.init_plot()

		# Run fusion 
		while True: 

			success, msg = self.register_new_frame()
			self.vis.show() # Print data and plot registration details 
			if ~success: 
				break
			self.clear_frame_data() # Reset information 

		self.vis.create_video("./results.mp4")

if __name__ == "__main__":
	args = argparse.ArgumentParser() 
	# Path locations
	args.add_argument('--datadir', required=True,type=str,help='path to folder containing RGBD video data')

	# Arguments for tsdf
	args.add_argument('--voxel_size', default=0.005, type=float, help='length of each voxel cube in TSDF')

	# For GPU
	args.add_argument('--gpu', 	  dest='gpu', action="store_true",help='Try to use GPU for faster optimization')
	args.add_argument('--no-gpu', dest='gpu', action="store_false",help='Uses CPU')
	args.set_defaults(gpu=True)

	# Arguments for loading frames 
	args.add_argument('--source_frame', default=0, type=int, help='frame index to create the deformable model')
	args.add_argument('--skip_rate', default=1, type=int, help='frame rate while running code')


	# Arguments for debugging  
	args.add_argument('--debugging_level', default=2, type=int, help='Logging Information. 0: No logging, 1: Only print, 2: Logging + Visualization')
	args.add_argument('--visualizer', default='open3d', type=str, help='if debugging_level == 2, also plot registration')

	opt = args.parse_args()

	# Logging details 
	logging.basicConfig(stream=sys.stdout, level=logging.INFO) 

	dfusion = DynamicFusion(opt)
	dfusion()