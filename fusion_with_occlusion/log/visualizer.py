# Visualize & log results using plotly or open3d
import os
import pynput # To pause/control and registration

class Visualizer: # Base Class which contains all the preprocessing to visualize results
	def __init__(self,opt):

		self.savepath = os.path.join(opt.datadir,"results")
		os.makedirs(self.savepath,exist_ok=True)
		os.makedirs(os.path.join(self.savepath,"images"),exist_ok=True)
		os.makedirs(os.path.join(self.savepath,"video"),exist_ok=True)
		
	def plot_graph(self,graph):
		pass
	def plot_mesh(self,mesh): 
		pass
	def plotRGBDImage(self,image):
		pass
	def visualize(self):
		pass	
	def save_image(self):
		pass
	def create_video(self,video_path,start_frame=0,frame_rate=33):
		pass