# Python Imports
import numpy as np
import open3d as o3d

# import tkinter
import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d

# Nueral Tracking Modules
from utils import image_proc

# Fusion Modules
from .visualizer import Visualizer

class VisualizeMatplotlib(Visualizer):
	def __init__(self,opt):
		super().__init__(opt)

	# Todo needs to updated for arbitary subplots	
	def create_fig3D(self,azim=0,elev=-90,title="Matplotlib plot"):	
		self.fig = plt.figure()
		self.ax = self.fig.add_subplot(111, projection='3d')
		self.title = title

		# Viewpoint 
		self.azim = azim
		self.elev = elev
		self.ax.view_init(azim=self.azim, elev=self.elev)
		
		# Mouse-click to update
		self.fig.canvas.mpl_connect('button_press_event', self.mouse_click)

	
	# Register mouse events
	def mouse_click(self,event):
		self.azim = self.ax.azim
		self.elev = self.ax.elev	

	def plot_tsdf_deformation(self):
		"""
			Plot tsdf deformation.
			Used for testing deformation 
		"""	

		assert hasattr(self,"fig"), "Matplotlib Figure not created. Run vis.create_fig() first"

		plt.cla()


		assert hasattr(self,"tsdf") and hasattr(self,"graph") and hasattr(self,"warpfield")	

		new_world_points,valid_points = self.warpfield.deform_tsdf()
		# Plot results

		self.ax.axis('off')
		self.ax.scatter(new_world_points[:, 0], new_world_points[:, 1], new_world_points[:, 2], marker='s', s=500,alpha=0.05)
		self.ax.scatter(self.warpfield.deformed_nodes[:, 0], self.warpfield.deformed_nodes[:, 1], self.warpfield.deformed_nodes[:, 2], s=100, c="green",alpha=1)
		self.ax.scatter(self.tsdf.world_pts[:, 0], self.tsdf.world_pts[:, 1], self.tsdf.world_pts[:, 2], marker='s', s=500, c="red",alpha=0.05)

		self.ax.set_title(self.title)

	def set_ax_lim(self,lim_min=-10,lim_max=10):
		
		if type(lim_min) == int or type(lim_min) == float: 
			lim_min = [lim_min]

		if type(lim_max) == int or type(lim_max) == float: 
			lim_max = [lim_max]

		assert (len(lim_min) == 1  and len(lim_max) == 1) or (len(lim_min) == 3  and len(lim_max) == 3), f"Axis limits must have length 1 or 3, found:{lim_min},{lim_max}"	

		if len(lim_min) == 1:
			lim_min = [lim_min[0]]*3

		if len(lim_max) == 1:
			lim_max = [lim_max[0]]*3


		assert hasattr(self,"fig"), "Matplotlib Figure not created. Run vis.create_fig() first"

		self.ax.set_xlim([lim_min[0], lim_max[0]])
		self.ax.set_ylim([lim_min[1], lim_max[1]])
		self.ax.set_zlim([lim_min[2], lim_max[2]])

	def show(self):
		plt.show()

	def draw(self,pause=0.1):	
		plt.draw()
		plt.pause(pause)	
				

	def imshow(self,im):
		print("reached here?")
		if im.dtype == np.float32 and im.max() > 1:
			im /= im.max() 
		plt.imshow(im)
		plt.show()	

