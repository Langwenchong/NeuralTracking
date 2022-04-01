
from .vis_open3d import VisualizeOpen3D
from .vis_matplotlib import VisualizeMatplotlib
from .vis_plotly import VisualizerPlotly

def get_visualizer(opt):
	

	print(opt)

	if opt.debugging_level == 1:
		return Visualizer()
	
	elif opt.debugging_level == 2: 

		if opt.visualizer.lower() == "matplotlib": 
			return VisualizeMatplotlib(opt)

		elif opt.visualizer.lower() == "open3d": 
			return VisualizeOpen3D(opt)

		elif opt.visualizer.lower() == "plotly": 
			return VisualizerPlotly(opt)
		else: 
			NotImplementedError("Current possible visualizers:")

	else: 	 
		NotImplementedError("Visualization Options")
