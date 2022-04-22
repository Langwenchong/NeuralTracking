# Run various tests 
import sys
import logging
sys.path.append("../")  # Making it easier to load Neural tracking modules

# Fusion modules 
from frame_loader import RGBDVideoLoader
from tsdf import TSDFVolume # Create main TSDF module where the 3D volume is stored
from embedded_deformation_graph import EDGraph # Create ED graph from mesh, depth image, tsdf 
from vis import get_visualizer # Visualizer 
from run_model import Deformnet_runner # Neural Tracking Moudle 
from warpfield import WarpField # Connects ED Graph and TSDF/Mesh/Whatever needs to be deformed  


# Testing modules 
from fusion_tests import deformation_test
from fusion_tests import arap_tests
from fusion_tests import update_graph_test

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

logging.getLogger('embedded_deformation_graph').setLevel(logging.DEBUG)
update_graph_test.test1()
logging.getLogger('embedded_deformation_graph').setLevel(logging.INFO)


# ARAP Tests Register sphere
logging.getLogger('run_model').setLevel(logging.DEBUG)
arap_tests.test1(use_gpu=True)
logging.getLogger('run_model').setLevel(logging.INFO)


print("Running deformation tests")
logging.getLogger('warpfield').setLevel(logging.DEBUG)
deformation_test.test1(use_gpu=False)
deformation_test.test1(use_gpu=True)
deformation_test.test2(use_gpu=False)
logging.getLogger('warpfield').setLevel(logging.INFO)
print("Completed deformation tests")
