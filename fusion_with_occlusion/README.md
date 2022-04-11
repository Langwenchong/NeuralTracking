# Perform Non-Rigid Registration Using Neural Tracking + DynamicFusion

# Run
```
python3 fusion.py --datadir <path-to-folder>
``` 

Each datadir folder contains
1. color: folder containing rgb images as 00%d.png
2. depth: folder containing depth image as 00%d.png 
3. intrinsics.txt: 4x4 camera intrinsic matrix 
4. graph_config.json (optional): parameters to generate graph 

# Dependencies 
```
pip install pynput pycuda cupy pykdtree
```  

# Files: 
- [ ] fusion.py 					// Main file for fusion based on dynamic fusion 
- [ ] frameLoader.py 				// Loads RGBD frames 
- [ ] tsdf.py 						// Class to store fused model, deformed model using warpfield and its volumetric represenation
- [ ] embedded_deformation_graph.py // Class to strore + update graph data 
- [ ] warpfield.py 					// Stores skinning weights and transformation parameters of each timestep 
- [ ] log/visualizer.py 			// Base class to visualize details using open3d, plotly, matplotlib  
- [ ] run_model.py 					// Run Neural Tracking to estimate transformation parameters
- [ ] evaluate.py 					// Run different evaluations such as deformation error and geometric error defined in DeepDeform

# Todo:- 
2. Update max depth in fusion.py, currently sending maximum depth of first frame. Can do better. Need to calculate the bound of the segmented object in complete video
3. Refactor TSDF code, one bigger kernel better that smaller kernel but we can create a folder dedicated to TSDF for better understanding
6. Check imports in each file and remove redudant imports of python packages
7. Deleting optical flow data for now. Might need to saved later. 
9. Find out usage of target_boundary_mask => Only used in gt comparision. 
13. ARAP updating all nodes for testing. Need to change it to only affected nodes. Rest copied from original data. Needs testing/debugging 
13. If some cluster has not nodes in reduced cluster what shoud we do ?  
14. Visvualize num_iterations hyperparamater for erosotion while adding new nodes 
15. Change from euclidean distance to geodesic distance for computing edges for graph when new nodes are added 
16. Replace direct get such as tsdf.canonical_model, warpfiled.get_deformed_model with get_canonical_model, get_deformed_graph_nodes 
17. Rename deformed_graph_nodes to deformed_nodes
18. Update logger to show colored log, defined in self.log 
19. Add to warpfield clear, is_deformed defined in tsdf, needs to be updated 
20. Add warp_point_with_nodes to Warpfield class as staticMethod, no need to be outside class
21. Write save/write functions after all modules are working.
22. Clean ARAP function in run_model, replace graph,warpfield, self.graph
23. Change graph_data["graph_nodes"] => graph_data["graph_nodes_at_source"]
24. During arap, If all members of the queue have updated neighbours. Means a seperate cluster has formed during adding nodes pahse something is wrong with the code. Hence break. Need to ada a check if no all members have no neigbours 	
# Tests: 
```
./tests.sh // Run Tests to see if working perfectly
```
1. Check normals are getting deformed perfectly. 
2. Check with and without gpu 
3. Check if images are getting skinned correctly  
4. Check if deformation happening correctly.
# Licesne 


# Acknowlegement
1. Neural Tracking
2. OcclusionFusion 