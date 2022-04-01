# Imports
import os
import logging # For logging info
import numpy as np 
import json # For loading graph config
from skimage import io
from timeit import default_timer as timer
import re

# Neural Tracking Modules
from model import dataset
from utils import utils, image_proc

from NeuralNRT._C import compute_mesh_from_depth_and_flow as compute_mesh_from_depth_and_flow_c
from NeuralNRT._C import erode_mesh as erode_mesh_c
from NeuralNRT._C import sample_nodes as sample_nodes_c
from NeuralNRT._C import compute_edges_geodesic as compute_edges_geodesic_c
from NeuralNRT._C import node_and_edge_clean_up as node_and_edge_clean_up_c
from NeuralNRT._C import compute_pixel_anchors_geodesic as compute_pixel_anchors_geodesic_c
from NeuralNRT._C import compute_clusters as compute_clusters_c
from NeuralNRT._C import update_pixel_anchors as update_pixel_anchors_c

class EDGraph: 
    def __init__(self,tsdf):

        # Log path 
        self.log = logging.getLogger(__name__)
        self.log.info(f"Creating Graph from:{tsdf.__class__.__name__}")


        self.opt = tsdf.fopt
        self.load_graph_config()        

        # create_graph_from_tsdf
        self.tsdf = tsdf
        self.create_graph_from_tsdf()


        # Save path
        self.savepath = os.path.join(self.tsdf.fopt.datadir,"results")
        os.makedirs(self.savepath,exist_ok=True)
        os.makedirs(os.path.join(self.savepath,"deformation"),exist_ok=True)
        os.makedirs(os.path.join(self.savepath,"updated_graph"),exist_ok=True)


    def load_graph_config(self):

        # Parameters for generating new graph
        if os.path.isfile(os.path.join(self.opt.datadir,'graph_config.json')):
            with open(os.path.join(self.opt.datadir,'graph_config.json')) as f:
                self.graph_generation_parameters = json.load(f) 

        else: # Defualt case
            self.graph_generation_parameters = {
                # Given a depth image, its mesh is constructed by joining using its adjecent pixels, 
                # based on the depth values, a perticular vertex could be far away from its adjacent pairs in for a particular face/triangle
                # `max_triangle_distance` controls this. For a sperical surfaces (such as a dolls, human body) this should be set high. But for clothes this can be small      
                'max_triangle_distance' : 0.05, 
                'erosion_num_iterations': 10,   # Will reduce outliers points (simliar to erosion in images)
                'erosion_min_neighbors' : 4,    # Will reduce outlier clusters
                'node_coverage'         : 0.05, # Sampling parameter which defines the number of nodes
                'min_neighbours'        : 4,    # Find minimum nunber of neigbours that must be present for ech node     
                'require_mask'          : True,
                }


    def create_graph_from_tsdf(self):
        
        assert hasattr(self,'tsdf'),  "TSDF not defined in graph. Add tsdf as attribute to EDGraph first." 
        vertices, faces, normals, colors = self.tsdf.get_mesh()  # Extract the new canonical pose using marching cubes

        self.create_graph_from_mesh(vertices,faces)


    # If RGBD Image run other steps
    def create_mesh_from_depth(self,im_data,depth_normalizer = 1000.0):
        """
            im_data: np.ndarray: 6xHxW RGB+PointImage 
        """

        #########################################################################
        # Load data.
        #########################################################################
        # Load intrinsics.
        intrinsics = np.loadtxt(intrinsics_path)

        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        # Load depth image.
        depth_image = im_data["im"][-1] 

        # Load mask image.
        mask_image = im_data[mask] 
        # if mask_image is None and self.graph_generation_parameters["require_mask"] 
        #########################################################################
        # Convert depth to mesh.
        #########################################################################
        width = depth_image.shape[1]
        height = depth_image.shape[0]

        # Invalidate depth values outside object mask.
        # We only define graph over dynamic object (inside the object mask).
        mask_image[mask_image > 0] = 1
        depth_image = depth_image * mask_image

        # Backproject depth images into 3D.
        point_image = depth_image.astype(np.float32)

        # Convert depth image into mesh, using pixelwise connectivity.
        # We also compute flow values, and invalidate any vertex with non-finite
        # flow values.
        vertices = np.zeros((0), dtype=np.float32)
        vertex_flows = np.zeros((0), dtype=np.float32)
        vertex_pixels = np.zeros((0), dtype=np.int32)
        faces = np.zeros((0), dtype=np.int32)

        compute_mesh_from_depth_c(
        point_image,
        self.graph_generation_parameters["max_triangle_distance"],
        vertices, vertex_pixels, faces)
        
        num_vertices = vertices.shape[0]
        num_faces = faces.shape[0]

        assert num_vertices > 0 and num_faces > 0

        return vertices,faces

    def erode_mesh(self,vertices,faces):
        """
            Erode the graph based on the graph strucuture. 
            Basically return vertices with greater than min_neighbours after x iterations 

            @params:
                vertices: Nx3 np.ndarray 
                faces: Mx3 np.ndarray(int)

            @returns: 
                non_eroded_vertices indices
        """    
        non_eroded_vertices = erode_mesh_c(
            vertices, faces,\
            self.graph_generation_parameters["erosion_num_iterations"],\
            self.graph_generation_parameters["erosion_min_neighbors"])
        return non_eroded_vertices

    def create_graph_from_mesh(self,vertices,faces):

        # Node sampling and edges computation
        USE_ONLY_VALID_VERTICES = True
        NODE_COVERAGE = self.graph_generation_parameters["node_coverage"]
        NUM_NEIGHBORS = 8
        ENFORCE_TOTAL_NUM_NEIGHBORS = False
        SAMPLE_RANDOM_SHUFFLE = False

        num_vertices = vertices.shape[0]
        num_faces = faces.shape[0]

        assert num_vertices > 0 and num_faces > 0

        # Erode mesh, to not sample unstable nodes on the mesh boundary.
        non_eroded_vertices = self.erode_mesh(vertices,faces)

        #########################################################################
        # Sample graph nodes.
        #########################################################################
        valid_vertices = non_eroded_vertices

        
        # Sample graph nodes.
        node_coords = np.zeros((0), dtype=np.float32)
        node_indices = np.zeros((0), dtype=np.int32)

        num_nodes = sample_nodes_c(
            vertices, valid_vertices,
            node_coords, node_indices, 
            NODE_COVERAGE, 
            USE_ONLY_VALID_VERTICES,
            SAMPLE_RANDOM_SHUFFLE
        )

        node_coords = node_coords[:num_nodes, :]
        node_indices = node_indices[:num_nodes, :]

        # Just for debugging
        # pcd_nodes = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(node_coords))
        # o3d.visualization.draw_geometries([pcd_nodes], mesh_show_back_face=True)

        #########################################################################
        # Compute graph edges.
        #########################################################################
        # Compute edges between nodes.
        graph_edges              = -np.ones((num_nodes, NUM_NEIGHBORS), dtype=np.int32)
        graph_edges_weights      =  np.zeros((num_nodes, NUM_NEIGHBORS), dtype=np.float32)
        graph_edges_distances    =  np.zeros((num_nodes, NUM_NEIGHBORS), dtype=np.float32)
        node_to_vertex_distances = -np.ones((num_nodes, num_vertices), dtype=np.float32)

        visible_vertices = np.ones_like(valid_vertices)

        compute_edges_geodesic_c(
            vertices, visible_vertices, faces, node_indices, 
            NUM_NEIGHBORS, NODE_COVERAGE, 
            graph_edges, graph_edges_weights, graph_edges_distances,
            node_to_vertex_distances,
            USE_ONLY_VALID_VERTICES,
            ENFORCE_TOTAL_NUM_NEIGHBORS
        )

        # Save current results 
        self.nodes           = node_coords
        # self.node_indices    = node_indices # Which vertex correspondes to graph node
        self.edges           = graph_edges 
        self.edges_weights   = graph_edges_weights 
        self.edges_distances = graph_edges_distances                  
        self.node_to_vertex_distances = node_to_vertex_distances    
        self.clusters = -np.ones((graph_edges.shape[0], 1), dtype=np.int32) # Will be calculated later 



        # Cluster nodes in graph
        NEIGHBORHOOD_DEPTH = 2
        MIN_CLUSTER_SIZE = 3

        MIN_NUM_NEIGHBORS = 2
        REMOVE_NODES_WITH_NOT_ENOUGH_NEIGHBORS = True

        # Remove nodes 
        valid_nodes_mask = np.ones((num_nodes, 1), dtype=bool)
        node_id_black_list = []

        if REMOVE_NODES_WITH_NOT_ENOUGH_NEIGHBORS:
            # Mark nodes with not enough neighbors
            node_and_edge_clean_up_c(graph_edges, valid_nodes_mask)
        else:
            self.log.info("You're allowing nodes with not enough neighbors!")

        # Remove black listed nodes
        reduced_graph_dict = self.get_reduced_graph(valid_nodes_mask)

        # Remove invalid graph
        self.nodes           = reduced_graph_dict["nodes"]
        # self.node_indices    = reduced_graph_dict["node_indices"] # Which vertex correspondes to graph node
        self.edges           = reduced_graph_dict["graph_edges"] 
        self.edges_weights   = reduced_graph_dict["graph_edges_weights"] 
        self.edges_distances = reduced_graph_dict["graph_edges_distances"]                  
        self.node_to_vertex_distances = reduced_graph_dict["node_to_vertex_distances"]
        self.clusters  = reduced_graph_dict["graph_clusters"] 

        #########################################################################
        # Compute clusters.
        #########################################################################
        self.compute_clusters()

    def compute_clusters(self):
        """
            Based on graph traversal find connected components and put them in seperate clusters. 
        """    
        clusters_size_list = compute_clusters_c(self.edges, self.clusters)

        for i, cluster_size in enumerate(clusters_size_list):
            if cluster_size <= 2:
                self.log.error(f"Cluster is too small {clusters_size_list}")
                self.log.error(f"It only has nodes:{np.where(self.clusters == i)[0]}")

    def get_reduced_graph(self,valid_nodes_mask):
            

        # Get the list of invalid nodes
        node_id_black_list = np.where(valid_nodes_mask == False)[0].tolist()

        # Get only graph corresponding info
        node_coords           = self.nodes[valid_nodes_mask.squeeze()]
        graph_edges           = self.edges[valid_nodes_mask.squeeze()] 
        graph_edges_weights   = self.edges_weights[valid_nodes_mask.squeeze()] 
        graph_edges_distances = self.edges_distances[valid_nodes_mask.squeeze()] 
        graph_clusters        = self.clusters[valid_nodes_mask.squeeze()]
        node_to_vertex_distances = self.node_to_vertex_distances[valid_nodes_mask.squeeze()]
        #########################################################################
        # Graph checks.
        #########################################################################
        num_nodes = node_coords.shape[0]

        # Check that we have enough nodes
        if (num_nodes == 0):
            print("No nodes! Exiting ...")
            exit()

        self.log.info(f"Node filtering: initial num nodes: {num_nodes} | invalid nodes: {len(node_id_black_list)}:{node_id_black_list}")

        # Update node ids only if we actually removed nodes
        if len(node_id_black_list) > 0:
            # 1. Mapping old indices to new indices
            count = 0
            node_id_mapping = {}
            for i, is_node_valid in enumerate(valid_nodes_mask):
                if not is_node_valid:
                    node_id_mapping[i] = -1
                else:
                    node_id_mapping[i] = count
                    count += 1

            # 2. Update graph_edges using the id mapping
            for node_id, graph_edge in enumerate(graph_edges):
                # compute mask of valid neighbors
                valid_neighboring_nodes = np.invert(np.isin(graph_edge, node_id_black_list))

                # make a copy of the current neighbors' ids
                graph_edge_copy           = np.copy(graph_edge)
                graph_edge_weights_copy   = np.copy(graph_edges_weights[node_id])
                graph_edge_distances_copy = np.copy(graph_edges_distances[node_id])

                # set the neighbors' ids to -1
                graph_edges[node_id]           = -np.ones_like(graph_edge_copy)
                graph_edges_weights[node_id]   =  np.zeros_like(graph_edge_weights_copy)
                graph_edges_distances[node_id] =  np.zeros_like(graph_edge_distances_copy)

                count_valid_neighbors = 0
                for neighbor_idx, is_valid_neighbor in enumerate(valid_neighboring_nodes):
                    if is_valid_neighbor:
                        # current neighbor id
                        current_neighbor_id = graph_edge_copy[neighbor_idx]    

                        # get mapped neighbor id       
                        if current_neighbor_id == -1: mapped_neighbor_id = -1
                        else:                         mapped_neighbor_id = node_id_mapping[current_neighbor_id]    

                        graph_edges[node_id, count_valid_neighbors]           = mapped_neighbor_id
                        graph_edges_weights[node_id, count_valid_neighbors]   = graph_edge_weights_copy[neighbor_idx]
                        graph_edges_distances[node_id, count_valid_neighbors] = graph_edge_distances_copy[neighbor_idx]

                        count_valid_neighbors += 1

                # normalize edges' weights
                sum_weights = np.sum(graph_edges_weights[node_id])
                if sum_weights > 0:
                    graph_edges_weights[node_id] /= sum_weights
                else:
                    print("Hmmmmm", graph_edges_weights[node_id])
                    raise Exception("Not good")

        # TODO: Check if some cluster is not present in reduced nodes. Then exit or raise error  

        reduced_graph_dict = {} # Store graph dict 

        reduced_graph_dict["nodes"]                 = node_coords
        reduced_graph_dict["graph_edges"]           = graph_edges 
        reduced_graph_dict["graph_edges_weights"]   = graph_edges_weights 
        reduced_graph_dict["graph_edges_distances"] = graph_edges_distances                  
        reduced_graph_dict["graph_clusters"]        = graph_clusters                  

        reduced_graph_dict["num_nodes"]             =  np.array(num_nodes, dtype=np.int64) # Number of nodes in this graph
        reduced_graph_dict["node_to_vertex_distances"] = node_to_vertex_distances # NxV geodesic distance between vertices of mesh and graph nodes

        return reduced_graph_dict

    def update_deformation_parameters(self):       
        pass

    def deform_tsdf(self):
        pass 

    def deform(self,points):
        pass   

    def update_nodes(self,new_verts):
        """
            Given vertices which are have outside the coverage of the graph. 
            Sample nodes and create edges betweem them and old graph

            @params: 
                new_verts: (Nx3) np.ndarry: New vertices from which nodes are sampled

            @returns: 
                update: bool: Whether new nodes were added to graph     
        """

        # Sample nodes such that they are node_coverage apart from one another 
        new_node_list = []
        for i, x in enumerate(new_verts):
            if len(new_node_list) == 0:
                new_node_list.append(i)
                continue

            # If node already covered
            min_nodes_dist = np.min(np.linalg.norm(new_verts[new_node_list] - x, axis=1))
            if min_nodes_dist < graph_data["node_coverage"]: # Not sure if correct
                continue
            else:
                new_node_list.append(i)
        print("New Node List:", new_node_list,len(new_node_list))
        if len(new_node_list) == 0: # No new nodes were added
            return False

        # Find their co-ordinates, initialize for later use 
        new_node_coords = new_verts[new_node_list]
        print("New Node Coords:", new_node_coords)

        node_coverage = self.graph_generation_parameters["node_coverage"]
        min_neighbours = self.graph_generation_parameters["min_neighbours"]


        while True:  # Remove new_nodes with <= 4 neighbors (selected by experimentation)

            new_nodes = np.concatenate([self.nodes, new_node_coords], axis=0)

            # Find Distance matrix for new nodes and graph 
            new_node_distance_matrix = self.calculate_distance_matrix(new_node_coords, new_nodes)
            nn_nodes = np.argsort(new_node_distance_matrix, axis=1)[:, 1:9]  # First eight neighbours (0th index represents same point)

            nn_dists = np.array([new_node_distance_matrix[i, nn_nodes[i, :]] for i in range(nn_nodes.shape[0]) ])

            nn_nodes[nn_dists > 2*graph_data["node_coverage"]] = -1
            removable_nodes_mask = np.where(np.sum(nn_nodes != -1, axis=1) <= min_neighbours)[0]
            
            # If no nodes need to be removed, exit loop
            if len(removable_nodes_mask) == 0:
                break

            new_node_list = np.delete(new_node_list, removable_nodes_mask)
            print("Removing nodes as they don't have enough neighbors:",new_node_list)

        if len(new_node_list) == 0:
            return graph_data

        print("Adding Nodes:", new_node_list)

        if plot_update:
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=1280, height=960)

            old_graph = viz_utils.create_open3d_graph(
                viz_utils.transform_pointcloud_to_opengl_coords(graph_data["graph_nodes"]),
                graph_data["graph_edges"])
            new_regions = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
                viz_utils.transform_pointcloud_to_opengl_coords(new_verts)))
            new_regions.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1,0,0]]),(new_verts.shape[0],1)))

            new_node_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
                viz_utils.transform_pointcloud_to_opengl_coords(new_verts[new_node_list])))
            new_node_pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0,0,1]]),(len(new_node_list),1)))

            vis.add_geometry(old_graph[0])
            vis.add_geometry(old_graph[1])
            vis.add_geometry(new_regions)
            vis.add_geometry(new_node_pc)

            vis.run()
            vis.close()


        # Update graph data
        graph_data["graph_nodes"] = np.concatenate([graph_data["graph_nodes"], new_verts[new_node_list]], axis=0)
        graph_data["graph_edges"] = np.concatenate([graph_data["graph_edges"], nn_nodes], axis=0)

        nn_weights = nn_dists.copy()
        nn_weights[nn_weights == -1] = np.inf # Updating dists to calculate weights
        nn_weights = np.exp(-0.5 * (nn_dists / graph_data["node_coverage"]) ** 2)
        nn_weights /= np.sum(nn_weights, axis=1, keepdims=True)
        graph_data["graph_edges_weights"] = np.concatenate([graph_data["graph_edges_weights"], nn_weights], axis=0)

        graph_data["num_nodes"] = np.array(graph_data["graph_nodes"].shape[0], dtype=np.int64) 

        graph_data["graph_clusters"] = -np.ones((graph_data["graph_edges"].shape[0], 1), dtype=np.int32)
        clusters_size_list = compute_clusters_c(graph_data["graph_edges"], graph_data["graph_clusters"])

        for i, cluster_size in enumerate(clusters_size_list):
            if cluster_size <= 2:
                print("Cluster is too small {}".format(clusters_size_list))
                nodes_to_remove = np.where(graph_data["graph_clusters"] == i)[0]
                print("It only has nodes:", nodes_to_remove, " removing them.")
                
                graph_data = remove_nodes(graph_data,nodes_to_remove)

        return graph_data

    def save(self):
        #########################################################################
        # Save data.
        #########################################################################
        os.makedirs(os.path.join(self.tsdf.fopt.datadir,str(self.tsdf.frame_index)),exist_ok=True)

        dst_graph_nodes_dir = os.path.join(self.tsdf.fopt.datadir,str(self.tsdf.frame_index), "graph_nodes")
        if not os.path.exists(dst_graph_nodes_dir): os.makedirs(dst_graph_nodes_dir)

        dst_graph_edges_dir = os.path.join(self.tsdf.fopt.datadir,str(self.tsdf.frame_index), "graph_edges")
        if not os.path.exists(dst_graph_edges_dir): os.makedirs(dst_graph_edges_dir)

        dst_graph_edges_weights_dir = os.path.join(self.tsdf.fopt.datadir,str(self.tsdf.frame_index), "graph_edges_weights")
        if not os.path.exists(dst_graph_edges_weights_dir): os.makedirs(dst_graph_edges_weights_dir)

        dst_node_deformations_dir = os.path.join(self.tsdf.fopt.datadir,str(self.tsdf.frame_index), "graph_node_deformations")
        if not os.path.exists(dst_node_deformations_dir): os.makedirs(dst_node_deformations_dir)

        dst_graph_clusters_dir = os.path.join(self.tsdf.fopt.datadir,str(self.tsdf.frame_index), "graph_clusters")
        if not os.path.exists(dst_graph_clusters_dir): os.makedirs(dst_graph_clusters_dir)

        dst_pixel_anchors_dir = os.path.join(self.tsdf.fopt.datadir,str(self.tsdf.frame_index), "pixel_anchors")
        if not os.path.exists(dst_pixel_anchors_dir): os.makedirs(dst_pixel_anchors_dir)

        dst_pixel_weights_dir = os.path.join(self.tsdf.fopt.datadir,str(self.tsdf.frame_index), "pixel_weights")
        if not os.path.exists(dst_pixel_weights_dir): os.makedirs(dst_pixel_weights_dir)

        output_graph_nodes_path           = os.path.join(dst_graph_nodes_dir, pair_name           + "_{}_{:.2f}.bin".format("geodesic", NODE_COVERAGE))
        output_graph_edges_path           = os.path.join(dst_graph_edges_dir, pair_name           + "_{}_{:.2f}.bin".format("geodesic", NODE_COVERAGE))
        output_graph_edges_weights_path   = os.path.join(dst_graph_edges_weights_dir, pair_name   + "_{}_{:.2f}.bin".format("geodesic", NODE_COVERAGE))
        output_graph_clusters_path        = os.path.join(dst_graph_clusters_dir, pair_name        + "_{}_{:.2f}.bin".format("geodesic", NODE_COVERAGE))
        output_pixel_anchors_path         = os.path.join(dst_pixel_anchors_dir, pair_name         + "_{}_{:.2f}.bin".format("geodesic", NODE_COVERAGE))
        output_pixel_weights_path         = os.path.join(dst_pixel_weights_dir, pair_name         + "_{}_{:.2f}.bin".format("geodesic", NODE_COVERAGE))


        utils.save_graph_nodes(output_graph_nodes_path, node_coords)
        utils.save_graph_edges(output_graph_edges_path, graph_edges)
        utils.save_graph_edges_weights(output_graph_edges_weights_path, graph_edges_weights)
        utils.save_graph_clusters(output_graph_clusters_path, graph_clusters)
        utils.save_int_image(output_pixel_anchors_path, pixel_anchors)
        utils.save_float_image(output_pixel_weights_path, pixel_weights)


    def load_graph_savepaths(self):    
        

        # Load all types of data avaible
        graph_dict = {}
        if os.path.isdir(os.path.join(self.opt.datadir,"graph_nodes")):
            for file in os.listdir(os.path.join(self.opt.datadir,"graph_nodes")):
            
                file_data = file[:-4].split('_')
                if len(file_data) == 4: # Using our setting frame_<frame_index>_geodesic_<node_coverage>.bin
                    frame_index = int(file_data[1])
                    node_coverage = float(file_data[-1])
                elif len(file_data) == 6: # Using name setting used by authors <random_str>_<Obj-Name>_<Source-Frame-Index>_<Target-Frame-Index>_geodesic_<Node-Coverage>.bin
                    frame_index = int(file_data[2])
                    node_coverage = float(file_data[-1])
                else:
                    raise NotImplementedError(f"Unable to understand file:{file} to get graph data")

                graph_dicts[frame_index] = {}
                graph_dicts[frame_index]["graph_nodes_path"]             = os.path.join(self.opt.datadir, "graph_nodes",        file)
                graph_dicts[frame_index]["graph_edges_path"]             = os.path.join(self.opt.datadir, "graph_edges",        file)
                graph_dicts[frame_index]["graph_edges_weights_path"]     = os.path.join(self.opt.datadir, "graph_edges_weights",file)
                graph_dicts[frame_index]["graph_clusters_path"]          = os.path.join(self.opt.datadir, "graph_clusters",     file)
                graph_dicts[frame_index]["pixel_anchors_path"]           = os.path.join(self.opt.datadir, "pixel_anchors",      file)
                graph_dicts[frame_index]["pixel_weights_path"]           = os.path.join(self.opt.datadir, "pixel_weights",      file)
                graph_dicts[frame_index]["node_coverage"]                = node_coverage

        self.graph_save_path = graph_dicts

    def load_graph(self,frame_index=0):               
        self.node           =          utils.load_graph_nodes(self.graph_save_path[frame_index]["graph_nodes_path"])
        self.edges          =          utils.load_graph_edges(self.graph_save_path[frame_index]["graph_edges_path"])
        self.edges_weights  =  utils.load_graph_edges_weights(self.graph_save_path[frame_index]["graph_edges_weights_path"])
        self.clusters       =       utils.load_graph_clusters(self.graph_save_path[frame_index]["graph_clusters_path"])

    def get_graph_path(self,index):
        """
            This function returns the paths to the graph generated for a particular frame, and geodesic distance (required for sampling nodes, estimating edge weights etc.)
        """

        if index not in self.graph_dicts:
            self.graph_dicts[index] = create_graph_data_using_depth(\
                os.path.join(self.seq_dir,"depth",self.images_path[index].replace('jpg','png')),\
                max_triangle_distance=self.graph_generation_parameters['max_triangle_distance'],\
                erosion_num_iterations=self.graph_generation_parameters['erosion_num_iterations'],\
                erosion_min_neighbors=self.graph_generation_parameters['erosion_min_neighbors'],\
                node_coverage=self.graph_generation_parameters['node_coverage'],\
                require_mask=self.graph_generation_parameters['require_mask']
                )

        return self.graph_dicts[index]

    def get_graph(self,index,cropper):
        # Graph
        graph_path_dict = self.get_graph_path(index)

        graph_nodes, graph_edges, graph_edges_weights, _, graph_clusters, pixel_anchors, pixel_weights = dataset.DeformDataset.load_graph_data(
            graph_path_dict["graph_nodes_path"], graph_path_dict["graph_edges_path"], graph_path_dict["graph_edges_weights_path"], None, 
            graph_path_dict["graph_clusters_path"], graph_path_dict["pixel_anchors_path"], graph_path_dict["pixel_weights_path"], cropper
        )

        num_nodes = np.array(graph_nodes.shape[0], dtype=np.int64)  

        graph_dict = {}
        graph_dict["graph_nodes"]               = graph_nodes
        graph_dict["graph_edges"]               = graph_edges
        graph_dict["graph_edges_weights"]       = graph_edges_weights
        graph_dict["graph_clusters"]            = graph_clusters
        graph_dict["pixel_weights"]             = pixel_weights
        graph_dict["pixel_anchors"]             = pixel_anchors
        graph_dict["num_nodes"]                 = num_nodes
        graph_dict["node_coverage"]             = graph_path_dict["node_coverage"]
        graph_dict["graph_neighbours"]          = min(len(graph_nodes),4) 
        return graph_dict


    def load_dict(self,path):
        data = loadmat(path)
        model_data = {}
        for k in data: 
            if '__' in k:
                continue
            else:
                model_data[k] = np.array(data[k])
                if model_data[k].shape == (1,1):
                    model_data[k] = np.array(model_data[k][0,0])
        return model_data






if __name__ == "__main__":
    #########################################################################
    # Options
    #########################################################################
    # Depth-to-mesh conversion

    MAX_TRIANGLE_DISTANCE = 0.05

    # Erosion of vertices in the boundaries
    EROSION_NUM_ITERATIONS = 10
    EROSION_MIN_NEIGHBORS = 4

    # Node sampling and edges computation
    NODE_COVERAGE = 0.05 # in meters
    USE_ONLY_VALID_VERTICES = True
    NUM_NEIGHBORS = 8
    ENFORCE_TOTAL_NUM_NEIGHBORS = False
    SAMPLE_RANDOM_SHUFFLE = False

    # Pixel anchors
    NEIGHBORHOOD_DEPTH = 2

    MIN_CLUSTER_SIZE = 3
    MIN_NUM_NEIGHBORS = 2 

    # Node clean-up
    REMOVE_NODES_WITH_NOT_ENOUGH_NEIGHBORS = True

    #########################################################################
    # Paths.
    #########################################################################
    seq_dir = os.path.join("example_data" , "train", "seq258")

    depth_image_path = os.path.join(seq_dir, "depth", "000000.png")
    mask_image_path = os.path.join(seq_dir, "mask", "000000_shirt.png")
    scene_flow_path = os.path.join(seq_dir, "scene_flow", "shirt_000000_000110.sflow")
    intrinsics_path = os.path.join(seq_dir, "intrinsics.txt")

    pair_name = "generated_shirt_000000_000110"







    
