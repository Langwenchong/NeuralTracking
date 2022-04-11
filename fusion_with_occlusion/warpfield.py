# Define a class to show how to deform the data using graph, skinning weights and points
# This class essentially connects ED graph and TSDF volume 
import os
import open3d as o3d
import logging
import numpy as np 
from numba import cuda as ncuda
from numba import njit, prange

# CPU module imports
from pykdtree.kdtree import KDTree as KDTreeCPU

# Neural Tracking modules 
from utils import utils, image_proc
from NeuralNRT._C import compute_mesh_from_depth as compute_mesh_from_depth_c

class WarpField:
    def __init__(self, graph, tsdf, visualizer, kdtree_leaf_size=16):
        """
            Warp field to deform unerlying volume/mesh/image 
            Args: 
                graph: EDGraph class containing node details created using NNRT module 
                data_structure: TSDFVolume/Open3DMesh/DepthImage: Class containing the estimated tsdf volume
                fopt: Hyperparameter for fusion
                kdtree_leaf_size: Number of nodes in leaf in kdtree (larger leaf size means smaller tree depth)
        
            This module mainly does:
                1. Skinning the data structure w.r.t graph nodes
                2. Deform the data structure from source frame to target frame using the esitmated transformations by neural tracking  
        """

        # Initialize sub-modules
        self.graph = graph
        self.tsdf  = tsdf 
        self.vis   = visualizer
        self.frame_id = self.tsdf.fopt.source_frame

        self.log = logging.getLogger(__name__)
        self.log.info(f"Adding Warpfield for:{tsdf.__class__.__name__} at frame_id:{self.frame_id}")

        # Initialize KDTree for finding anchors/skinning
        self.kdtree_leaf_size = kdtree_leaf_size

        # Set defualt deformation values, see self.update() for more details 
        N = self.graph.nodes.shape[0]
        self.source_frame_kdtree = KDTreeCPU(self.graph.nodes, leafsize=self.kdtree_leaf_size) # Update the main kdtree 
        self.rotations = np.tile(np.eye(3,dtype=np.float32).reshape((1,3,3)),(N,1,1))
        self.translations = np.zeros((N,3),dtype=np.float32)
        self.deformed_graph_nodes = self.graph.nodes.copy() # Stores the deformed node positions. Note these are w.r.t source frame  

        # Boolean variable to check whether warpfield is updating or not after addinf new nodes
        self.updating_warpfield = False 

        # How many graph nodes are utilized to deform voxels (Default=4)
        self.graph_neighbours = min(N,4)
        self.node_coverage = self.graph.graph_generation_parameters["node_coverage"]
        self.gpu = self.tsdf.fopt.gpu 



        # Save path for data
        self.savepath = os.path.join(self.tsdf.fopt.datadir,"results")
        os.makedirs(self.savepath,exist_ok=True)
        os.makedirs(os.path.join(self.savepath,"deformation"),exist_ok=True)


    ##########################################
    #     Modules to skin data structures    #
    ##########################################
    def skin(self,points,nodes=None):
        """
            Calculate skinning weights and their anchors
            
            @params:
                points: (Nx3) np.ndarray: points for which skinning weights are getting calculated
                nodes: (Mx3) graph nodes for which skinning is getting calculated

            @results:
                amchors: (Nx4) np.ndarray (inrt), closest neighbours of each point, -1 if outside node coverage  
                weights: (Nx4) np.ndarray , skinning weights for closest neighbours of each point  


            Use KDTree GPU for faster computation   
        """
        points = points.astype(np.float32)
        
        if nodes is None: # If no node information is passed, it means use graph nodes at source grame
            kdtree = self.source_frame_kdtree
        else:    
            kdtree = KDTreeCPU(nodes, leafsize=self.kdtree_leaf_size)
        dist, anchors = kdtree.query(points, k=self.graph_neighbours)

        # Removes miss alignments but reduces new surface from being added
        dist[dist > 2*self.node_coverage] = np.inf

        anchors[dist == np.inf] = -1
        weights = np.exp(-dist**2 / (2.0 * (self.node_coverage**2)))  # Without normalization
        
        if self.graph_neighbours == 1:
            anchors = anchors.reshape(-1, 1)
            weights = weights.reshape(-1, 1)
        
        
        valid_pts = np.sum(weights,axis=-1) > 1e-6 # Points with no weights should not be deformed
        weights /= (np.sum(weights,axis=-1,keepdims=True)+1e-6) # Normalized weights




        anchors = anchors.astype(np.int32)  
        weights = weights.astype(np.float32)


        anchors[~valid_pts] = -1
        weights[~valid_pts] = 0


        return anchors,weights,valid_pts

    def skin_tsdf(self):    

        if not hasattr(self,"world_anchors") or not hasattr(self,"world_weights") or self.updating_warpfield:  
            points = self.tsdf.world_pts
            anchors,weights,valid_pts = self.skin(points)
            self.world_anchors = anchors  
            self.world_weights = weights
            self.world_valid_pts = valid_pts        

        # If not updating the warpfield return previous used values
        return self.world_anchors,self.world_weights,self.world_valid_pts

    def skin_image(self,nodes,image_data):
        """
            Calculate skinning weights for Neural Tracking calculatation      
            Note:- here the skinning is computed w.r.t defromed nodes and not original nodes
        """

        self.source_im = image_data["im"]   

        # Get vertices and their indices 
        # Invalidate depth values outside object mask.
        # We only define graph over dynamic object (inside the object mask).
        
        if "mask" in image_data and image_data["mask"] is not None: 
            mask_image = image_data["mask"].copy()
            point_image = image_data["im"][3:].copy()
            mask_image[mask_image > 0] = 1
            point_image = point_image * mask_image[None,...]


        vertices = np.zeros((0), dtype=np.float32)
        vertex_pixels = np.zeros((0), dtype=np.int32)
        faces = np.zeros((0), dtype=np.int32)

        compute_mesh_from_depth_c(
            point_image,
            self.graph.graph_generation_parameters["max_triangle_distance"],
            vertices, vertex_pixels, faces)


        #########################################################################
        # Compute pixel anchors.
        #########################################################################
        anchors,weights,valid_verts = self.skin(vertices,nodes=nodes)

        im_h,im_w = point_image.shape[1:3]
        pixel_anchors = np.zeros((im_h,im_w,self.graph_neighbours), dtype=np.int32)
        pixel_weights = np.zeros((im_h,im_w,self.graph_neighbours), dtype=np.float32)

        pix_y = vertex_pixels[:,1]
        pix_x = vertex_pixels[:,0]

        pixel_anchors[pix_y,pix_x,:] = anchors
        pixel_weights[pix_y,pix_x,:] = weights

        # self.log.info(f"Skinned Source Image, valid pixels:{np.sum(np.all(pixel_anchors != -1, axis=2))}")
        self.log.info(f"Skinned Source Image, valid pixels:{np.sum(np.all(pixel_anchors != -1, axis=2))}")

        # Just for debugging.
        # pixel_anchors_image = np.sum(pixel_anchors, axis=2)
        # pixel_anchors_mask = np.copy(pixel_anchors_image).astype(np.uint8)
        # pixel_anchors_mask[...] = 1
        # pixel_anchors_mask[pixel_anchors_image == -4] = 0
        # utils.save_grayscale_image("output/pixel_anchors_mask.jpeg", pixel_anchors_mask) 
        skin_data = {"pixel_anchors":pixel_anchors,"pixel_weights":pixel_weights}
        return skin_data

    # Load already saved data
    def load_skin_image(self):
        self.pixel_anchors = utils.load_int_image(graph_dicts[self.frame_id]["pixel_anchors_path"])
        self.pixel_weights = utils.load_float_image(graph_dicts[self.frame_id]["pixel_weights_path"])    


    ##############################################
    #     Functions to deform data structures    #
    ##############################################
    @staticmethod
    @njit(parallel=True)
    def deform_lbs(node_rotations, node_translations, world_pts, world_anchors, world_weights,valid_pts):
        
        deformed_world_pts = np.empty_like(world_pts, dtype=world_pts.dtype)
        for i in prange(world_pts.shape[0]):

            if valid_pts[i]:
                world_weights_normalized = world_weights[i]
                deformed_world_pts[i, :] = 0
                for k in range(world_weights.shape[1]):
                    if world_weights_normalized[k] == 0:
                        continue
                    # deformed_points_k = np.dot(node_rotations[world_anchors[i, k]],
                    #                             world_pts[i] - graph_nodes[world_anchors[i, k]]) \
                    #                     + graph_nodes[world_anchors[i, k]]\
                    #                     + node_translations[world_anchors[i, k]]
                    deformed_points_k = np.dot(node_rotations[world_anchors[i,k]],world_pts[i]) + node_translations[world_anchors[i,k]] # Translation already 
                    deformed_world_pts[i] += world_weights_normalized[k] * deformed_points_k  # (num_pixels, 3, 1)

            else: # No deformation 
                deformed_world_pts[i] = world_pts
                   
        return deformed_world_pts


    @staticmethod   
    @ncuda.jit()
    def deform_lbs_cuda(deformed_pos, volume, anchors, weights, rotations, translations,valid_pts):
        x, y = ncuda.grid(2)
        X_SIZE, Y_SIZE, Z_SIZE = volume.shape[:3]
        if x < 0 or x > X_SIZE-1 or y < 0 or y > Y_SIZE-1:
            return

        for z in range(Z_SIZE):
            voxel_x = volume[x,y,z,0] 
            voxel_y = volume[x,y,z,1]
            voxel_z = volume[x,y,z,2]


            if valid_pts[x,y,z]:
                deformed_pos[x,y,z,0] = 0.0
                deformed_pos[x,y,z,1] = 0.0
                deformed_pos[x,y,z,2] = 0.0
                for i in range(anchors.shape[3]):
                    if weights[x,y,z,i] == 0:
                        continue
                    # new_x, new_y, new_z = Warpfield.warp_point_with_nodes(rotations[anchors[x, y, z, i]],
                    new_x, new_y, new_z = warp_point_with_nodes(rotations[anchors[x, y, z, i]],
                                                                translations[anchors[x, y, z, i]],
                                                                voxel_x, voxel_y, voxel_z)
                    deformed_pos[x,y,z,0] += new_x * weights[x, y, z, i]
                    deformed_pos[x,y,z,1] += new_y * weights[x, y, z, i]
                    deformed_pos[x,y,z,2] += new_z * weights[x, y, z, i]

            else: # No deformation 
                deformed_pos[x,y,z,0] = voxel_x
                deformed_pos[x,y,z,1] = voxel_y
                deformed_pos[x,y,z,2] = voxel_z



    def deform(self,points,anchors,weights,reshape_gpu_vol,valid_pts):

        rotations = self.rotations.astype(np.float32)
        translations = self.translations.astype(np.float32)

        ### Write gpu code for deform world points lbs, currently only works on volume points 
        if self.gpu:
            points = np.ascontiguousarray(points.reshape(reshape_gpu_vol+[3]))
            anchors = anchors.reshape(reshape_gpu_vol+[self.graph_neighbours])
            weights = weights.reshape(reshape_gpu_vol+[self.graph_neighbours])
            valid_pts = valid_pts.reshape(reshape_gpu_vol)
            deform_pts = np.ascontiguousarray(np.zeros_like(points))
            
            threadsperblock = (16, 16)
            blockspergrid = (np.ceil(points.shape[0] / threadsperblock[0]).astype('uint'), np.ceil(points.shape[1] / threadsperblock[1]).astype('uint'))
            self.deform_lbs_cuda[blockspergrid, (threadsperblock)](deform_pts, points, 
            anchors, weights, 
            rotations, translations,
            valid_pts)
            ncuda.synchronize() 
            deform_pts = deform_pts.reshape(-1,3)
        
        else:
            points = np.ascontiguousarray(points)
            
            deform_pts = self.deform_lbs(rotations, translations,
                            points,
                            anchors, weights,
                            valid_pts)

        return deform_pts

    # Normals are transformed similarly, according to the weighted sum
    # of each normal transformed by the inverse transpose of the node
    # transformations, and then renormalized. 
    # See Embedded Deformation for Shape Manipulation: https://people.inf.ethz.ch/~sumnerb/research/embdef/Sumner2007EDF.pdf 
    # Currently setting translation as 0, TODO: Actually need to update deform function which may not take transformations 
    def deform_normals(self,normals,anchors,weights,reshape_gpu_vol,valid_pts):

        rotations = self.rotations.astype(np.float32)
        translations = np.zeros_like(self.translations,dtype=np.float32)

        ### Write gpu code for deform world points lbs, currently only works on volume points 
        if self.gpu:
            normals = np.ascontiguousarray(normals.reshape(reshape_gpu_vol+[3]))
            anchors = anchors.reshape(reshape_gpu_vol+[self.graph_neighbours])
            weights = weights.reshape(reshape_gpu_vol+[self.graph_neighbours])
            valid_pts = valid_pts.reshape(reshape_gpu_vol)

            deform_pts = np.ascontiguousarray(np.zeros_like(normals))
            
            threadsperblock = (16, 16)
            blockspergrid = (np.ceil(normals.shape[0] / threadsperblock[0]).astype('uint'), np.ceil(normals.shape[1] / threadsperblock[1]).astype('uint'))
            self.deform_lbs_cuda[blockspergrid, (threadsperblock)](deform_pts, normals, 
            anchors, weights, 
            rotations, translations,
            valid_pts)
            ncuda.synchronize() 
            
            deform_pts = deform_pts.reshape(-1,3)
        
        else:
            deform_pts = self.deform_lbs(rotations, translations,
                            normals,
                            anchors, weights,
                            valid_pts)

        # Normalize 
        deform_normals = deform_pts/np.linalg.norm(deform_pts,axis=1,keepdims=True)

        return deform_normals    

    def deform_mesh(self,vertices,normals):
        """
            Deform Canonical voxel grid
        """    

        # Get skinning weights 
        # Todo, this can be optimised such that skin is only computed for new vertices 
        skin_anchors,skin_weights,valid_verts = self.skin(vertices)

        reshape_gpu_vol = [vertices.shape[0],1,1]        
        deformed_vertices = self.deform(vertices,skin_anchors,skin_weights,reshape_gpu_vol,valid_verts)    
        deformed_normals = self.deform_normals(normals,skin_anchors,skin_weights,reshape_gpu_vol,valid_verts)    

        return deformed_vertices,deformed_normals        
                    
    def deform_tsdf(self):
        """
            Deform TSDF voxel grid
        """    

        assert self.frame_id == self.tsdf.frame_id, f"Warpfield maps to:{self.frame_id}th frame but TSDF maps to:{self.tsdf.frame_id}th frame"

        world_anchors,world_weights,world_valid_pts = self.skin_tsdf()
        reshape_gpu_vol = list(self.tsdf._vol_dim)        
        deformed_tsdf = self.deform(self.tsdf.world_pts,world_anchors,world_weights,reshape_gpu_vol,world_valid_pts)    

        return deformed_tsdf,world_valid_pts



    ##############################################
    #     Functions to update transformation     #
    ##############################################

    # Update transformations after estimation. Thus obtain direct transformation from source frame to current target frame directly 
    def update_transformations(self,nnrt_data):
        """
            Update transformation parameters based on Neural Tracking 
            nnrt_data: dict : estimated transformation parameters
        """
        
        # Check if already updated. Then make no changes.


        # Assert warpfield and tsdf map to same frame (source frame). Aftet updating warpfield will map to target frame   
        assert self.frame_id == self.tsdf.frame_id,f"Warpfield maps to:{self.frame_id}th frame but TSDF maps to:{self.tsdf.frame_id}th frame"

        assert self.rotations.shape[0] == self.deformed_graph_nodes.shape[0],f"Nodes mismatch between transformation data:{self.rotations.shape} and deformed node:{self.deformed_graph_nodes.shape} data"
        # Update deformed_nodes
        N = self.deformed_graph_nodes.shape[0]

        # Update transformation paramaeters to get transformation from source frame to target frame of all nodes
        # Make sure to update translations first, so we can use previous rotations
        self.translations = np.array([ nnrt_data["node_rotations"][i]@self.translations[i]for i in range(N)])\
                            - np.array([ nnrt_data["node_rotations"][i]@self.deformed_graph_nodes[i] for i in range(N)])\
                            + self.deformed_graph_nodes + nnrt_data["node_translations"]

        # Since                     
        self.rotations = np.array([ nnrt_data["node_rotations"][i]@self.rotations[i]for i in range(N)])

        # Finally update graph nodes
        self.deformed_graph_nodes = nnrt_data["deformed_graph_nodes"]                            
        self.frame_id = nnrt_data["target_frame_id"]

    def get_transformation_wrt_graph_node(self):
        """
            Transformation parameters are saved w.r.t origin. 
            This function return transformations w.r.t each graph node 
            or Given R,t => T(4x4) = [[R,-Rg + g + T],[0,0,0,1]] 
        """    
        N = self.rotations.shape[0]

        rotations = self.rotations
        translations = self.translations - self.graph.nodes[:N]\
                    + np.array([ rotations[i]@self.graph.nodes[i]for i in range(N)])

        return rotations,translations

    def get_transformation_wrt_origin(self,rotations,translations):
        """
            Transformation parameters are saved w.r.t origin 
            or Given R,t => T(4x4) = [[R,T],[0,0,0,1]]
        """    
        N = rotations.shape[0]

        rotations = rotations
        translations = translations + self.graph.nodes[:N]\
                    - np.array([ rotations[i]@self.graph.nodes[i]for i in range(N)])

        return rotations,translations                


    #######################################
    #  Modules for adding nodes in graph  #
    #######################################
    def find_unreachable_nodes(self,points):
        """
            Find points outside the node coverage of any graph
            @params: 
                points: (Nx3) np.ndarray 

            Note:- vertices are sorted in descending order w.r.t distance from ED Graph    
        """
        dist, nearest_node = self.source_frame_kdtree.query(points, k=1)
        dist = dist.reshape(-1)
        unreachable_verts = np.where(dist > self.node_coverage)[0]
        print("Unreachable nodes:",unreachable_verts)
        if len(unreachable_verts) == 0:
            return np.zeros((0,3)) # Empty array 


        # Sort vertices by their distance to graph 
        dist = dist[unreachable_verts]
        dist_arg_sort = np.argsort(dist)[::-1]
        print("Dist:")
        print("Sorted by their distance:",dist_arg_sort)
        unreachable_verts = unreachable_verts[dist_arg_sort]

        new_verts = points[unreachable_verts]
        return new_verts 

    def update_graph(self):
        """
            1. After fusion if new nodes are added to the graph.
            2. Update skinning parameters for data structure(tsdf)  
        """

        self.updating_warpfield = True

        # Use the embedded deformation graph 
        canonical_model = self.tsdf.get_canonical_model()

        # Erode mesh to remove outliers 
        canonical_model_vertices, canonical_model_faces = canonical_model[0],canonical_model[1]

        # TODO: Visvualize num_iterations hyperparamater
        canonical_model_non_eroded_indices = self.graph.erode_mesh(canonical_model_vertices, canonical_model_faces,num_iterations=3) 
        canonical_model_non_eroded_indices = canonical_model_non_eroded_indices.reshape(-1)

        # print(canonical_model_non_eroded_indices)
        # print(canonical_model_non_eroded_indices.shape)


        # Get vertices not skinned, sorted by their ditance from graph 
        new_verts = self.find_unreachable_nodes(canonical_model_vertices[canonical_model_non_eroded_indices])
        
        old_num_nodes = self.graph.nodes.shape[0]
        update = self.graph.update(new_verts) # Update graph and return whether succesfully updated or not 

        if update:
            ##################################
            # Update skinning parameters     #
            ##################################
            self.source_frame_kdtree = KDTreeCPU(self.graph.nodes, leafsize=self.kdtree_leaf_size) # Update the main kdtree 
            self.skin_tsdf()

            ##################################    
            # Update deformation parameters  #
            ##################################

            visible_nodes = np.zeros(self.graph.nodes.shape[0],dtype=np.bool)
            visible_nodes[:old_num_nodes] = True

            graph_data = self.graph.get_reduced_graph(visible_nodes) # Get all nodes 


            # Note
            graph_data["graph_nodes"] = graph_data["nodes"] # While running ARAP to add nodes, source_frame = canonical frame adding

            # for k in graph_data:
            #     try: 
            #         print(k,graph_data[k].shape)
            #     except: 
            #         continue 

            transformation_data = {"source_frame_id":self.tsdf.fopt.source_frame,"target_frame_id":self.frame_id}
            transformation_data["node_rotations"],transformation_data["node_translations"] = self.get_transformation_wrt_graph_node() # Get their transformations     
            transformation_data["deformed_graph_nodes"] = self.graph.nodes[visible_nodes] + transformation_data["node_translations"]    

            estimated_new_graph_parameters = self.model.run_arap(graph_data,transformation_data,self.graph,self)

            # TODO: Needs refactoring  
            self.deformed_graph_nodes = self.graph.nodes + estimated_new_graph_parameters["node_translations"] # Just update graph nodes (Unlike during registration no need to save previous data)

            self.rotations,self.translations = self.get_transformation_wrt_origin(estimated_new_graph_parameters["node_rotations"],estimated_new_graph_parameters["node_translations"])


            

        # Finally update the frame id to target frame showing that the defornmation is completed
        self.frame_id = self.tsdf.frame_id
        self.updating_warpfield = False

        return update


    def get_deformed_graph_nodes(self):

        # Make sure the translation nodes are updated 
        assert self.frame_id == self.tsdf.frame_id,f"Warpfield maps to:{self.frame_id}th frame but TSDF maps to:{self.tsdf.frame_id}th frame" # Assert warpfield updated with previous deformation 
        assert self.graph.nodes.shape == self.translations.shape # Assert new graph nodes and their translation is updated  

        # TODO checks for  

        return self.deformed_graph_nodes

    def save_deformation_parameters(self):
        pass

    def load_deformation_parameters(self):    
        pass
        # assert graph nodes size and parameters size 

    def clear(self):
        del self.source_im
        

# @staticmethod
@ncuda.jit(device=True)
def warp_point_with_nodes(nodes_rotation, nodes_translation, now_x, now_y, now_z):
    # now_x = pos_x-node_positions[0]
    # now_y = pos_y-node_positions[1]
    # now_z = pos_z-node_positions[2]


    new_x = nodes_rotation[0, 0] * now_x + \
        nodes_rotation[0, 1] * now_y +\
        nodes_rotation[0, 2] * now_z

    new_y = nodes_rotation[1, 0] * now_x + \
        nodes_rotation[1, 1] * now_y +\
        nodes_rotation[1, 2] * now_z

    new_z = nodes_rotation[2, 0] * now_x + \
        nodes_rotation[2, 1] * now_y +\
        nodes_rotation[2, 2] * now_z

    new_x += nodes_translation[0]
    new_y += nodes_translation[1]
    new_z += nodes_translation[2]

    return new_x, new_y, new_z

# @staticmethod
# @njit(parallel=True)
# def deform_world_points_dqs(node_quaternions, node_translations, world_pts, world_anchors, world_weights, graph_nodes):
#     deformed_world_pts = np.empty_like(world_pts, dtype=np.float32)
#     for i in prange(world_pts.shape[0]):

#         world_weights_normalized = world_weights[i]/np.sum(world_weights[i])
#         if np.sum(world_weights[i]) <= 1e-6:  # Consider no deformation
#             deformed_world_pts[i] = world_pts[i]
#         else:
#             deformed_world_pts[i, :] = 0
#             for k in range(world_weights_normalized.shape[0]):
#                 if world_weights_normalized[k] == 1e-6:
#                     continue

#                 ref_position = world_pts[i] - graph_nodes[world_anchors[i, k]]
#                 q = node_quaternions[world_anchors[i, k]]
#                 rotated_point = ref_position + np.cross(q[1:]*2.0, np.cross(q[1:], ref_position) + ref_position*q[0])
#                 deformed_world_pts[i] += world_weights_normalized[k]\
#                                         * (rotated_point + graph_nodes[world_anchors[i, k]]
#                                         + node_translations[world_anchors[i, k]])

#     return deformed_world_pts
