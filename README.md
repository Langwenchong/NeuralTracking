# Neural Non-Rigid Tracking (NeurIPS 2020) + NonRigid Registration

### [Project Page](https://www.niessnerlab.org/projects/bozic2020nnrt.html) | [Paper](https://arxiv.org/abs/2006.13240) | [Video](https://www.youtube.com/watch?time_continue=1&v=nqYaxM6Rj8I&feature=emb_logo) | [Data](https://github.com/AljazBozic/DeepDeform)

<p align="center">
  <img width="100%" src="media/cloth_non_rigid.gif"/>
</p>

**Note:- This repo also includes the non-rigid registration pipeline. The implemetation follows the dynamicfusion using neural tracking as the optimization framework.**

This repository contains the code for the NeurIPS 2020 paper [Neural Non-Rigid Tracking](https://arxiv.org/abs/2006.13240), where we introduce a novel, end-to-end learnable, differentiable non-rigid tracker that enables state-of-the-art non-rigid reconstruction. 

By enabling gradient back-propagation through a weighted non-linear least squares solver, we are able to learn correspondences and confidences in an end-to-end manner such that they are optimal for the task of non-rigid tracking. 

Under this formulation, correspondence confidences can be learned via self-supervision, informing a learned robust optimization, where outliers and wrong correspondences are automatically down-weighted to enable effective tracking.

<p align="center">
  <img width="100%" src="media/teaser.jpg"/>
</p>



## Installation
Follow the orignal installation guideline. [Link](https://github.com/DeformableFriends/NeuralTracking#installation)

#### Addiontal dependencies for non-rigid registation
1. [pykdtree](https://github.com/storpipfugl/pykdtree)
```
git clone https://github.com/storpipfugl/pykdtree
cd <pykdtree_dir>
python setup.py install
```
2. Scikit-image with masked marching cubes. You would have to use another fork to run marching cubes. 
```
git clone https://github.com/shubhMaheshwari/scikit-image.git
python setup.py build_ext -i
pip install .
```

## Usage 
Below is the command on how to run on deepdeform dataset
```
cd fusion
python fusion.py <path-to-data> <source-frame> <skip-interval> <key-frame-interval> <voxel-size>
```
Arguments:
-  **path-to-data**: Location of RGBD images. The folder must contain color, depth, mask subfolders. 
-   **source-frame**: (Default 0th frame) to would be used as source.  
-   **skip-interval**: (Default 1) increments in for-loop. How many frames to skip
-   **key-frame-interval**: (Default 50). TSDF gets reinitalized at every 50 frame. Similiar to Fusion4D
-   **voxel-size**: (Default 0.01). Size of voxel in meters for tsdf volume creation. 


## Data

The raw image data and flow alignments can be obtained at the [DeepDeform](https://github.com/AljazBozic/DeepDeform) repository.

The additionally generated graph data can be downloaded using this [link](http://kaldir.vc.in.tum.de/download/deepdeform_graph_v1.7z).

Both archives are supposed to be extracted in the same directory.

If you want to generate data on your own, also for a new sequence, you can specify frame pair and run:
```
python create_graph_data.py
```



## Citation
If you find our work useful in your research, please consider citing:

	@article{
	bozic2020neuraltracking,
	title={Neural Non-Rigid Tracking},
	author={Aljaz Bozic and Pablo Palafox and Michael Zoll{\"o}fer and Angela Dai and Justus Thies and Matthias Nie{\ss}ner},
	booktitle={NeurIPS},
	year={2020}
    }

    

## Related work
Some other related work on non-rigid tracking by our group:
* [Bozic et al. - DeepDeform: Learning Non-rigid RGB-D Reconstruction with Semi-supervised Data (2020)](https://niessnerlab.org/projects/bozic2020deepdeform.html)
* [Li et al. - Learning to Optimize Non-Rigid Tracking (2020)](https://niessnerlab.org/projects/li2020learning.html)


## License

The code from this repository is released under the MIT license, except where otherwise stated (i.e., `pwcnet.py`, `Eigen`).
