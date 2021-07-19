import os
import cv2
import numpy as np
import matplotlib.pyplot as  plt

def generate_mask(method,data):
	if method == "otsu":
		return generate_mask_depth(data[-1,...])
	elif method == "color":
		return generate_mask_color(np.moveaxis(data[:3,...],0,-1))
	else:
		NotImplementedError(f"Method:{method} not implemented")

def generate_mask_depth(depth_image_path,cropper=None):
	depth_image = cv2.imread(depth_image_path,0)

	if cropper is not None:
		depth_image = cropper(depth_image)

	depth_image[depth_image == 0] = depth_image.max()
	ret2,th = cv2.threshold(depth_image,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	
	return (1 - th).astype(bool)



if __name__ == "__main__":
	deepdeform_dataset_path = "../../AT-Datasets/DeepDeform/val/seq005"
	depth_dir = os.path.join(deepdeform_dataset_path,"depth")
	im_depth= cv2.imread(os.path.join(depth_dir,'000100.png'),0)
	mask = generate_mask_depth(os.path.join(depth_dir,'000100.png'))

	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax1.imshow(im_depth.astype(np.uint8))
	ax2 = fig.add_subplot(122)
	ax2.imshow(mask)
	plt.show()
