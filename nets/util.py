
from PIL import Image
import numpy as np
from skimage import color
import torch
import torch.nn.functional as F

def load_img(img_path): # transfer image into array
	out_np = np.asarray(Image.open(img_path))
	if(out_np.ndim==2):
		out_np = np.tile(out_np[:,:,None],3)
	return out_np

def resize_img(img, HW=(256,256), resample=3): # resize image to (H,W) using resample resampler
	return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))
	# resample=3 means use PIL.Image.LANCZOS, a high quality downsampling resampler

def preprocess_img(img_rgb_orig, HW=(256,256), resample=3):
	# return original size L and resized L as torch Tensors
	img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
	
	img_lab_orig = color.rgb2lab(img_rgb_orig) # lab before resize
	img_lab_rs = color.rgb2lab(img_rgb_rs) # lab after resize

	# get l channel in image
	img_l_orig = img_lab_orig[:,:,0] 
	img_l_rs = img_lab_rs[:,:,0]

	# change array into tensor
	tens_orig_l = torch.Tensor(img_l_orig)[None,None,:,:]
	tens_rs_l = torch.Tensor(img_l_rs)[None,None,:,:]

	return (tens_orig_l, tens_rs_l)

def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
	# tens_orig_l 	1 x 1 x H_orig x W_orig
	# out_ab 		1 x 2 x H x W

	HW_orig = tens_orig_l.shape[2:]
	HW = out_ab.shape[2:]

	# call resize function if needed
	if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
		out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
	else:
		out_ab_orig = out_ab

	# merge two tensor together

	out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
	return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0)))

def preprocess_data(data):
    data_l=np.empty(data[0,0,:,:][None,None,:,:].shape)
    data_ab=np.empty(data[0,1:,:,:][None,:,:,:].shape)

    for data_rgb in data:
        data_rgb=torch.transpose(data_rgb,0,1)
        data_rgb=torch.transpose(data_rgb,1,2)

        data_lab=color.rgb2lab(data_rgb)
        data_lab=np.transpose(data_lab,(2,0,1))

        data_l=np.append(data_l,data_lab[0,:,:][None,None,:,:],axis=0)
        data_ab=np.append(data_ab,data_lab[1:,:,:][None,:,:,:],axis=0)

    ans_l=torch.Tensor(data_l)[1:,:,:,:]
    ans_ab=torch.Tensor(data_ab)[1:,:,:,:]
    return (ans_l,ans_ab)
