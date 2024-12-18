# @author: msharrock
# version: 0.0.1

'''
Extraction methods for DeepBleed

'''

import os

import cv2
import nibabel as nib
from skimage.morphology import disk, binary_dilation, binary_erosion, remove_small_objects

def skullstrip_2outputs(image):
    window_center, window_width = 40, 80
    img1 = image.get_fdata()
    img1[img1 < 0] = 0
    img1[img1 > 200] = 0
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img1[img1 < img_min] = img_min
    img1[img1 > img_max] = img_max
    img1 = (img1 - img1.min()) / np.ptp(img1)

    window_center, window_width = 75, 100
    img2 = image.get_fdata()
    img2[img2 < 0] = 0
    img2[img2 > 200] = 0
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img2[img2 < img_min] = img_min
    img2[img2 > img_max] = img_max
    img2 = (img2 - img2.min()) / np.ptp(img2)

    img_bw = img1.copy()
    img_bw[img_bw > 0] = 1
    for slice in range(0, img_bw.shape[2]):
        if slice < round(img_bw.shape[2] / 8) or slice > (img_bw.shape[2] - round(img_bw.shape[2] / 8)):
            img_bw[:, :, slice] = 0
        if img_bw[:, :, slice].sum() > 0:
            img_bw[:, :, slice] = binary_erosion(img_bw[:, :, slice].astype(np.uint8),
                                                 disk(4, dtype=bool))
            img_bw[:, :, slice] = remove_small_objects(img_bw[:, :, slice].astype(bool), 1000)
            img_bw[:, :, slice] = binary_dilation(img_bw[:, :, slice].astype(np.uint8),
                                                  disk(4, dtype=bool))
    img1[img_bw == 0] = 0
    img1 = 1.0 * (img1 - img1.min()) / np.ptp(img1)
    image_output1 = nib.Nifti1Image(img1, image.affine, image.header)

    img2[img_bw == 0] = 0
    img2 = 1.0 * (img2 - img2.min()) / np.ptp(img2)
    image_output2 = nib.Nifti1Image(img2, image.affine, image.header)

    return image_output1, image_output2
