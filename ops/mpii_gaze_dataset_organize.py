'''
   Credit to mjdietzx: https://github.com/mjdietzx/SimGAN/blob/master/utils/mpii_gaze_dataset_organize.py

   This script was used to organize mpii_gaze_dataset into something usable.
   It is provided as-is and is kind of hacky but better than nothing...
'''

import glob
import os
import uuid

import numpy as np
from PIL import Image
import scipy.io as sio


save_dir = '/mnt/data1/images/eye_gaze/MPIIGaze_Dataset'


def butchered_mp_normalized_matlab_helper(mat_file_path):
    """
    Normalized data is provided in matlab files in MPIIGaze Dataset and these are tricky to load with Python.
    This function was made with guessing and checking. Very frustrating.

    :param mat_file_path: Full path to MPIIGaze Dataset matlab file.
    :return: np array of images.
    """
    x = sio.loadmat(mat_file_path)
    y = x.get('data')
    z = y[0, 0]

    left_imgs = z['left']['image'][0, 0]
    right_imgs = z['right']['image'][0, 0]

    for img in np.concatenate((left_imgs, right_imgs)):
        Image.fromarray(img).resize((55, 35), resample=Image.ANTIALIAS).save(
            os.path.join(save_dir, '{}.png'.format(uuid.uuid4())))

    return

if __name__ == '__main__':
    try: os.makedirs(save_dir)
    except: pass

    for filename in glob.iglob('/mnt/data1/images/eye_gaze/MPIIGaze/Data/Normalized/**/*.mat', recursive=True):
        print(filename)
        butchered_mp_normalized_matlab_helper(filename)
