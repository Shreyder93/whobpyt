# importage

import warnings
warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt
from matplotlib import cm

import os,sys,glob,numpy as np, pandas as pd

import time

from skimage import measure

import nibabel as nib
from nilearn.plotting import plot_surf, plot_surf_stat_map, plot_roi, plot_anat, plot_surf_roi

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from nilearn.image import index_img

from dipy.core.gradients import gradient_table
from dipy.reconst import shm
from dipy.direction import peaks

from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.io.dpy import Dpy


import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap

from dipy.tracking.distances import approx_polygon_track

import nilearn


# ---------------------------------------------------------------------------------------------------------------

start_time = time.time()

seed_density = sys.argv[1]  
seed_density = int(seed_density)

# ---------------------------------------------------------------------------------------------------------------

# Load the DWI data

bvecs_file = '/external/rprshnas01/public_datasets/HCP/HCP_S900/100307/T1w/Diffusion/bvecs'
bvecs_dat = np.loadtxt(bvecs_file)

bvals_file = '/external/rprshnas01/public_datasets/HCP/HCP_S900/100307/T1w/Diffusion/bvals'
bvals_dat = np.loadtxt(bvals_file)


# laoding a dwi file is similar to loading in a dconn file ...

dwi_file = '/external/rprshnas01/public_datasets/HCP/HCP_S900/100307/T1w/Diffusion/data.nii.gz'
dwi_img = nib.load(dwi_file)
dwi_dat = dwi_img.get_data()

b0_img = index_img(dwi_img,0)

# load brain mask file ...

nbm_file = '/external/rprshnas01/public_datasets/HCP/HCP_S900/100307/T1w/Diffusion/nodif_brain_mask.nii.gz'
nbm_img = nib.load(nbm_file)
nbm_dat = nbm_img.get_data()


gtab = gradient_table(bvals_dat, bvecs_dat)

affine = dwi_img.affine

print('loaded dwi files \n')

# ---------------------------------------------------------------------------------------------------------------

# Take approx. 5 mins to run.

from dipy.reconst import dti
from dipy.segment.mask import median_otsu
from dipy.tracking import utils

dwi_data = dwi_img.get_fdata()

# Specify the volume index to the b0 volumes
dwi_data, dwi_mask = median_otsu(dwi_data, vol_idx=[0], numpass=1)

dti_model = dti.TensorModel(gtab)

# This step may take a while
dti_fit = dti_model.fit(dwi_data, mask=dwi_mask)

# Create the seeding mask
fa_img = dti_fit.fa
seed_mask = fa_img.copy()
seed_mask[seed_mask >= 0.2] = 1
seed_mask[seed_mask < 0.2] = 0

# Create the seeds
seeds = utils.seeds_from_mask(seed_mask, affine=affine, density=seed_density)

print('estimated fibre orienations \n')
# ---------------------------------------------------------------------------------------------------------------

from dipy.reconst.shm import CsaOdfModel
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion

sh_order = 2

csa_model = CsaOdfModel(gtab, sh_order=sh_order)

gfa = csa_model.fit(dwi_data, mask=seed_mask).gfa

stopping_criterion = ThresholdStoppingCriterion(gfa, .25)

from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.csdeconv import auto_response_ssst

# Takes approx. 2 mins to run ...

response, ratio = auto_response_ssst(gtab, dwi_data, roi_radii=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=sh_order)
csd_fit = csd_model.fit(dwi_data, mask=seed_mask)

from dipy.direction import ProbabilisticDirectionGetter
from dipy.data import small_sphere
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines

fod = csd_fit.odf(small_sphere)
pmf = fod.clip(min=0)
prob_dg = ProbabilisticDirectionGetter.from_pmf(pmf, max_angle=30., sphere=small_sphere)

print('auxiliary variables set \n')

# ---------------------------------------------------------------------------------------------------------------

streamline_generator = LocalTracking(prob_dg, stopping_criterion, seeds, affine, step_size=.5)

# Takes approx. 15 minutes to run at seed_density = 1. --> ~300,000 streamlines
# Takes approx. 90 minutes to run at seed_density = 2. --> ~2,700,000 streamlines
# Takes approx. 7 hours to run at seed_density = 3. --> ~9,100,000 streamlines

streamlines = Streamlines(streamline_generator)

sft = StatefulTractogram(streamlines, dwi_img, Space.RASMM)

streamlines = Streamlines(streamline_generator)

print('streamlines generated \n')

print('Number of streamlines = ', len(streamlines), '\n')

# ---------------------------------------------------------------------------------------------------------------

# This step take 3 minutes to run. 
streamlines_ds = np.array([approx_polygon_track(s) for s in streamlines])

print('streamlines downsampled \n')

# ---------------------------------------------------------------------------------------------------------------

# Load the label file and brain mask file

label_file_path = '/external/rprshnas01/netdata_kcni/jglab/MemberSpaces/Data/Shrey/PyTepFit/Data/fs_directory/100307/mri/100307_Schaefer2018_400Parcels_7Networks_rewritten.nii.gz'
mask_file_path = '/external/rprshnas01/public_datasets/HCP/HCP_S900/100307/T1w/Diffusion/nodif_brain_mask.nii.gz'

label_img = nib.load(label_file_path)
mask_img = nib.load(mask_file_path)

resampled_label_dat = nilearn.image.resample_img(label_file_path, target_affine=nbm_img.affine, target_shape=mask_img.shape,interpolation='nearest')

resampled_label_data = resampled_label_dat.get_data()

print('label data loaded \n')

# ---------------------------------------------------------------------------------------------------------------

conn_mat, grouping = utils.connectivity_matrix(streamlines_ds,
                                               b0_img.affine,
                                               resampled_label_data,
                                               return_mapping=True,
                                               mapping_as_streamlines=True)


conn_mat = conn_mat[1:, 1:]

print('Str Conn Mtx completed \n')

# ---------------------------------------------------------------------------------------------------------------

out_dir = '/external/rprshnas01/kcni/hharita/Code/whobpyt/Intro_to_dMRI_workshop'
np.savetxt(out_dir + '/version_2_sub_100307_dwi_str_conn_ntx_density_{0}.txt'.format(seed_density), conn_mat, fmt='%d')

print("Output saved! \n")

# ---------------------------------------------------------------------------------------------------------------

print("Time taken to complete : --- %s seconds ---" % (time.time() - start_time))

# ---------------------------------------------------------------------------------------------------------------