# This is version 2.
# Here I basically use the same script as WhoBPyT_HCP_subs.py with a slight modification to load the .csv SC weights files. 
# This is the 'pre' step towards isolating the DMN (or X ntwx) and seeing where FC comes from.
# Resolution: 200 Schaefer Parcellations (possible with higher).

# Importage

import warnings
warnings.filterwarnings('ignore')

# os stuff
import os
import sys
# sys.path.append('..')

import nibabel as nib
from nilearn.plotting import plot_surf, plot_surf_stat_map, plot_roi, plot_anat, plot_surf_roi
from nilearn.image import index_img

import seaborn as sns

# whobpyt stuff
import whobpyt
from whobpyt.data.dataload import dataloader
# from whobpyt.models.jansen_rit import RNNJANSEN
from whobpyt.models.wong_wang import RNNRWW
from whobpyt.datatypes.modelparameters import ParamsModel
from whobpyt.optimization.modelfitting import Model_fitting

# array and pd stuff
import numpy as np
import pandas as pd

# viz stuff
import matplotlib.pyplot as plt

import pickle

import time

print("Imports done ...")

# ----------------------------------------------------------------------------------------------------------------

# Setup stuff ...

start_time = time.time()

sub_id = sys.argv[1]  
sub_id = int(sub_id)

print("Stuff setup ...")

# ----------------------------------------------------------------------------------------------------------------

# Paths

Wts_Path = '/external/rprshnas01/netdata_kcni/jglab/MemberSpaces/Data/Shrey/Improved_WWD_HCP_model_runs/All_Subs_SC_Wts/Davide_HCP_Data_Matrix'

pconn_path = '/external/rprshnas01/netdata_kcni/jglab/MemberSpaces/Data/Shrey/Shrey_SS_parcellated_Func_Conns_II/'

# _sub_list = [200614, 199958, 177746, 164131, 141826, 130619, 127933, 116726, 100307, 100206]
# sub_list = list(reversed(_sub_list))

parcs = np.arange(0,200,1)

print("Paths set ...")

# ----------------------------------------------------------------------------------------------------------------

# define options for wong-wang model
node_size = 200
mask = np.tril_indices(node_size, -1)
num_epoches = 20 #50
batch_size = 20
step_size = 0.05
input_size = 2
tr = 2.0
repeat_size = 5

print("Set options ...")

# ----------------------------------------------------------------------------------------------------------------

# Load SC

_df = pd.read_csv(Wts_Path + '/{0}/{0}_new_atlas_Yeo.nii.csv'.format(sub_id), delimiter=' ',header=None)
df_trimmed = _df.iloc[:-31, :-31]
np_array = df_trimmed.values
sc_mtx = np_array + np_array.T # --> Symmetric

pre_laplachian_HCP_SC = sc_mtx.copy()

HCP_SC = pre_laplachian_HCP_SC.copy()

SC = HCP_SC.copy()
sc = np.log1p(SC) / np.linalg.norm(np.log1p(SC))

print("Loaded SC ...")

# ----------------------------------------------------------------------------------------------------------------

# Load FC

pconn_path = '/external/rprshnas01/netdata_kcni/jglab/MemberSpaces/Data/Shrey/Shrey_SS_parcellated_Func_Conns_II/'

pconn1LR = pconn_path + '{0}_rfMRI_REST1_RL_Schaefer200_cifti_parcellated.ptseries.nii'.format(sub_id)
_pconn_img1LR = nib.load(pconn1LR)
_pconn_dat1LR = _pconn_img1LR.get_data()
_pconn_dat1LR = _pconn_dat1LR/1

ts = _pconn_dat1LR.copy() # ts_pd.values
ts = ts / np.max(ts)
fc_emp = np.corrcoef(ts.T)

print("Loaded FC ...")

# ----------------------------------------------------------------------------------------------------------------

# %%
# prepare data structure of the model
data_mean = dataloader(ts, num_epoches, batch_size)

# %%
# get model parameters structure and define the fitted parameters by setting non-zero variance for the model
par = ParamsModel('RWW',  g=[400, 1/np.sqrt(10)], g_EE=[1.5, 1/np.sqrt(50)], g_EI =[0.8,1/np.sqrt(50)], \
                          g_IE=[0.6,1/np.sqrt(50)], I_0 =[0.2, 0], std_in=[0.0,0], std_out=[0.00,0])

print("Loaded other Stuff ...")

# ----------------------------------------------------------------------------------------------------------------

# %%
# call model want to fit
model = RNNRWW(node_size, batch_size, step_size, repeat_size, tr, sc, True, par)

# %%
# initial model parameters and set the fitted model parameter in Tensors
model.setModelParameters()

# %%
# call model fit
F = Model_fitting(model, data_mean, num_epoches, 2)

# %%
# model training
F.train(learningrate= 0.05)

# %%
# model test with 20 window for warmup
F.test(20)


print("Finished running WhoBPyT ...")

# ----------------------------------------------------------------------------------------------------------------

output_path = '/external/rprshnas01/netdata_kcni/jglab/MemberSpaces/Data/Shrey/WhoBPyT/200_subjects_WhoBPyT_run'

filename = output_path + '/Subj_{0}_fittingresults_stim_exp.pkl'.format(sub_id)
with open(filename, 'wb') as f:
    pickle.dump(F, f)
    
# outfilename = output_path + '/Subj_{0}_simEEG_stim_exp.pkl'.format(sub_id)
# with open(outfilename, 'wb') as f:
#     pickle.dump(F.output_sim, f)

print("Output saved!")

# ----------------------------------------------------------------------------------------------------------------
    
print("Time taken to complete : --- %s seconds ---" % (time.time() - start_time))
