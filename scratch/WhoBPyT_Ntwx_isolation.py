# Importage
import warnings
warnings.filterwarnings('ignore')

# os stuff
import os
import sys

import nibabel as nib
from nilearn.plotting import plot_surf, plot_surf_stat_map, plot_roi, plot_anat, plot_surf_roi
from nilearn.image import index_img

import seaborn as sns
import time
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
import pickle
# viz stuff
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------

# Setup stuff ...

start_time = time.time()

sub_id = int(sys.argv[1])  

ntwx_name = sys.argv[2]

a = int(sys.argv[3])
b = int(sys.argv[4])
c = int(sys.argv[5])
d = int(sys.argv[6])

print("Stuff setup ...")

# ----------------------------------------------------------------------------------------------------------------
data_path = '/external/rprshnas01/netdata_kcni/jglab/MemberSpaces/Data/Shrey/WhoBPyT/200_subjects_WhoBPyT_run'
pconn_path = '/external/rprshnas01/netdata_kcni/jglab/MemberSpaces/Data/Shrey/Shrey_SS_parcellated_Func_Conns_II/'

pconn1LR = pconn_path + '{0}_rfMRI_REST1_RL_Schaefer200_cifti_parcellated.ptseries.nii'.format(sub_id)
pconn_img1LR = nib.load(pconn1LR)
pconn_dat1LR = pconn_img1LR.get_data()
pconn_dat1LR = pconn_dat1LR/1

node_size = 200
mask = np.tril_indices(node_size, -1)

_var = np.corrcoef(pconn_dat1LR.T)

filename = data_path + '/Subj_{0}_fittingresults_stim_exp.pkl'.format(sub_id)
with open(filename, 'rb') as f:
    ss_og_data = pickle.load(f)

filename1 = data_path + '/Subj_{0}_fittingresults_stim_exp.pkl'.format(sub_id)
with open(filename1, 'rb') as f1:
    ntwx_iso_data = pickle.load(f1)
    
filename2 = data_path + '/Subj_{0}_fittingresults_stim_exp.pkl'.format(sub_id)
with open(filename2, 'rb') as f2:
    ntwx_iso_data_cc_cut = pickle.load(f2)
    
# ss_og_data = g

_test_var = np.corrcoef(np.corrcoef(ss_og_data.output_sim.bold_test)[mask],_var[mask])[0][1]
print('Correlation b/w empirical and original whobpyt simulation = ', _test_var)

# dmn_iso_data = g2 

# dmn_iso_data_cc_cut = g3

original_sc = ss_og_data.model.sc.copy()

print('Data loaded')

# ----------------------------------------------------------------------------------------------------------------

# Structurally isolate the DMN ...

# Create a new matrix with the same shape as the original matrix

def structurally_isolate_func_ntwx(a,b,c,d):
    
    modified_matrix = original_sc.copy()

    modified_matrix[a:b,0:a] = 0
    modified_matrix[a:b,b:c] = 0
    modified_matrix[a:b,d:200] = 0
    modified_matrix[c:d,0:a] = 0
    modified_matrix[c:d,b:c] = 0
    modified_matrix[c:d,d:200] = 0

    modified_matrix[0:a,a:b] = 0
    modified_matrix[b:c,a:b] = 0
    modified_matrix[d:200,a:b] = 0
    modified_matrix[0:a,c:d] = 0
    modified_matrix[b:c,c:d] = 0
    modified_matrix[d:200,c:d] = 0
    
    modified_matrix = modified_matrix/np.linalg.norm(modified_matrix)
    
    return modified_matrix


isolated_ntwx = structurally_isolate_func_ntwx(a,b,c,d)

print('Target Network isolated!')
      
# ----------------------------------------------------------------------------------------------------------------

ntwx_isolated_sc = isolated_ntwx.copy()
      
ntwx_iso_data.model.sc = ntwx_isolated_sc.copy()

ntwx_iso_data.test(20)

output_path = '/external/rprshnas01/netdata_kcni/jglab/MemberSpaces/Data/Shrey/WhoBPyT/Ntwx_Lesion_WhoBPyT_200_subs'

ntwx_iso_fc_con_mat = np.corrcoef(ntwx_iso_data.output_sim.bold_test)

np.savetxt(output_path + '/Subj_{0}_{1}_lesion_fc_con_mat.txt'.format(sub_id, ntwx_name), ntwx_iso_fc_con_mat)    
# ---------------------------------------------------------------------------------------------------------------- 

modified_matrix_2 = isolated_ntwx.copy()
modified_matrix_2[c:d,a:b] = 0
modified_matrix_2[a:b,c:d] = 0

print('Corpus Callosum cut!')

# ---------------------------------------------------------------------------------------------------------------- 

ntwx_isolated_corpus_cut_sc = modified_matrix_2.copy()

ntwx_iso_data_cc_cut.model.sc = ntwx_isolated_corpus_cut_sc.copy()

ntwx_iso_data_cc_cut.test(20)

ntwx_iso_cc_cut_fc_con_mat = np.corrcoef(ntwx_iso_data_cc_cut.output_sim.bold_test)

np.savetxt(output_path + '/Subj_{0}_{1}_lesion_cc_cut_fc_con_mat.txt'.format(sub_id, ntwx_name), ntwx_iso_cc_cut_fc_con_mat) 

# ---------------------------------------------------------------------------------------------------------------- 

print("Time taken to complete : --- %s seconds ---" % (time.time() - start_time))
