# Importage
import warnings
warnings.filterwarnings('ignore')

# os stuff
import os
import sys

import time

import nibabel as nib
from nilearn.plotting import plot_surf, plot_surf_stat_map, plot_roi, plot_anat, plot_surf_roi
from nilearn.image import index_img

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

# ----------------------------------------------------------------------------------------------------------------

# Setup stuff ...

start_time = time.time()

sub_id = int(sys.argv[1])  

ntwx_name = sys.argv[2]

a = int(sys.argv[3])
b = int(sys.argv[4])
c = int(sys.argv[5])
d = int(sys.argv[6])

print("Stuff setup ...\n")

# ----------------------------------------------------------------------------------------------------------------

data_path = '/external/rprshnas01/netdata_kcni/jglab/MemberSpaces/Data/Shrey/WhoBPyT/200_subjects_WhoBPyT_run_pkls'
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
    test_data = pickle.load(f1)
    
filename2 = data_path + '/Subj_{0}_fittingresults_stim_exp.pkl'.format(sub_id)
with open(filename2, 'rb') as f2:
    alt_data = pickle.load(f2)

_test_var = np.corrcoef(np.corrcoef(ss_og_data.output_sim.bold_test)[mask],_var[mask])[0][1]
print('Correlation b/w empirical and original whobpyt simulation = ', _test_var, '\n')

print('Data loaded \n')

# ----------------------------------------------------------------------------------------------------------------

def ntwx_only_fc(fc, a,b,c,d):
    
    ntwx_only_lh = fc[a:b, a:b]
    ntwx_only_rh = fc[c:d, c:d]

    ntwx_only_lhrh = fc[a:b, c:d]
    ntwx_only_rhlh = fc[c:d, a:b]
    
#     mean_fc = ((np.mean(np.abs(ntwx_only_lh)) + np.mean(np.abs(ntwx_only_rh)))/2 + (np.mean(np.abs(ntwx_only_lhrh)) + np.mean(np.abs(ntwx_only_rhlh)))/2)
#     mean_fc = np.mean(ntwx_only_lh) + np.mean(ntwx_only_rh) + (np.mean(ntwx_only_lhrh) + np.mean(ntwx_only_rhlh))/2
#     mean_fc = (np.mean(ntwx_only_lh) + np.mean(ntwx_only_rh) + np.mean(ntwx_only_lhrh) + np.mean(ntwx_only_rhlh))/4
    mean_fc = np.mean(ntwx_only_lh) + np.mean(ntwx_only_rh) + (np.mean(ntwx_only_lhrh) + np.mean(ntwx_only_rhlh))/2
    
    return mean_fc

# ----------------------------------------------------------------------------------------------------------------

og_whobpyt_bold_ts = ss_og_data.output_sim.bold_test
og_whobpyt_bold_fc = np.corrcoef(ss_og_data.output_sim.bold_test)
og_whobpyt_ntwx_fc = ntwx_only_fc(og_whobpyt_bold_fc, a,b,c,d)

list_of_ntwx_resilience_fc = []


list_of_ntwx_resilience_fc.append(og_whobpyt_ntwx_fc)

fitted_sc = np.abs(ss_og_data.model.gains_con.detach().numpy())
    
sorted_pairs= []
node_degree_all_nodes = []

for k in range(len(fitted_sc)):
    node_degree = np.sum(fitted_sc[k])
    node_degree_all_nodes.append(node_degree)
    

value_index_pairs = list(enumerate(node_degree_all_nodes))

sorted_pairs = sorted(value_index_pairs, key=lambda x:x[1],reverse=True)    
    
filtered_pairs = [(left, right) for left, right in sorted_pairs if (a <= left < b) or (c <= left < d)]    

_df = pd.DataFrame(filtered_pairs,columns=['Index','Node Degree'])   
idx_list = _df['Index'].tolist()
_regions = idx_list.copy()

p = len(_regions)

percent_lesions = []
for j in range(10):
    q=(np.round((10*(j+1)/100)*p))
    percent_lesions.append(q)
    
percent_lesions = [int(x) for x in percent_lesions]

alternate_sc = test_data.model.sc.copy()

for i in range(len(percent_lesions)):
    if i == 0:
        alternate_sc[_regions[:percent_lesions[i]],:] = 0
    elif i > 0:
        alternate_sc[_regions[percent_lesions[i-1]:percent_lesions[i]],:] = 0

    alt_data.model.sc = alternate_sc.copy()

    # run the model
    alt_data.test(20)
    
    ntwx_resilience_bold_ts = alt_data.output_sim.bold_test
    ntwx_resilience_bold_fc = np.corrcoef(ntwx_resilience_bold_ts)
    ntwx_resilience_ntwx_fc = ntwx_only_fc(ntwx_resilience_bold_fc, a,b,c,d)
    
    list_of_ntwx_resilience_fc.append(ntwx_resilience_ntwx_fc)
    
print('Network Resilience tested!')
    
# ----------------------------------------------------------------------------------------------------------------

# Save output list

output_path = '/external/rprshnas01/netdata_kcni/jglab/MemberSpaces/Data/Shrey/WhoBPyT/200_subjects_WhoBPyT_run_Ntwx_Resilience_VI/'

# 200_subjects_WhoBPyT_run_Ntwx_Resilience --> mean_fc = lh+rh+(lhrh+rhlh)/2 (high values)
# 200_subjects_WhoBPyT_run_Ntwx_Resilience_II --> mean_fc = (lh+rh)/2+(lhrh+rhlh)/2 (medium values)
# 200_subjects_WhoBPyT_run_Ntwx_Resilience_III --> mean_fc = (lh+rh+lhrh+rhlh)/4 (low values)
# 200_subjects_WhoBPyT_run_Ntwx_Resilience_IV --> mean_fc = lh+rh+(lhrh+rhlh)/2 ; only the ntwx cols are 0, not all cols.
# 200_subjects_WhoBPyT_run_Ntwx_Resilience_V --> percent_lesions

# Open a text file for writing
with open(output_path + "{0}_list_of_{1}_ntwx_resilience_fc.txt".format(sub_id,ntwx_name), "w") as file:
    # Iterate through the list and write each integer on a new line
    for num in list_of_ntwx_resilience_fc:
        file.write(str(num) + "\n")

# ----------------------------------------------------------------------------------------------------------------

print("Time taken to complete : --- %s seconds ---" % (time.time() - start_time))
