# -*- coding: utf-8 -*-
r"""
=================================
Fitting JR model (Tepit data)
=================================

"""

# sphinx_gallery_thumbnail_number = 1



# %%
# Importage
# ---------------------------------------------------
#

# os stuff
import os
import sys
sys.path.append('..')

# whobpyt stuff
import whobpyt
from whobpyt.data.dataload import dataloader
from whobpyt.models.jansen_rit import RNNJANSEN
from whobpyt.models.wong_wang import RNNRWW
from whobpyt.datatypes.modelparameters import ParamsModel
from whobpyt.optimization.modelfitting import Model_fitting

# array and pd stuff
import numpy as np
import pandas as pd
import scipy.io


# viz stuff
import matplotlib.pyplot as plt

#gdown
import gdown
# %%
# define destination path and download data
des_dir = '../'
if not os.path.exists(des_dir):
    os.makedirs(des_dir)  # create folder if it does not exist
url = 'https://drive.google.com/drive/folders/1uXrtehuMlLBvPCV8zDaUYxF-MoMaD0fk'
os.chdir(des_dir)
gdown.download_folder(url, quiet = True, use_cookies = False)
os.chdir('examples/')


# %%
# get  EEG data
base_dir = '../Tepit/'
eeg_file = base_dir + 'eeg_data.npy'
eeg_data_all = np.load(eeg_file)
eeg_data = eeg_data_all.mean(0) 

eeg_data = eeg_data[:,700:1100] / 12

# %%
# get stimulus weights on regions
ki0 =np.loadtxt(base_dir + 'stim_weights.txt')[:,np.newaxis]

# %%
# get SC and distance template
sc_file = base_dir + 'Schaefer2018_200Parcels_7Networks_count.csv'
dist_file = base_dir + 'Schaefer2018_200Parcels_7Networks_distance.csv'
sc_df = pd.read_csv(sc_file, header=None, sep=' ')
sc = sc_df.values
dist_df = pd.read_csv(dist_file, header=None, sep=' ')
dist = dist_df.values
sc = np.log1p(sc) / np.linalg.norm(np.log1p(sc))

# %%
# define options for JR model
node_size = sc.shape[0]

output_size = eeg_data.shape[0]
batch_size = 20
step_size = 0.0001
num_epoches = 120
tr = 0.001
state_size = 6
base_batch_num = 200
time_dim = 400
state_size = 6
base_batch_num = 20
hidden_size = int(tr/step_size)



# %%
# prepare data structure of the model
data_mean = dataloader(eeg_data.T, num_epoches, batch_size)

# %%
# get model parameters structure and define the fitted parameters by setting non-zero variance for the model
lm = np.zeros((output_size,200))
lm_v = np.zeros((output_size,200))
par = ParamsModel('JR', A = [3.25, 0], a= [100, 2], B = [22, 0], b = [50, 1], g=[40, 2], g_f=[1, 0], g_b=[1, 0],\
                    c1 = [135, 1], c2 = [135*0.8, 1], c3 = [135*0.25, 1], c4 = [135*0.25, 1],\
                    std_in=[1, 1/10], vmax= [5, 0], v0=[6,0], r=[0.56, 0], y0=[2 , 1/4],\
                    mu = [1., 0.4], #k = [10, .3],
                    #cy0 = [5, 0], ki=[ki0, 0], k_aud=[k_aud0, 0], lm=[lm, 1.0 * np.ones((output_size, 200))+lm_v], \
                    cy0 = [50, 1], ki=[ki0, 0], lm=[lm, 5 * np.ones((output_size, node_size))+lm_v])

# %%
# call model want to fit
model = RNNJANSEN(node_size, batch_size, step_size, output_size, tr, sc, lm, dist, True, False, par)

# %%
# initial model parameters and set the fitted model parameter in Tensors
model.setModelParameters()

# %%
# call model fit
F = Model_fitting(model, data_mean, num_epoches, 0)

# %%
# model training
u = np.zeros((node_size,hidden_size,time_dim))
u[:,:,110:120]= 200
F.train(u=u)

# %%
# model test with 20 window for warmup
F.test(200, u =u)

# %%
# Plot SC and fitted SC

fig, ax = plt.subplots(1, 2, figsize=(5, 4))
im0 = ax[0].imshow(sc, cmap='bwr')
ax[0].set_title('The empirical SC')
fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
im1 = ax[1].imshow(F.model.sc_fitted.detach().numpy(), cmap='bwr')
ax[1].set_title('The fitted SC')
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
plt.show()


# %%
# Plot the EEG
fig, ax = plt.subplots(1,3, figsize=(12,8))
ax[0].plot(F.output_sim.P_test.T)
ax[0].set_title('Test: sourced EEG')
ax[1].plot(F.output_sim.eeg_test.T)
ax[1].set_title('Test')
ax[2].plot(eeg_data.T)
ax[2].set_title('empirical')
plt.show()