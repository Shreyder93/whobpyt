#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
  echo "Error: No argument provided. Please provide a number."
  exit 1
fi

# Read the number from the first argument
number=$1

# Check if the argument is a valid number
re='^[0-9]+$'
if ! [[ $number =~ $re ]]; then
  echo "Error: Invalid argument. Please provide a valid number."
  exit 1
fi

# # FSL
# module load bio/FSL/6.0.5.1-centos7_64

# Connectome Workbench
# module load bio/ConnectomeWorkbench/1.5.0-rh_linux64 --> doesn't work with new version, so use the older version ... 

module load bio/ConnectomeWorkbench/1.3.2-foss-2018b

wb_command -cifti-parcellate /external/rprshnas01/public_datasets/HCP/HCP_S900/$number/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii /external/rprshnas01/netdata_kcni/jglab/Code/libraries_of_others/github/CBIG/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/HCP/fslr32k/cifti/Schaefer2018_400Parcels_7Networks_order.dlabel.nii COLUMN /nethome/kcni/hharita/Code/whobpyt/scratch/testing_ConnectomeWB/${number}_rfMRI_400Schaefer_7Ntwx_cifti_parcellated.ptseries.nii

wb_command -cifti-correlation /nethome/kcni/hharita/Code/whobpyt/scratch/testing_ConnectomeWB/${number}_rfMRI_400Schaefer_7Ntwx_cifti_parcellated.ptseries.nii /nethome/kcni/hharita/Code/whobpyt/scratch/testing_ConnectomeWB/${number}_rfMRI_400Schaefer_7Ntwx_cifti_parcellated.pconn.nii


mv /nethome/kcni/hharita/Code/whobpyt/scratch/testing_ConnectomeWB/${number}_rfMRI_400Schaefer_7Ntwx_cifti_parcellated.ptseries.nii /nethome/kcni/hharita/Code/whobpyt/scratch/testing_ConnectomeWB/${number}_rfMRI_400Schaefer_7Ntwx_cifti_parcellated.pconn.nii /external/rprshnas01/netdata_kcni/jglab/MemberSpaces/Data/Shrey/Shrey_SS_parcellated_Func_Conns_III