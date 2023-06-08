#!/bin/bash

#SBATCH --job-name=fmriprep_anat
#SBATCH --output=logs/%x_%j.out 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24000
#SBATCH --time=24:00:00

input_string="$1"

module load tools/Singularity/3.8.5


## set up a trap that will clear the ramdisk if it is not cleared

function cleanup_ramdisk {
    echo -n "Cleaning up ramdisk directory /$SLURM_TMPDIR/ on "
    date
    rm -rf /$SLURM_TMPDIR
    echo -n "done at "
    date
}

# trap the termination signal, and call the function 'trap_term' when
# that happens, so results may be saved.

trap "cleanup_ramdisk" TERM

singularity run --cleanenv -B /external/rprshnas01/netdata_kcni/jglab/MemberSpaces/Data/Shrey/rs_fMRI_eyes_closed_depression_Dataset_Openneuro/proper_attempt:/data \
-B /external/rprshnas01/netdata_kcni/jglab/MemberSpaces/Data/Shrey/rs_fMRI_eyes_closed_depression_Dataset_Openneuro/proper_attempt/output_folder:/output \
-B /external/rprshnas01/netdata_kcni/jglab/MemberSpaces/Data/Shrey/rs_fMRI_eyes_closed_depression_Dataset_Openneuro/proper_attempt/work:/work \
/external/rprshnas01/external_data/openneuro/tigrlab_containers/FMRIPREP/nipreps_fmriprep_20.2.7-2022-01-24-5df135ac568c.simg \
/data /output participant \
--participant_label $input_string \
-w /work \
--skip-bids-validation \
--skull-strip-t1w skip \
--output-space T1w MNI152NLin2009cAsym \
--use-syn-sdc \
--cifti-output 91k \
--ignore slicetiming \
--fs-license-file /data/license.txt