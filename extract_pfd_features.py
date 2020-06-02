import sys, os, glob
sys.path.append('/home/psr/software/psrchive/install/lib/python2.7/site-packages')
sys.path.append('/home/psr')
import numpy as np
from ubc_AI.training import pfddata
import subprocess
import fnmatch
#pfd_files_new_batch_nonpulsars = glob.glob('/beegfs/vishnu/scripts/neural_network/train/new_batch_nonpulsars/*.pfd')
#pfd_files_new_batch_nonpulsars = glob.glob('/fred/oz002/vishnu/neural_network/lowlat_cands/new_batch_nonpulsars/*.pfd')
#unlabelled_data = glob.glob('/fred/oz002/vishnu/LOWLAT/candidate_plots/')
pfd_files = glob.glob()
#for mediafile in glob.iglob(os.path.join('/fred/oz002/vishnu/LOWLAT/candidate_plots/', "2012-*", "*.pfd")):
matches = []
# If you want to recursively get all candidates in subdirectories
#for root, dirnames, filenames in os.walk('/fred/oz002/vishnu/LOWLAT/candidate_plots/'):
#    for filename in fnmatch.filter(filenames, '*.pfd'):
#        matches.append(os.path.join(root, filename)) 
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

# Extract 4 features based on Zhu et.al 2014
batch_number=0
for value in chunks(matches,6000):

    batch_number+=1
    # Initialise data objects from getdata class
    data_object_new_batch_nonpulsars = [pfddata(f) for f in value]


    #1 time vs phase plot
    time_phase_plots_new_batch_nonpulsars = [f.getdata(intervals=48) for f in data_object_new_batch_nonpulsars]

    #2 freq vs phase plot
    freq_phase_plots_new_batch_nonpulsars = [f.getdata(subbands=48) for f in data_object_new_batch_nonpulsars]
    #3 Pulse Profile
    pulse_profile_new_batch_nonpulsars = [f.getdata(phasebins=64) for f in data_object_new_batch_nonpulsars]
    #4 DM Curve
    dm_curve_new_batch_nonpulsars = [f.getdata(DMbins=60) for f in data_object_new_batch_nonpulsars]
    ###Save all features as numpy array files
    np.save('/fred/oz002/vishnu/neural_network/lowlat_cands/unlaballed_data/time_phase_data_unlabelled_batch_%d.npy' %int(batch_number), time_phase_plots_new_batch_nonpulsars)

    np.save('freq_phase_data_new_batch_nonpulsars.npy', freq_phase_plots_new_batch_nonpulsars)
    np.save('pulse_profile_data_new_batch_nonpulsars_batch_%d.npy' %int(batch_number), pulse_profile_new_batch_nonpulsars)
    np.save('dm_curve_data_new_batch_nonpulsars_batch_%d.npy' %int(batch_number), dm_curve_new_batch_nonpulsars)
