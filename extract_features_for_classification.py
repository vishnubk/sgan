import sys, os, glob
sys.path.append('/home/psr/software/psrchive/install/lib/python2.7/site-packages')
sys.path.append('/home/psr')
import numpy as np
from ubc_AI.training import pfddata
from ubc_AI.psrarchive_reader import ar2data 
import subprocess, argparse
import fnmatch

class NotADirectoryError(Exception):
    pass

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError("Directory path is not valid.")



parser = argparse.ArgumentParser(description='Extract pfd or ar files as numpy array files.')
parser.add_argument('-i', '--input_path', help='Absolute path of Input directory', default="/fred/oz002/vishnu/sgan/sample_data/")
parser.add_argument('-o', '--output', help='Output file name',  default="/fred/oz002/vishnu/sgan/sample_data/")
parser.add_argument('-b', '--batch_size', help='No. of pfd or ar files that will be read in one batch', default='1', type=int)
args = parser.parse_args()
path_to_data = args.input_path
batch_size = args.batch_size
output_path = args.output
dir_path(path_to_data)

pfd_files = sorted(glob.glob(path_to_data + '*.pfd'))
ar2_files = sorted(glob.glob(path_to_data + '*.ar2'))
basename_pfd_files = [os.path.basename(filename) for filename in pfd_files]
basename_ar2_files = [os.path.basename(filename) for filename in ar2_files]
# If you want to recursively get all candidates in subdirectories
#for root, dirnames, filenames in os.walk('/fred/oz002/vishnu/LOWLAT/candidate_plots/'):
#    for filename in fnmatch.filter(filenames, '*.pfd'):
#        matches.append(os.path.join(root, filename)) 
def chunks(pfd_files, n):
    """Yield n-sized chunks from list of pfd or ar2 files."""
    for i in range(0, len(pfd_files), n):
        yield pfd_files[i:i + n]

# Extract 4 features based on the work done by Zhu et.al 2014
if pfd_files:
    if batch_size > 1:
        batch_number=0
        for value in chunks(pfd_files,batch_size):

            batch_number+=1
            # Initialise data objects from getdata class
            data_object = [pfddata(filename) for filename in value]
            time_phase_data = [filename.getdata(intervals=48) for filename in data_object]
            freq_phase_data = [filename.getdata(subbands=48) for filename in data_object]
            pulse_profile_data = [filename.getdata(phasebins=64) for filename in data_object]
            dm_curve_data = [filename.getdata(DMbins=60) for filename in data_object]
    
            ###Save all features as numpy array files
            np.save(output_path + 'time_phase_data_batch_%d.npy' %int(batch_number), time_phase_data)
            np.save(output_path + 'freq_phase_data_batch_%d.npy'%int(batch_number), freq_phase_data)
            np.save(output_path + 'pulse_profile_data_batch_%d.npy' %int(batch_number), pulse_profile_data)
            np.save(output_path + 'dm_curve_data_batch_%d.npy' %int(batch_number), dm_curve_data)


    else:

        data_object = [pfddata(filename) for filename in pfd_files]
        time_phase_data = [filename.getdata(intervals=48) for filename in data_object]
        freq_phase_data = [filename.getdata(subbands=48) for filename in data_object]
        pulse_profile_data = [filename.getdata(phasebins=64) for filename in data_object]
        dm_curve_data = [filename.getdata(DMbins=60) for filename in data_object]
        ###Save all features as numpy array files
        for i in range(len(pfd_files)):
            np.save(output_path + basename_pfd_files[i][:-4] + '_time_phase.npy', time_phase_data[i])
            np.save(output_path + basename_pfd_files[i][:-4] + '_freq_phase.npy', freq_phase_data[i])
            np.save(output_path + basename_pfd_files[i][:-4] + '_pulse_profile.npy', pulse_profile_data[i])
            np.save(output_path + basename_pfd_files[i][:-4] + '_dm_curve.npy', dm_curve_data[i])

if ar2_files:
    if batch_size > 1:
        batch_number=0
        for value in chunks(ar2_files,batch_size):

            batch_number+=1
            # Initialise data objects from getdata class
            data_object = [ar2data(filename) for filename in value]
            time_phase_data = [filename.getdata(intervals=48) for filename in data_object]
            freq_phase_data = [filename.getdata(subbands=48) for filename in data_object]
            pulse_profile_data = [filename.getdata(phasebins=64) for filename in data_object]
            dm_curve_data = [filename.getdata(DMbins=60) for filename in data_object]

            ###Save all features as numpy array files
            np.save(output_path + 'time_phase_data_batch_%d.npy' %int(batch_number), time_phase_data)
            np.save(output_path + 'freq_phase_data_batch_%d.npy'%int(batch_number), freq_phase_data)
            np.save(output_path + 'pulse_profile_data_batch_%d.npy' %int(batch_number), pulse_profile_data)
            np.save(output_path + 'dm_curve_data_batch_%d.npy' %int(batch_number), dm_curve_data)


    else:

        data_object = [ar2data(filename) for filename in ar2_files]
        time_phase_data = [filename.getdata(intervals=48) for filename in data_object]
        freq_phase_data = [filename.getdata(subbands=48) for filename in data_object]
        pulse_profile_data = [filename.getdata(phasebins=64) for filename in data_object]
        dm_curve_data = [filename.getdata(DMbins=60) for filename in data_object]
        ###Save all features as numpy array files
        for i in range(len(ar2_files)):
            np.save(output_path + basename_ar2_files[i][:-4] + '_time_phase.npy', time_phase_data[i])
            np.save(output_path + basename_ar2_files[i][:-4] + '_freq_phase.npy', freq_phase_data[i])
            np.save(output_path + basename_ar2_files[i][:-4] + '_pulse_profile.npy', pulse_profile_data[i])
            np.save(output_path + basename_ar2_files[i][:-4] + '_dm_curve.npy', dm_curve_data[i])
