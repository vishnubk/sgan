from keras.utils import to_categorical
from sklearn.ensemble import StackingClassifier
from keras.models import load_model
import time, sys, os, glob
import numpy as np
import argparse, pickle, errno

class NotADirectoryError(Exception):
    pass

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError("Directory path is not valid.")


parser = argparse.ArgumentParser(description='Score PFD files based on SGAN Machine Learning Model')
parser.add_argument('-i', '--input_path', help='Absolute path of Input directory', default="/fred/oz002/vishnu/sgan/sample_data/")
parser.add_argument('-o', '--output', help='Output file name',  default="/fred/oz002/vishnu/sgan/sample_data/")
parser.add_argument('-b', '--batch_size', help='No. of pfd files that will be read in one batch', default='1', type=int)
args = parser.parse_args()
path_to_data = args.input_path
batch_size = args.batch_size
output_path = args.output
dir_path(path_to_data)

pfd_files = sorted(glob.glob(path_to_data + '*.pfd'))
basename_pfd_files = [os.path.basename(filename) for filename in pfd_files]



labelled_samples = 50814
unlabelled_samples = 265172
attempt_no = 4
freq_phase_model = load_model('semi_supervised_trained_models/freq_phase_best_discriminator_model_labelled_%d_unlabelled_%d_trial_%d.h5'%(labelled_samples, unlabelled_samples,  attempt_no))
time_phase_model = load_model('semi_supervised_trained_models/time_phase_best_discriminator_model_labelled_%d_unlabelled_%d_trial_%d.h5'%(labelled_samples, unlabelled_samples,  attempt_no))
dm_curve_model = load_model('semi_supervised_trained_models/dm_curve_best_discriminator_model_labelled_%d_unlabelled_%d_trial_%d.h5'%(labelled_samples, unlabelled_samples,  attempt_no))
pulse_profile_model = load_model('semi_supervised_trained_models/pulse_profile_best_discriminator_model_labelled_%d_unlabelled_%d_trial_%d.h5'%(labelled_samples, unlabelled_samples,  attempt_no))

logistic_model = pickle.load(open('semi_supervised_trained_models/logistic_regression_labelled_%d_unlabelled_%d_trial_%d.pkl'%(labelled_samples, unlabelled_samples, attempt_no), 'rb'))


dm_curve_combined_array = [np.load(filename[:-4] + '_dm_curve.npy') for filename in pfd_files]
pulse_profile_combined_array = [np.load(filename[:-4] + '_pulse_profile.npy') for filename in pfd_files]
freq_phase_combined_array = [np.load(filename[:-4] + '_freq_phase.npy') for filename in pfd_files]
time_phase_combined_array = [np.load(filename[:-4] + '_time_phase.npy') for filename in pfd_files]


reshaped_time_phase = [np.reshape(f,(48,48,1)) for f in time_phase_combined_array]
reshaped_freq_phase = [np.reshape(f,(48,48,1)) for f in freq_phase_combined_array]
reshaped_pulse_profile = [np.reshape(f,(64,1)) for f in pulse_profile_combined_array]
reshaped_dm_curve = [np.reshape(f,(60,1)) for f in dm_curve_combined_array]

dm_curve_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_dm_curve])
pulse_profile_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_pulse_profile])
freq_phase_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_freq_phase])
time_phase_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_time_phase])

predictions_freq_phase = freq_phase_model.predict([freq_phase_data])
predictions_time_phase = time_phase_model.predict([time_phase_data])
predictions_dm_curve = dm_curve_model.predict([dm_curve_data])
predictions_pulse_profile = pulse_profile_model.predict([pulse_profile_data])

predictions_time_phase = np.rint(predictions_time_phase)
predictions_time_phase = np.argmax(predictions_time_phase, axis=1)
predictions_time_phase = np.reshape(predictions_time_phase, len(predictions_time_phase))

predictions_dm_curve = np.rint(predictions_dm_curve)
predictions_dm_curve = np.argmax(predictions_dm_curve, axis=1)
predictions_dm_curve = np.reshape(predictions_dm_curve, len(predictions_dm_curve))


predictions_pulse_profile = np.rint(predictions_pulse_profile)
predictions_pulse_profile = np.argmax(predictions_pulse_profile, axis=1)
predictions_pulse_profile = np.reshape(predictions_pulse_profile, len(predictions_pulse_profile))

predictions_freq_phase = np.rint(predictions_freq_phase)
predictions_freq_phase = np.argmax(predictions_freq_phase, axis=1)
predictions_freq_phase = np.reshape(predictions_freq_phase, len(predictions_freq_phase))


stacked_predictions = np.stack((predictions_freq_phase, predictions_time_phase, predictions_dm_curve, predictions_pulse_profile), axis=1)
stacked_predictions = np.reshape(stacked_predictions, (len(dm_curve_data),4))
#classified_results = logistic_model.predict(stacked_predictions) # if you want a classification score
classified_results = logistic_model.predict_proba(stacked_predictions)[:,1] # If you want a regression score

with open('sgan_ai_score.csv', 'w') as f:
    f.write('Filename, SGAN_score' + '\n') 
    for i in range(len(pfd_files)):
        f.write(basename_pfd_files[i] + ',' + str(classified_results[i]) + '\n')



