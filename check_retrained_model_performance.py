from keras.utils import to_categorical
from sklearn.ensemble import StackingClassifier
from keras.models import load_model
import time, sys, os, glob
import numpy as np
import argparse, pickle, errno
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
class NotADirectoryError(Exception):
    pass

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError("Directory path is not valid.")


parser = argparse.ArgumentParser(description='Score PFD or AR files based on Retrained SGAN model and calculate performance against a test-set')
parser.add_argument('-i', '--test_set_input_path', help='Absolute path of Test set directory', default="sample_data/")
parser.add_argument('-t', '--test_set', help='Test set file name',  default="test_set_relabelled_jan_2021.csv")

args = parser.parse_args()
path_to_data = args.test_set_input_path
test_set_file = args.test_set_input_path
dir_path(path_to_data)

test_set = pd.read_csv(path_to_data + test_set_file)

candidate_files = path_to_data + test_set['Filename'].to_numpy()
true_labels = test_set['Classification'].to_numpy()
basename_candidate_files = [os.path.basename(filename) for filename in candidate_files]




freq_phase_model = load_model('best_retrained_models/freq_phase_best_discriminator_model.h5')
time_phase_model = load_model('best_retrained_models/time_phase_best_discriminator_model.h5')
dm_curve_model = load_model('best_retrained_models/dm_curve_best_discriminator_model.h5')
pulse_profile_model = load_model('best_retrained_models/pulse_profile_best_discriminator_model.h5')


logistic_model = pickle.load(open('best_retrained_models/sgan_retrained.pkl', 'rb'))


dm_curve_combined_array = [np.load(filename[:-4] + '_dm_curve.npy') for filename in candidate_files]
pulse_profile_combined_array = [np.load(filename[:-4] + '_pulse_profile.npy') for filename in candidate_files]
freq_phase_combined_array = [np.load(filename[:-4] + '_freq_phase.npy') for filename in candidate_files]
time_phase_combined_array = [np.load(filename[:-4] + '_time_phase.npy') for filename in candidate_files]


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
classified_results = logistic_model.predict(stacked_predictions) # if you want a classification score
#classified_results = logistic_model.predict_proba(stacked_predictions)[:,1] # If you want a regression score

f_score = f1_score(true_labels, classified_results, average='binary')
precision = precision_score(true_labels, classified_results, average='binary')
recall = recall_score(true_labels, classified_results, average='binary')
accuracy = (true_labels == classified_results).sum()/len(true_labels)
tn, fp, fn, tp = confusion_matrix(true_labels, classified_results).ravel()
specificity = tn/(tn + fp)
gmean = math.sqrt(specificity * recall)
fpr = fp/(tn + fp)
print('Results with SGAN Retraining')
print('SGAN Model File:', 'best_retrained_models/sgan_retrained.pkl', 'Accuracy:', accuracy, 'F Score:', f_score, 'Precision:', precision, 'Recall:', recall, 'False Positive Rate:', fpr, 'Specificity:', specificity, 'G-Mean:', gmean)


