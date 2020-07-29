#!/usr/bin/env python
# coding: utf-8



import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D,  Activation, Dropout, Flatten, Dense, LeakyReLU, BatchNormalization, ZeroPadding2D, Reshape
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, log_loss
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras import backend
from keras.optimizers import Adam
#from keras import optimizers
import numpy as np
import math, time, pickle
import itertools
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, concatenate
from keras.models import Model, load_model
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import glob, os, sys
from sklearn.linear_model import LogisticRegression
import argparse, errno
import pandas as pd
from classifiers import Train_SGAN_DM_Curve, Train_SGAN_Pulse_Profile, Train_SGAN_Freq_Phase, Train_SGAN_Time_Phase
class NotADirectoryError(Exception):
    pass

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError("Directory path is not valid.")


'''
#Class Labels format
0 -> Non-Pulsar
1 -> Pulsar
-1 -> Unlabelled Candidate
'''
parser = argparse.ArgumentParser(description='Re-train SGAN Machine Learning Model using User input PFD Files')
parser.add_argument('-i', '--input_path', help='Absolute path of Input directory', default="/fred/oz002/vishnu/sgan/sample_data/")
parser.add_argument('-v', '--validation_path', help='Absolute path of validation data directory', default="/fred/oz002/vishnu/sgan/validation_data/")
parser.add_argument('-u', '--unlabelled_path', help='Absolute path of unlabelled data directory', default="/fred/oz002/vishnu/sgan/unlabelled_data/")
parser.add_argument('-o', '--output', help='Output path to save model',  default="/fred/oz002/vishnu/sgan/sample_data/")
parser.add_argument('-l', '--labels', help='File with classification labels',  default="/fred/oz002/vishnu/sgan/sample_data/training_labels.csv")
parser.add_argument('-w', '--validation_labels', help='File with validation classification labels',  default="/fred/oz002/vishnu/sgan/validation_data/validation_labels.csv")
parser.add_argument('-b', '--batch_size', help='No. of pfd files that will be read in one batch', default='1', type=int)



args = parser.parse_args()
path_to_data = args.input_path
validation_data_path = args.validation_path
unlabelled_data_path = args.unlabelled_path
batch_size = args.batch_size
output_path = args.output
training_labels_file = args.labels
validation_labels_file = args.validation_labels
dir_path(path_to_data)

training_labels = pd.read_csv(training_labels_file)
validation_labels = pd.read_csv(validation_labels_file)


for index, row in training_labels.iterrows():
    if not os.path.isfile(path_to_data + row['Filename']):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), row['Filename'])

pfd_files = sorted(glob.glob(path_to_data + '*.pfd'))
validation_files = sorted(glob.glob(validation_data_path + '*.pfd'))
unlabelled_files = sorted(glob.glob(unlabelled_data_path + '*.pfd'))


#basename_pfd_files = [os.path.basename(filename) for filename in pfd_files]

''' Load Data'''
dm_curve_combined_array = [np.load(filename[:-4] + '_dm_curve.npy') for filename in pfd_files]
pulse_profile_combined_array = [np.load(filename[:-4] + '_pulse_profile.npy') for filename in pfd_files]
freq_phase_combined_array = [np.load(filename[:-4] + '_freq_phase.npy') for filename in pfd_files]
time_phase_combined_array = [np.load(filename[:-4] + '_time_phase.npy') for filename in pfd_files]

''' Reshaping the data for the neural-nets to read '''

reshaped_time_phase = [np.reshape(f,(48,48,1)) for f in time_phase_combined_array]
reshaped_freq_phase = [np.reshape(f,(48,48,1)) for f in freq_phase_combined_array]
reshaped_pulse_profile = [np.reshape(f,(64,1)) for f in pulse_profile_combined_array]
reshaped_dm_curve = [np.reshape(f,(60,1)) for f in dm_curve_combined_array]

''' Rescale the data between -1 and +1 '''

dm_curve_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_dm_curve])
pulse_profile_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_pulse_profile])
freq_phase_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_freq_phase])
time_phase_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_time_phase])




''' Repeat above steps for validation data'''

dm_curve_validation_combined_array = [np.load(filename[:-4] + '_dm_curve.npy') for filename in validation_files]
pulse_profile_validation_combined_array = [np.load(filename[:-4] + '_pulse_profile.npy') for filename in validation_files]
freq_phase_validation_combined_array = [np.load(filename[:-4] + '_freq_phase.npy') for filename in validation_files]
time_phase_validation_combined_array = [np.load(filename[:-4] + '_time_phase.npy') for filename in validation_files]


reshaped_time_phase_validation = [np.reshape(f,(48,48,1)) for f in time_phase_validation_combined_array]
reshaped_freq_phase_validation = [np.reshape(f,(48,48,1)) for f in freq_phase_validation_combined_array]
reshaped_pulse_profile_validation = [np.reshape(f,(64,1)) for f in pulse_profile_validation_combined_array]
reshaped_dm_curve_validation = [np.reshape(f,(60,1)) for f in dm_curve_validation_combined_array]

dm_curve_validation_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_dm_curve_validation])
pulse_profile_validation_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_pulse_profile_validation])
freq_phase_validation_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_freq_phase_validation])
time_phase_validation_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_time_phase_validation])


''' Repeat above steps for unlabelled data'''

dm_curve_unlabelled_combined_array = [np.load(filename[:-4] + '_dm_curve.npy') for filename in unlabelled_files]
pulse_profile_unlabelled_combined_array = [np.load(filename[:-4] + '_pulse_profile.npy') for filename in unlabelled_files]
freq_phase_unlabelled_combined_array = [np.load(filename[:-4] + '_freq_phase.npy') for filename in unlabelled_files]
time_phase_unlabelled_combined_array = [np.load(filename[:-4] + '_time_phase.npy') for filename in unlabelled_files]


reshaped_time_phase_unlabelled = [np.reshape(f,(48,48,1)) for f in time_phase_unlabelled_combined_array]
reshaped_freq_phase_unlabelled = [np.reshape(f,(48,48,1)) for f in freq_phase_unlabelled_combined_array]
reshaped_pulse_profile_unlabelled = [np.reshape(f,(64,1)) for f in pulse_profile_unlabelled_combined_array]
reshaped_dm_curve_unlabelled = [np.reshape(f,(60,1)) for f in dm_curve_unlabelled_combined_array]

dm_curve_unlabelled_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_dm_curve_unlabelled])
pulse_profile_unlabelled_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_pulse_profile_unlabelled])
freq_phase_unlabelled_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_freq_phase_unlabelled])
time_phase_unlabelled_data = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in reshaped_time_phase_unlabelled])


label_population = training_labels['Classification'].value_counts()
print('Note: Label 0 are treated as NON-PULSAR candidates, Label 1 are treated as PULSAR candidates and Label -1 are treated as UNLABELLED candidates.')
for index, row in label_population.iteritems():
    print('Total Number of Candidates with label %d is %d' % (index,row))

training_labels = training_labels['Classification'].to_numpy()
validation_labels = validation_labels['Classification'].to_numpy()
unlabelled_lables = np.tile(-1, int(dm_curve_unlabelled_data.shape[0]))

dm_curve_instance = Train_SGAN_DM_Curve(dm_curve_data, training_labels, dm_curve_validation_data, validation_labels, dm_curve_unlabelled_data, unlabelled_lables, 2)
pulse_profile_instance = Train_SGAN_Pulse_Profile(pulse_profile_data, training_labels, pulse_profile_validation_data, validation_labels, pulse_profile_unlabelled_data, unlabelled_lables, 2)
freq_phase_instance = Train_SGAN_Freq_Phase(freq_phase_data, training_labels, freq_phase_validation_data, validation_labels, freq_phase_unlabelled_data, unlabelled_lables, 2)
time_phase_instance = Train_SGAN_Time_Phase(time_phase_data, training_labels, time_phase_validation_data, validation_labels, time_phase_unlabelled_data, unlabelled_lables, 2)



#########
d_model, c_model = dm_curve_instance.define_discriminator()
g_model = dm_curve_instance.define_generator()
gan_model = dm_curve_instance.define_gan(g_model, d_model)
dm_curve_instance.train(g_model, d_model, c_model, gan_model)
print('DM Curve Model Retraining has been completed')
d_model, c_model = pulse_profile_instance.define_discriminator()
g_model = pulse_profile_instance.define_generator()
gan_model = pulse_profile_instance.define_gan(g_model, d_model)
pulse_profile_instance.train(g_model, d_model, c_model, gan_model)

print('Pulse Profile Model Retraining has been completed')




d_model, c_model = freq_phase_instance.define_discriminator()
g_model = freq_phase_instance.define_generator()
gan_model = freq_phase_instance.define_gan(g_model, d_model)
freq_phase_instance.train(g_model, d_model, c_model, gan_model)

print('Freq-Phase Model Retraining has been completed')

d_model, c_model = time_phase_instance.define_discriminator()
g_model = time_phase_instance.define_generator()
gan_model = time_phase_instance.define_gan(g_model, d_model)
time_phase_instance.train(g_model, d_model, c_model, gan_model)

print('Time-Phase Model Retraining has been completed')

freq_phase_model = load_model('best_retrained_models/freq_phase_best_discriminator_model.h5')
time_phase_model = load_model('best_retrained_models/time_phase_best_discriminator_model.h5')
dm_curve_model = load_model('best_retrained_models/dm_curve_best_discriminator_model.h5')
pulse_profile_model = load_model('best_retrained_models/pulse_profile_best_discriminator_model.h5')

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

model = LogisticRegression()
stacked_results = np.stack((predictions_freq_phase, predictions_time_phase, predictions_dm_curve, predictions_pulse_profile), axis=1)
stacked_results = np.reshape(stacked_results, (len(predictions_freq_phase), 4))
model.fit(stacked_results, training_labels)
pickle.dump(model, open('best_retrained_models/sgan_retrained.pkl', 'wb'))

#logistic_model = pickle.load(open('best_retrained_models/sgan_retrained.pkl', 'rb'))
#logistic_model = pickle.load(open('semi_supervised_trained_models/logistic_regression_labelled_50814_unlabelled_265172_trial_4.pkl', 'rb'))

