from keras.utils import to_categorical
from sklearn.ensemble import StackingClassifier
from keras.models import load_model
import time, sys
import numpy as np
total_epochs=400

labelled_samples = 30000
unlabelled_samples = 29283
attempt_no = 1
freq_phase_results = load_model('/fred/oz002/vishnu/sgan/semi_supervised_trained_models/freq_phase_best_discriminator_model_labelled_%d_unlabelled_%d_trial_%d.h5'%(labelled_samples, unlabelled_samples,  attempt_no))
time_phase_results = load_model('/fred/oz002/vishnu/sgan/semi_supervised_trained_models/time_phase_best_discriminator_model_labelled_%d_unlabelled_%d_trial_%d.h5'%(labelled_samples, unlabelled_samples,  attempt_no))
dm_curve_results = load_model('/fred/oz002/vishnu/sgan/semi_supervised_trained_models/dm_curve_best_discriminator_model_labelled_%d_unlabelled_%d_trial_%d.h5'%(labelled_samples, unlabelled_samples,  attempt_no))
pulse_profile_results = load_model('/fred/oz002/vishnu/sgan/semi_supervised_trained_models/pulse_profile_best_discriminator_model_labelled_%d_unlabelled_%d_trial_%d.h5'%(labelled_samples, unlabelled_samples,  attempt_no))


model = LogisticRegression()

#now test it in test data
dm_curve_test = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in dm_curve_test])
freq_phase_test = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in freq_phase_test])
time_phase_test = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in time_phase_test])
pulse_profile_test = np.array([np.interp(a, (a.min(), a.max()), (-1, +1)) for a in pulse_profile_test])
print('Size of test set is %d'%(len(dm_curve_test)))
for i in range(2):
    X_with_class = dm_curve_test[dm_curve_label_test == i]

    print('Number of examples in test set with label %d is %d' %(i, len(X_with_class)))

predictions_freq_phase = freq_phase_results.predict([freq_phase_test])
predictions_time_phase = time_phase_results.predict([time_phase_test])
predictions_dm_curve = dm_curve_results.predict([dm_curve_test])
predictions_pulse_profile = pulse_profile_results.predict([pulse_profile_test])

float_value_time_phase = np.amax(predictions_time_phase, 1)
float_value_freq_phase = np.amax(predictions_freq_phase, 1)
float_value_dm_curve = np.amax(predictions_dm_curve, 1)
float_value_pulse_profile = np.amax(predictions_pulse_profile, 1)

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

float_value_stacked_predictions_test = np.stack((float_value_freq_phase, float_value_time_phase, float_value_dm_curve, float_value_pulse_profile), axis=1)
stacked_predictions_test = np.stack((predictions_freq_phase, predictions_time_phase, predictions_dm_curve, predictions_pulse_profile), axis=1)
stacked_predictions_test = np.reshape(stacked_predictions_test, (25408,4))

classified_results = model.predict(stacked_predictions_test)
float_value_classified_results = model.predict_proba(stacked_predictions_test)[:,1]
t1 = time.time()
total_n = (t1-t0)
print('Time taken for the code to execute was %s seconds' %(total_n))
classified_results = np.rint(float_value_classified_results)
f_score = f1_score(freq_phase_label_test, classified_results, average='binary')
precision = precision_score(freq_phase_label_test, classified_results, average='binary')
recall = recall_score(freq_phase_label_test, classified_results, average='binary')
accuracy = (freq_phase_label_test == classified_results).sum()/len(freq_phase_label_test)
tn, fp, fn, tp = confusion_matrix(freq_phase_label_test, classified_results).ravel()
specificity = tn/(tn + fp)
gmean = math.sqrt(specificity * recall)
fpr = fp/(tn + fp)
print('Results with HTRU-S Lowlat Survey Supervised Learning: correct one')
print('Accuracy:', accuracy, 'F Score:', f_score, 'Precision:', precision, 'Recall:', recall, 'False Positive Rate:', fpr, 'Specificity:', specificity, 'G-Mean:', gmean)

class_names = ['Non-Pulsar','Pulsar']

cnf_matrix = confusion_matrix(freq_phase_label_test, classified_results)
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized Confusion Matrix')
np.save('/fred/oz002/vishnu/sgan/semi_supervised_trained_models/predictions/true_labels.npy', freq_phase_label_test)
np.save('/fred/oz002/vishnu/sgan/semi_supervised_trained_models/predictions/predicted_labels.npy', float_value_classified_results)

#
#if not os.path.isfile('/fred/oz002/vishnu/sgan/semi_supervised_trained_models/combined_score.csv'):
#    with open('/fred/oz002/vishnu/sgan/semi_supervised_trained_models/combined_score.csv', 'w') as f:
#        f.write('Labelled Samples' + ',' + 'Accuracy' + ',' + 'F-Score' + ',' + 'Precision' + ',' + 'Recall' + ',' + 'FPR' + ',' + 'Specificity' + ',' + 'G-Mean' + ',' + 'Total Epochs' + '\n')
#
#with open('/fred/oz002/vishnu/sgan/semi_supervised_trained_models/combined_score.csv', 'a') as f:
#        f.write(str(labelled_samples) + ',' + str(accuracy) + ',' + str(f_score) + ',' + str(precision) + ',' + str(recall) + ',' + str(fpr) + ',' + str(specificity) + ',' + \
#     str(gmean) + ',' + str(total_epochs) + '\n')


