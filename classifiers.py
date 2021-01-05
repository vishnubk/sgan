import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Conv1D, MaxPooling1D,  Activation, Dropout, Flatten, Dense, LeakyReLU, BatchNormalization, ZeroPadding2D, Reshape, Lambda, Input, Dense, concatenate
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, log_loss
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras import backend 
from keras.optimizers import Adam
import numpy as np
import math, time
import itertools
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.initializers import RandomNormal
import glob, os, sys
#import argparse, errno
import pandas as pd
class Train_SGAN_DM_Curve:

    """
       Class to retrain SGAN using DM Curve Data.

                                                 """
    def __init__(self, data, labels, validation_data, validation_labels, unlabelled_data, unlabelled_labels, batch_size, bin_size=60):
        self.data = data
        self.labels = labels
        self.validation_data = validation_data
        self.validation_labels = validation_labels
        self.unlabelled_data = unlabelled_data
        self.unlabelled_labels = unlabelled_labels
        self.bin_size = bin_size
        self.batch_size = batch_size

    def Conv1DTranspose(self, input_tensor, filters, kernel_size, strides=2, padding='same'):

        """
        input_tensor: tensor, with the shape (batch_size, time_steps, dims)
        filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
        kernel_size: int, size of the convolution kernel
        strides: int, convolution step size
        padding: 'same' | 'valid'
        """
        x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
        x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
        x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
        return x

    def custom_activation(self, output):
        logexpsum = backend.sum(backend.exp(output), axis=-1, keepdims=True)
        result = logexpsum / (logexpsum + 1.0)
        return result

    def define_discriminator(self, n_classes=2):
        in_shape = (self.bin_size, 1)
        in_image = Input(shape=in_shape)
        model = Conv1D(32, 3, padding='same')(in_image)
        model = LeakyReLU(alpha=0.1)(model)
        model = MaxPooling1D(2, padding='same')(model)
        model = Dropout(0.25)(model)
        model = Conv1D(64, 3, padding='same')(model)
        model = LeakyReLU(alpha=0.1)(model)
        model = MaxPooling1D(pool_size=(2),padding='same')(model)
        model = Conv1D(128, 3, padding='same')(model)
        model = LeakyReLU(alpha=0.1)(model)
        model = MaxPooling1D(pool_size=(2),padding='same')(model)
        model = Dropout(0.4)(model)
        model = Flatten()(model)
        model = Dense(128)(model)
        model = LeakyReLU(alpha=0.1)(model)
        model = Dropout(0.5)(model)
        fe = Dense(n_classes)(model)
        # supervised output
        c_out_layer = Activation('softmax')(fe)
        # define and compile supervised discriminator model
        c_model = Model(in_image, c_out_layer)
        c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.0008, beta_1=0.5), metrics=['accuracy'])
        # unsupervised output
        d_out_layer = Lambda(self.custom_activation)(fe)
        # define and compile unsupervised discriminator model
        d_model = Model(in_image, d_out_layer)
        d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0008, beta_1=0.5))
        return d_model, c_model

    def define_generator(self, latent_dim=100):
        n_outputs=self.bin_size
        # image generator input
        in_lat = Input(shape=(latent_dim,))
        init = RandomNormal(mean=0.0, stddev=0.02)
        gen = Dense(32, kernel_initializer=init, use_bias=False)(in_lat)
        gen = BatchNormalization()(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Dense(16, kernel_initializer=init, use_bias=False)(gen)
        gen = BatchNormalization()(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Dense(32, kernel_initializer=init, use_bias=False)(gen)
        gen = BatchNormalization()(gen)
        gen = LeakyReLU(alpha=0.2)(gen)

        out_layer = Dense(n_outputs, activation='tanh')(gen)
        out_layer = Reshape((n_outputs, 1))(out_layer)
        # define model
        model = Model(in_lat, out_layer)
        return model



    # define the combined generator and discriminator model, for updating the generator
    def define_gan(self, g_model, d_model):
        # make weights in the discriminator not trainable
        d_model.trainable = False
        # connect image output from generator as input to discriminator
        gan_output = d_model(g_model.output)
        # define gan model as taking noise and outputting a classification
        model = Model(g_model.input, gan_output)
        # compile model
        opt = Adam(lr=0.003, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    ## select a supervised subset of the dataset, ensures classes are balanced
    def select_supervised_samples(self, dataset, attempt_no, n_samples=10000, n_classes=2):
        X, y = dataset
        X_list, y_list = list(), list()
        n_per_class = int(n_samples / n_classes)
        for i in range(n_classes):
            # get all images for this class
            X_with_class = X[y == i]
            print('Number of examples with label %d is %d' %(i, len(X_with_class)))
            np.random.seed(attempt_no)
            # choose random instances
            ix = np.random.randint(0, len(X_with_class), n_per_class)
            # add to list
            [X_list.append(X_with_class[j]) for j in ix]
            [y_list.append(i) for j in ix]
        return np.asarray(X_list), np.asarray(y_list)


    # select real samples
    def generate_real_samples(self, dataset, n_samples, noisy_labels=True):
        # split into images and labels
        images, labels = dataset
        # choose random instances
        ix = np.random.randint(0, images.shape[0], n_samples)
        # select images and labels
        X, labels = images[ix], labels[ix]
        # generate class labels
        if noisy_labels:
            y = np.random.uniform(0.9, 1, (n_samples, 1))
        else:
            y = np.ones((n_samples, 1))
        
        return [X, labels], y

    # generate points in latent space as input for the generator
    def generate_latent_points(self, latent_dim, n_samples):
        # generate points in the latent space
        z_input = np.random.randn(latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        z_input = z_input.reshape(n_samples, latent_dim)
        return z_input

    ## use the generator to generate n fake examples, with class labels
    def generate_fake_samples(self, generator, latent_dim, n_samples, noisy_labels=True):
        # generate points in latent space
        z_input = self.generate_latent_points(latent_dim, n_samples)
        # predict outputs
        images = generator.predict(z_input)
        # create class labels
        if noisy_labels:
            y = np.random.uniform(0.0,0.2, (n_samples, 1))
        else:
            y = np.zeros((n_samples, 1))
        
        return images, y


    def summarize_performance(self, step, g_model, c_model, latent_dim, dataset, validation_dataset, epoch_number, model_accuracy, save_best_model=True):
        ''' June 2020 update for this function. 

              1. After careful thought, this function now saves the best model based on the performance of validation data labels. This is the best way to have a fair comparison 
                 between semi-supervised & supervised algorithms.
 
              2. This is also important to do if we compare these results to another algorithm say 'Random forest'. DO NOT choose your best model based on results from the 
                 test data. That leads to an over-estimation of your model performance. Use the test dataset only in the end to compare all your models. 
        '''

        ''' Comment out this block of code below in case you want to view the plots the generator makes after every epoch'''
        # prepare fake examples
        #n_samples = 100
        #X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
        ## scale from [-1,1] to [0,1]
        #X = (X + 1) / 2.0
        ## plot images
        #for i in range(n_samples):
        #    # define subplot
        #    pyplot.subplot(10, 10, 1 + i)
        #    pyplot.axis('off')
        #    pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
        #    pyplot.plot(X[i])
        #pyplot.close()
             ##  save plot to file
        #filename1 = '/fred/oz002/vishnu/pulsar_candidate_demystifier/codes/semi_supervised_trained_models/dm_curve_generated_plot_%04d_labelled_%d_unlabelled_%d_trial_%d.png' % (step+1, labelled_samples, unlabelled_samples, attempt_no)
        #pyplot.savefig(filename1)
        
        validation_X, validation_y = validation_dataset
        _, acc = c_model.evaluate(validation_X, validation_y, verbose=0)
        with open('training_logs/model_performance_sgan_dm_curve.txt', 'a') as f:
            f.write('intermediate_models/dm_curve_c_model_epoch_%d.h5' % int(epoch_number) + ',' + '%.3f' % (acc) + '\n')

        if save_best_model == True:
            if acc > model_accuracy:
                print('Current Model has %.3f training accuracy which is better than previous best of %.3f. Will save it as as new best model.' % (acc * 100, model_accuracy * 100 ))
                filename2 = 'best_retrained_models/dm_curve_best_generator_model.h5'  
                g_model.save(filename2)
                filename3 = 'best_retrained_models/dm_curve_best_discriminator_model.h5'
                c_model.save(filename3)
                model_accuracy = acc

            else:
                print('Current Model is not as good as the best model. This model will not be saved.')

            return model_accuracy
        else:
            print('Classifier Accuracy: %.3f%%' % (acc * 100))
            # save the generator model
            filename2 = 'intermediate_models/dm_curve_g_model_epoch_%d.h5' %int(epoch_number)
            g_model.save(filename2)
            # save the classifier model
            filename3 = 'intermediate_models/dm_curve_c_model_epoch_%d.h5' %int(epoch_number)
            c_model.save(filename3)
            print('>Saved: %s, and %s' % (filename2, filename3))
            return model_accuracy

    ## train the generator and discriminator
    def train(self, g_model, d_model, c_model, gan_model, latent_dim=100, n_epochs=10):

        ''' Define datasets. unlabelled_labels are all equal to -1. These labels are not used during training.'''

        dataset = [self.data, self.labels]
        unlabelled_dataset = [self.unlabelled_data, self.unlabelled_labels] 
        validation_dataset = [self.validation_data, self.validation_labels]
        n_batch = self.batch_size 
        X_sup, y_sup = dataset
        epoch_number = 0
        model_accuracy = 0.
        # calculate the number of batches per training epoch
        bat_per_epo = int((dataset[0].shape[0] + unlabelled_dataset[0].shape[0]) / n_batch)

        print('batch per epoch is %d' % bat_per_epo)
        # calculate the number of training iterations
        n_steps = bat_per_epo * n_epochs
        # calculate the size of half a batch of samples
        half_batch = int(n_batch / 2)
        print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (n_epochs, n_batch, half_batch, bat_per_epo, n_steps))
        # manually enumerate epochs
        for i in range(n_steps):
            ''' update supervised discriminator (c) '''
            [Xsup_real, ysup_real], _ = self.generate_real_samples([X_sup, y_sup], half_batch)
            Xsup_real = np.reshape(Xsup_real, (half_batch,self.bin_size,1))
            c_loss, c_acc = c_model.train_on_batch(Xsup_real, ysup_real)
            ''' update unsupervised discriminator (d) '''
            [X_real, _], y_real = self.generate_real_samples(unlabelled_dataset, half_batch)
            X_real = np.reshape(X_real, (half_batch,self.bin_size,1))
            d_loss1 = d_model.train_on_batch(X_real, y_real)
            X_fake, y_fake = self.generate_fake_samples(g_model, latent_dim, half_batch)
            
            d_loss2 = d_model.train_on_batch(X_fake, y_fake)
            # update generator (g)
            X_gan, y_gan = self.generate_latent_points(latent_dim, n_batch), np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            #summarize loss on this batch
            print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (i+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss))
            #print('>%d, c[%.3f,%.0f]' % (i+1, c_loss, c_acc*100))
            # evaluate the model performance every so often
            if (i+1) % (bat_per_epo * 1) == 0:
                epoch_number+=1
                model_accuracy = self.summarize_performance(i, g_model, c_model, latent_dim, dataset, validation_dataset, epoch_number, model_accuracy)



class Train_SGAN_Pulse_Profile:

    """
       Class to retrain SGAN using Pulse Profile Data.

                                                 """
    def __init__(self, data, labels, validation_data, validation_labels, unlabelled_data, unlabelled_labels, batch_size, bin_size=64):
        self.data = data
        self.labels = labels
        self.validation_data = validation_data
        self.validation_labels = validation_labels
        self.unlabelled_data = unlabelled_data
        self.unlabelled_labels = unlabelled_labels
        self.bin_size = bin_size
        self.batch_size = batch_size

    def Conv1DTranspose(self, input_tensor, filters, kernel_size, strides=2, padding='same'):

        """
        input_tensor: tensor, with the shape (batch_size, time_steps, dims)
        filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
        kernel_size: int, size of the convolution kernel
        strides: int, convolution step size
        padding: 'same' | 'valid'
        """
        x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
        x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
        x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
        return x

    def custom_activation(self, output):
        logexpsum = backend.sum(backend.exp(output), axis=-1, keepdims=True)
        result = logexpsum / (logexpsum + 1.0)
        return result

    def define_discriminator(self, n_classes=2):
        in_shape = (self.bin_size, 1)
        in_image = Input(shape=in_shape)
        model = Conv1D(32, 7, padding='same')(in_image)
        model = LeakyReLU(alpha=0.1)(model)
        model = MaxPooling1D(2, padding='same')(model)
        model = Dropout(0.25)(model)
        model = Conv1D(64, 7, padding='same')(model)
        model = LeakyReLU(alpha=0.1)(model)
        model = MaxPooling1D(pool_size=(2),padding='same')(model)
        model = Conv1D(128, 7, padding='same')(model)
        model = LeakyReLU(alpha=0.1)(model)
        model = MaxPooling1D(pool_size=(2),padding='same')(model)
        model = Dropout(0.4)(model)
        model = Flatten()(model)
        model = Dense(128)(model)
        model = LeakyReLU(alpha=0.1)(model)
        model = Dropout(0.5)(model)
        fe = Dense(n_classes)(model)
        ''' supervised output '''
        c_out_layer = Activation('softmax')(fe)
        ''' define and compile supervised discriminator model '''
        c_model = Model(in_image, c_out_layer)
        c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.5), metrics=['accuracy'])
        ''' unsupervised output '''
        d_out_layer = Lambda(self.custom_activation)(fe)
        ''' define and compile unsupervised discriminator model '''
        d_model = Model(in_image, d_out_layer)
        d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.5))
        return d_model, c_model



    
    def define_generator(self, latent_dim=100):
        n_outputs=self.bin_size
        # image generator input
        in_lat = Input(shape=(latent_dim,))
        init = RandomNormal(mean=0.0, stddev=0.02)
        gen = Dense(32, kernel_initializer=init, use_bias=False)(in_lat)
        gen = BatchNormalization()(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Dense(16, kernel_initializer=init, use_bias=False)(gen)
        gen = BatchNormalization()(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Dense(32, kernel_initializer=init, use_bias=False)(gen)
        gen = BatchNormalization()(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        out_layer = Dense(n_outputs, activation='tanh')(gen)
        out_layer = Reshape((n_outputs, 1))(out_layer)
        # define model
        model = Model(in_lat, out_layer)
        return model




    # define the combined generator and discriminator model, for updating the generator
    def define_gan(self, g_model, d_model):
        # make weights in the discriminator not trainable
        d_model.trainable = False
        # connect image output from generator as input to discriminator
        gan_output = d_model(g_model.output)
        # define gan model as taking noise and outputting a classification
        model = Model(g_model.input, gan_output)
        # compile model
        opt = Adam(lr=0.001, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    ## select a supervised subset of the dataset, ensures classes are balanced
    def select_supervised_samples(self, dataset, attempt_no, n_samples=10000, n_classes=2):
        X, y = dataset
        X_list, y_list = list(), list()
        n_per_class = int(n_samples / n_classes)
        for i in range(n_classes):
            # get all images for this class
            X_with_class = X[y == i]
            print('Number of examples with label %d is %d' %(i, len(X_with_class)))
            np.random.seed(attempt_no)
            # choose random instances
            ix = np.random.randint(0, len(X_with_class), n_per_class)
            # add to list
            [X_list.append(X_with_class[j]) for j in ix]
            [y_list.append(i) for j in ix]
        return np.asarray(X_list), np.asarray(y_list)


    # select real samples
    def generate_real_samples(self, dataset, n_samples, noisy_labels=True):
        # split into images and labels
        images, labels = dataset
        # choose random instances
        ix = np.random.randint(0, images.shape[0], n_samples)
        # select images and labels
        X, labels = images[ix], labels[ix]
        # generate class labels
        if noisy_labels:
            y = np.random.uniform(0.9,1, (n_samples, 1))
        else:
            y = np.ones((n_samples, 1))

        return [X, labels], y

    # generate points in latent space as input for the generator
    def generate_latent_points(self, latent_dim, n_samples):
        # generate points in the latent space
        z_input = np.random.randn(latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        z_input = z_input.reshape(n_samples, latent_dim)
        return z_input

    ## use the generator to generate n fake examples, with class labels
    def generate_fake_samples(self, generator, latent_dim, n_samples, noisy_labels=True):
        # generate points in latent space
        z_input = self.generate_latent_points(latent_dim, n_samples)
        # predict outputs
        images = generator.predict(z_input)
        if noisy_labels:
        y = np.random.uniform(0, 0.2, (n_samples, 1))
        else:
            y = np.zeros((n_samples, 1))

        return images, y


    def summarize_performance(self, step, g_model, c_model, latent_dim, dataset, validation_dataset, epoch_number, model_accuracy, save_best_model=True):
        ''' June 2020 update for this function. 

              1. After careful thought, this function now saves the best model based on the performance of validation data labels. This is the best way to have a fair comparison 
                 between semi-supervised & supervised algorithms.
 
              2. This is also important to do if we compare these results to another algorithm say 'Random forest'. DO NOT choose your best model based on results from the 
                 test data. That leads to an over-estimation of your model performance. Use the test dataset only in the end to compare all your models. 
        '''

        ''' Comment out this block of code below in case you want to view the plots the generator makes after every epoch'''
        # prepare fake examples
        #n_samples = 100
        #X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
        ## scale from [-1,1] to [0,1]
        #X = (X + 1) / 2.0
        ## plot images
        #for i in range(n_samples):
        #    # define subplot
        #    pyplot.subplot(10, 10, 1 + i)
        #    pyplot.axis('off')
        #    pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
        #    pyplot.plot(X[i])
        #pyplot.close()
             ##  save plot to file
        #filename1 = '/fred/oz002/vishnu/pulsar_candidate_demystifier/codes/semi_supervised_trained_models/pulse_profile_generated_plot_%04d_labelled_%d_unlabelled_%d_trial_%d.png' % (step+1, labelled_samples, unlabelled_samples, attempt_no)
        #pyplot.savefig(filename1)
        
        validation_X, validation_y = validation_dataset
        _, acc = c_model.evaluate(validation_X, validation_y, verbose=0)
        with open('training_logs/model_performance_sgan_pulse_profile.txt', 'a') as f:
            f.write('intermediate_models/pulse_profile_c_model_epoch_%d.h5' % int(epoch_number) + ',' + '%.3f' % (acc) + '\n')

        if save_best_model == True:
            if acc > model_accuracy:
                print('Current Model has %.3f training accuracy which is better than previous best of %.3f. Will save it as as new best model.' % (acc * 100, model_accuracy * 100 ))
                filename2 = 'best_retrained_models/pulse_profile_best_generator_model.h5'  
                g_model.save(filename2)
                filename3 = 'best_retrained_models/pulse_profile_best_discriminator_model.h5'
                c_model.save(filename3)
                model_accuracy = acc

            else:
                print('Current Model is not as good as the best model. This model will not be saved.')

            return model_accuracy
        else:
            print('Classifier Accuracy: %.3f%%' % (acc * 100))
            # save the generator model
            filename2 = 'intermediate_models/pulse_profile_g_model_epoch_%d.h5' %int(epoch_number)
            g_model.save(filename2)
            # save the classifier model
            filename3 = 'intermediate_models/pulse_profile_c_model_epoch_%d.h5' %int(epoch_number)
            c_model.save(filename3)
            print('>Saved: %s, and %s' % (filename2, filename3))
            return model_accuracy

    ## train the generator and discriminator
    def train(self, g_model, d_model, c_model, gan_model, latent_dim=100, n_epochs=10):

        ''' Define datasets. unlabelled_labels are all equal to -1. These labels are not used during training.'''

        dataset = [self.data, self.labels]
        unlabelled_dataset = [self.unlabelled_data, self.unlabelled_labels] 
        validation_dataset = [self.validation_data, self.validation_labels]
        n_batch = self.batch_size 
        X_sup, y_sup = dataset
        epoch_number = 0
        model_accuracy = 0.
        # calculate the number of batches per training epoch
        bat_per_epo = int((dataset[0].shape[0] + unlabelled_dataset[0].shape[0]) / n_batch)

        print('batch per epoch is %d' % bat_per_epo)
        # calculate the number of training iterations
        n_steps = bat_per_epo * n_epochs
        # calculate the size of half a batch of samples
        half_batch = int(n_batch / 2)
        print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (n_epochs, n_batch, half_batch, bat_per_epo, n_steps))
        # manually enumerate epochs
        for i in range(n_steps):
            ''' update supervised discriminator (c) '''
            [Xsup_real, ysup_real], _ = self.generate_real_samples([X_sup, y_sup], half_batch)
            Xsup_real = np.reshape(Xsup_real, (half_batch,self.bin_size,1))
            c_loss, c_acc = c_model.train_on_batch(Xsup_real, ysup_real)
            ''' update unsupervised discriminator (d) '''
            [X_real, _], y_real = self.generate_real_samples(unlabelled_dataset, half_batch)
            X_real = np.reshape(X_real, (half_batch,self.bin_size,1))
            d_loss1 = d_model.train_on_batch(X_real, y_real)
            X_fake, y_fake = self.generate_fake_samples(g_model, latent_dim, half_batch)
            
            d_loss2 = d_model.train_on_batch(X_fake, y_fake)
            # update generator (g)
            X_gan, y_gan = self.generate_latent_points(latent_dim, n_batch), np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            #summarize loss on this batch
            print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (i+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss))
            #print('>%d, c[%.3f,%.0f]' % (i+1, c_loss, c_acc*100))
            # evaluate the model performance every so often
            if (i+1) % (bat_per_epo * 1) == 0:
                epoch_number+=1
                model_accuracy = self.summarize_performance(i, g_model, c_model, latent_dim, dataset, validation_dataset, epoch_number, model_accuracy)




class Train_SGAN_Freq_Phase:

    """
       Class to retrain SGAN using Freq Phase Data.

                                                 """
    def __init__(self, data, labels, validation_data, validation_labels, unlabelled_data, unlabelled_labels, batch_size, bin_size=48):
        self.data = data
        self.labels = labels
        self.validation_data = validation_data
        self.validation_labels = validation_labels
        self.unlabelled_data = unlabelled_data
        self.unlabelled_labels = unlabelled_labels
        self.bin_size = bin_size
        self.batch_size = batch_size

    def custom_activation(self, output):
        logexpsum = backend.sum(backend.exp(output), axis=-1, keepdims=True)
        result = logexpsum / (logexpsum + 1.0)
        return result

    def define_discriminator(self, n_classes=2):
        in_shape = (self.bin_size, self.bin_size, 1)
        in_image = Input(shape=in_shape)
        model = Conv2D(32, (7, 7), padding='same')(in_image)
        model = LeakyReLU(alpha=0.1)(model)
        model = MaxPooling2D((2, 2),padding='same')(model)
        model = Dropout(0.25)(model)
        model = Conv2D(64, (7, 7), padding='same')(model)
        model = LeakyReLU(alpha=0.1)(model)
        model = MaxPooling2D(pool_size=(2, 2),padding='same')(model)
        model = Dropout(0.25)(model)
        model = Conv2D(128, (7, 7), padding='same')(model)
        model = LeakyReLU(alpha=0.1)(model)
        model = MaxPooling2D(pool_size=(2, 2),padding='same')(model)
        model = Dropout(0.4)(model)
        model = Conv2D(256, (7, 7), padding='same')(model)
        model = LeakyReLU(alpha=0.1)(model)
        model = MaxPooling2D(pool_size=(2, 2),padding='same')(model)
        model = Dropout(0.4)(model)
        model = Flatten()(model)
        fe = Dense(n_classes)(model)
        ''' supervised output '''
        c_out_layer = Activation('softmax')(fe)
        ''' define and compile supervised discriminator model '''
        c_model = Model(in_image, c_out_layer)
        c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
        ''' unsupervised output '''
        d_out_layer = Lambda(self.custom_activation)(fe)
        ''' define and compile unsupervised discriminator model '''
        d_model = Model(in_image, d_out_layer)
        d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
        return d_model, c_model


    def define_generator(self, latent_dim=100):
        init = RandomNormal(mean=0.0, stddev=0.02)
        ''' image generator input '''
        in_lat = Input(shape=(latent_dim,))
        ''' foundation for 7x7 image '''
        n_nodes = 128 * 12 * 12
        gen = Dense(n_nodes, use_bias=False, kernel_initializer=init)(in_lat)
        gen = BatchNormalization()(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Reshape((12, 12, 128))(gen)

        ''' upsample to 24x24 '''
        gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', use_bias=False, kernel_initializer=init)(gen)
        gen = BatchNormalization()(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        ''' upsample to 48x48 '''
        gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
        gen = BatchNormalization()(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        ''' output '''
        out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
        ''' define model '''
        model = Model(in_lat, out_layer)
        return model


    # define the combined generator and discriminator model, for updating the generator
    def define_gan(self, g_model, d_model):
        # make weights in the discriminator not trainable
        d_model.trainable = False
        # connect image output from generator as input to discriminator
        gan_output = d_model(g_model.output)
        # define gan model as taking noise and outputting a classification
        model = Model(g_model.input, gan_output)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    ## select a supervised subset of the dataset, ensures classes are balanced
    def select_supervised_samples(self, dataset, attempt_no, n_samples=10000, n_classes=2):
        X, y = dataset
        X_list, y_list = list(), list()
        n_per_class = int(n_samples / n_classes)
        for i in range(n_classes):
            # get all images for this class
            X_with_class = X[y == i]
            print('Number of examples with label %d is %d' %(i, len(X_with_class)))
            np.random.seed(attempt_no)
            # choose random instances
            ix = np.random.randint(0, len(X_with_class), n_per_class)
            # add to list
            [X_list.append(X_with_class[j]) for j in ix]
            [y_list.append(i) for j in ix]
        return np.asarray(X_list), np.asarray(y_list)


    # select real samples

    def generate_real_samples(self, dataset, n_samples, noisy_labels=True):
        # split into images and labels
        images, labels = dataset
        # choose random instances
        ix = np.random.randint(0, images.shape[0], n_samples)
        # select images and labels
        X, labels = images[ix], labels[ix]
        # generate class labels
       
        if noisy_labels:
            y = np.random.uniform(0.7,1.2, (n_samples, 1))
        else:
            y = np.ones((n_samples, 1))
 
        #Ocasionally flip labels
        shuffler = np.random.uniform(0,1)
        if shuffler <= 0.05:
            y = np.random.uniform(0.0,0.3, (n_samples, 1))
        return [X, labels], y

    # generate points in latent space as input for the generator
    def generate_latent_points(self, latent_dim, n_samples):
        # generate points in the latent space
        z_input = np.random.randn(latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        z_input = z_input.reshape(n_samples, latent_dim)
        return z_input


    # use the generator to generate n fake examples, with class labels
    def generate_fake_samples(self, generator, latent_dim, n_samples, noisy_labels=True):
        # generate points in latent space
        z_input = self.generate_latent_points(latent_dim, n_samples)
        # predict outputs
        images = generator.predict(z_input)
        # create class labels
        if noisy_labels:
            y = np.random.uniform(0.0,0.3, (n_samples, 1))
        else:
            y = np.zeros((n_samples, 1))
        
        #Ocasionally flip labels
        shuffler = np.random.uniform(0,1)
        if shuffler <= 0.05:
            y = np.random.uniform(0.7,1.2, (n_samples, 1))

        return images, y



    def summarize_performance(self, step, g_model, c_model, latent_dim, dataset, validation_dataset, epoch_number, model_accuracy, save_best_model=True):
        ''' June 2020 update for this function. 

              1. After careful thought, this function now saves the best model based on the performance of validation data labels. This is the best way to have a fair comparison 
                 between semi-supervised & supervised algorithms.
 
              2. This is also important to do if we compare these results to another algorithm say 'Random forest'. DO NOT choose your best model based on results from the 
                 test data. That leads to an over-estimation of your model performance. Use the test dataset only in the end to compare all your models. 
        '''

        ''' Comment out this block of code below in case you want to view the plots the generator makes after every epoch'''
        # prepare fake examples
        #n_samples = 100
        #X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
        ## scale from [-1,1] to [0,1]
        #X = (X + 1) / 2.0
        ## plot images
        #for i in range(n_samples):
        #    # define subplot
        #    pyplot.subplot(10, 10, 1 + i)
        #    pyplot.axis('off')
        #    pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
        #    pyplot.plot(X[i])
        #pyplot.close()
             ##  save plot to file
        #filename1 = '/fred/oz002/vishnu/pulsar_candidate_demystifier/codes/semi_supervised_trained_models/freq_phase_generated_plot_%04d_labelled_%d_unlabelled_%d_trial_%d.png' % (step+1, labelled_samples, unlabelled_samples, attempt_no)
        #pyplot.savefig(filename1)
        
        validation_X, validation_y = validation_dataset
        _, acc = c_model.evaluate(validation_X, validation_y, verbose=0)
        with open('training_logs/model_performance_sgan_freq_phase.txt', 'a') as f:
            f.write('intermediate_models/freq_phase_c_model_epoch_%d.h5' % int(epoch_number) + ',' + '%.3f' % (acc) + '\n')

        if save_best_model == True:
            if acc > model_accuracy:
                print('Current Model has %.3f training accuracy which is better than previous best of %.3f. Will save it as as new best model.' % (acc * 100, model_accuracy * 100 ))
                filename2 = 'best_retrained_models/freq_phase_best_generator_model.h5'  
                g_model.save(filename2)
                filename3 = 'best_retrained_models/freq_phase_best_discriminator_model.h5'
                c_model.save(filename3)
                model_accuracy = acc

            else:
                print('Current Model is not as good as the best model. This model will not be saved.')

            return model_accuracy
        else:
            print('Classifier Accuracy: %.3f%%' % (acc * 100))
            # save the generator model
            filename2 = 'intermediate_models/freq_phase_g_model_epoch_%d.h5' %int(epoch_number)
            g_model.save(filename2)
            # save the classifier model
            filename3 = 'intermediate_models/freq_phase_c_model_epoch_%d.h5' %int(epoch_number)
            c_model.save(filename3)
            print('>Saved: %s, and %s' % (filename2, filename3))
            return model_accuracy

    ## train the generator and discriminator
    def train(self, g_model, d_model, c_model, gan_model, latent_dim=100, n_epochs=10):

        ''' Define datasets. unlabelled_labels are all equal to -1. These labels are not used during training.'''

        dataset = [self.data, self.labels]
        unlabelled_dataset = [self.unlabelled_data, self.unlabelled_labels] 
        validation_dataset = [self.validation_data, self.validation_labels]
        n_batch = self.batch_size 
        X_sup, y_sup = dataset
        epoch_number = 0
        model_accuracy = 0.
        # calculate the number of batches per training epoch
        bat_per_epo = int((dataset[0].shape[0] + unlabelled_dataset[0].shape[0]) / n_batch)

        print('batch per epoch is %d' % bat_per_epo)
        # calculate the number of training iterations
        n_steps = bat_per_epo * n_epochs
        # calculate the size of half a batch of samples
        half_batch = int(n_batch / 2)
        print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (n_epochs, n_batch, half_batch, bat_per_epo, n_steps))
        # manually enumerate epochs
        for i in range(n_steps):
            ''' update supervised discriminator (c) '''
            [Xsup_real, ysup_real], _ = self.generate_real_samples([X_sup, y_sup], half_batch)
            c_loss, c_acc = c_model.train_on_batch(Xsup_real, ysup_real)
            ''' update unsupervised discriminator (d) '''
            [X_real, _], y_real = self.generate_real_samples(unlabelled_dataset, half_batch)
            d_loss1 = d_model.train_on_batch(X_real, y_real)
            X_fake, y_fake = self.generate_fake_samples(g_model, latent_dim, half_batch)
            
            d_loss2 = d_model.train_on_batch(X_fake, y_fake)
            ''' update generator (g) '''
            X_gan, y_gan = self.generate_latent_points(latent_dim, n_batch), np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            #summarize loss on this batch
            print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (i+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss))
            #print('>%d, c[%.3f,%.0f]' % (i+1, c_loss, c_acc*100))
            # evaluate the model performance every so often
            if (i+1) % (bat_per_epo * 1) == 0:
                epoch_number+=1
                model_accuracy = self.summarize_performance(i, g_model, c_model, latent_dim, dataset, validation_dataset, epoch_number, model_accuracy)

class Train_SGAN_Time_Phase:

    """
       Class to retrain SGAN using Time-Phase Data.

                                                 """
    def __init__(self, data, labels, validation_data, validation_labels, unlabelled_data, unlabelled_labels, batch_size, bin_size=48):
        self.data = data
        self.labels = labels
        self.validation_data = validation_data
        self.validation_labels = validation_labels
        self.unlabelled_data = unlabelled_data
        self.unlabelled_labels = unlabelled_labels
        self.bin_size = bin_size
        self.batch_size = batch_size

    def custom_activation(self, output):
        logexpsum = backend.sum(backend.exp(output), axis=-1, keepdims=True)
        result = logexpsum / (logexpsum + 1.0)
        return result

    def define_discriminator(self, n_classes=2):
        in_shape = (self.bin_size, self.bin_size, 1)
        in_image = Input(shape=in_shape)
        model = Conv2D(32, (7, 7), padding='same')(in_image)
        model = LeakyReLU(alpha=0.1)(model)
        model = MaxPooling2D((2, 2),padding='same')(model)
        model = Dropout(0.25)(model)
        model = Conv2D(64, (7, 7), padding='same')(model)
        model = LeakyReLU(alpha=0.1)(model)
        model = MaxPooling2D(pool_size=(2, 2),padding='same')(model)
        model = Dropout(0.25)(model)
        model = Conv2D(128, (7, 7), padding='same')(model)
        model = LeakyReLU(alpha=0.1)(model)
        model = MaxPooling2D(pool_size=(2, 2),padding='same')(model)
        model = Dropout(0.4)(model)
        model = Conv2D(256, (7, 7), padding='same')(model)
        model = LeakyReLU(alpha=0.1)(model)
        model = MaxPooling2D(pool_size=(2, 2),padding='same')(model)
        model = Dropout(0.4)(model)
        model = Flatten()(model)
        fe = Dense(n_classes)(model)
        ''' supervised output '''
        c_out_layer = Activation('softmax')(fe)
        ''' define and compile supervised discriminator model '''
        c_model = Model(in_image, c_out_layer)
        c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
        ''' unsupervised output '''
        d_out_layer = Lambda(self.custom_activation)(fe)
        ''' define and compile unsupervised discriminator model '''
        d_model = Model(in_image, d_out_layer)
        d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
        return d_model, c_model


    def define_generator(self, latent_dim=100):
        init = RandomNormal(mean=0.0, stddev=0.02)
        ''' image generator input '''
        in_lat = Input(shape=(latent_dim,))
        ''' foundation for 7x7 image '''
        n_nodes = 128 * 12 * 12
        gen = Dense(n_nodes, use_bias=False, kernel_initializer=init)(in_lat)
        gen = BatchNormalization()(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Reshape((12, 12, 128))(gen)

        ''' upsample to 24x24 '''
        gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', use_bias=False, kernel_initializer=init)(gen)
        gen = BatchNormalization()(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        ''' upsample to 48x48 '''
        gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
        gen = BatchNormalization()(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        ''' output '''
        out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
        ''' define model '''
        model = Model(in_lat, out_layer)
        return model


    # define the combined generator and discriminator model, for updating the generator
    def define_gan(self, g_model, d_model):
        # make weights in the discriminator not trainable
        d_model.trainable = False
        # connect image output from generator as input to discriminator
        gan_output = d_model(g_model.output)
        # define gan model as taking noise and outputting a classification
        model = Model(g_model.input, gan_output)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    ## select a supervised subset of the dataset, ensures classes are balanced
    def select_supervised_samples(self, dataset, attempt_no, n_samples=10000, n_classes=2):
        X, y = dataset
        X_list, y_list = list(), list()
        n_per_class = int(n_samples / n_classes)
        for i in range(n_classes):
            # get all images for this class
            X_with_class = X[y == i]
            print('Number of examples with label %d is %d' %(i, len(X_with_class)))
            np.random.seed(attempt_no)
            # choose random instances
            ix = np.random.randint(0, len(X_with_class), n_per_class)
            # add to list
            [X_list.append(X_with_class[j]) for j in ix]
            [y_list.append(i) for j in ix]
        return np.asarray(X_list), np.asarray(y_list)


    # select real samples

    def generate_real_samples(self, dataset, n_samples, noisy_labels=True):
        # split into images and labels
        images, labels = dataset
        # choose random instances
        ix = np.random.randint(0, images.shape[0], n_samples)
        # select images and labels
        X, labels = images[ix], labels[ix]
        # generate class labels
        if noisy_labels:
            y = np.random.uniform(0.7,1.2, (n_samples, 1))
        else:
            y = np.ones((n_samples, 1))
        
        #Ocasionally flip labels
        shuffler = np.random.uniform(0,1)
        if shuffler <= 0.05:
            y = np.random.uniform(0.0,0.3, (n_samples, 1))

        return [X, labels], y
    # generate points in latent space as input for the generator
    def generate_latent_points(self, latent_dim, n_samples):
        # generate points in the latent space
        z_input = np.random.randn(latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        z_input = z_input.reshape(n_samples, latent_dim)
        return z_input


    # use the generator to generate n fake examples, with class labels
    def generate_fake_samples(self, generator, latent_dim, n_samples, noisy_labels=True):
        # generate points in latent space
        z_input = self.generate_latent_points(latent_dim, n_samples)
        # predict outputs
        images = generator.predict(z_input)
        # create class labels
        if noisy_labels:
            y = np.random.uniform(0.0,0.3, (n_samples, 1))
        else:
            y = np.zeros((n_samples, 1))

        #Ocasionally flip labels
        shuffler = np.random.uniform(0,1)
        if shuffler <= 0.05:
            y = np.random.uniform(0.7,1.2, (n_samples, 1))

        return images, y



    def summarize_performance(self, step, g_model, c_model, latent_dim, dataset, validation_dataset, epoch_number, model_accuracy, save_best_model=True):
        ''' June 2020 update for this function. 

              1. After careful thought, this function now saves the best model based on the performance of validation data labels. This is the best way to have a fair comparison 
                 between semi-supervised & supervised algorithms.
 
              2. This is also important to do if we compare these results to another algorithm say 'Random forest'. DO NOT choose your best model based on results from the 
                 test data. That leads to an over-estimation of your model performance. Use the test dataset only in the end to compare all your models. 
        '''

        ''' Comment out this block of code below in case you want to view the plots the generator makes after every epoch'''
        # prepare fake examples
        #n_samples = 100
        #X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
        ## scale from [-1,1] to [0,1]
        #X = (X + 1) / 2.0
        ## plot images
        #for i in range(n_samples):
        #    # define subplot
        #    pyplot.subplot(10, 10, 1 + i)
        #    pyplot.axis('off')
        #    pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
        #    pyplot.plot(X[i])
        #pyplot.close()
             ##  save plot to file
        #filename1 = '/fred/oz002/vishnu/pulsar_candidate_demystifier/codes/semi_supervised_trained_models/time_phase_generated_plot_%04d_labelled_%d_unlabelled_%d_trial_%d.png' % (step+1, labelled_samples, unlabelled_samples, attempt_no)
        #pyplot.savefig(filename1)
        
        validation_X, validation_y = validation_dataset
        _, acc = c_model.evaluate(validation_X, validation_y, verbose=0)
        with open('training_logs/model_performance_sgan_time_phase.txt', 'a') as f:
            f.write('intermediate_models/time_phase_c_model_epoch_%d.h5' % int(epoch_number) + ',' + '%.3f' % (acc) + '\n')

        if save_best_model == True:
            if acc > model_accuracy:
                print('Current Model has %.3f training accuracy which is better than previous best of %.3f. Will save it as as new best model.' % (acc * 100, model_accuracy * 100 ))
                filename2 = 'best_retrained_models/time_phase_best_generator_model.h5'  
                g_model.save(filename2)
                filename3 = 'best_retrained_models/time_phase_best_discriminator_model.h5'
                c_model.save(filename3)
                model_accuracy = acc

            else:
                print('Current Model is not as good as the best model. This model will not be saved.')

            return model_accuracy
        else:
            print('Classifier Accuracy: %.3f%%' % (acc * 100))
            # save the generator model
            filename2 = 'intermediate_models/time_phase_g_model_epoch_%d.h5' %int(epoch_number)
            g_model.save(filename2)
            # save the classifier model
            filename3 = 'intermediate_models/time_phase_c_model_epoch_%d.h5' %int(epoch_number)
            c_model.save(filename3)
            print('>Saved: %s, and %s' % (filename2, filename3))
            return model_accuracy

    ## train the generator and discriminator
    def train(self, g_model, d_model, c_model, gan_model, latent_dim=100, n_epochs=10):

        ''' Define datasets. unlabelled_labels are all equal to -1. These labels are not used during training.'''

        dataset = [self.data, self.labels]
        unlabelled_dataset = [self.unlabelled_data, self.unlabelled_labels] 
        validation_dataset = [self.validation_data, self.validation_labels]
        n_batch = self.batch_size 
        X_sup, y_sup = dataset
        epoch_number = 0
        model_accuracy = 0.
        # calculate the number of batches per training epoch
        bat_per_epo = int((dataset[0].shape[0] + unlabelled_dataset[0].shape[0]) / n_batch)

        print('batch per epoch is %d' % bat_per_epo)
        # calculate the number of training iterations
        n_steps = bat_per_epo * n_epochs
        # calculate the size of half a batch of samples
        half_batch = int(n_batch / 2)
        print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (n_epochs, n_batch, half_batch, bat_per_epo, n_steps))
        # manually enumerate epochs
        for i in range(n_steps):
            ''' update supervised discriminator (c) '''
            [Xsup_real, ysup_real], _ = self.generate_real_samples([X_sup, y_sup], half_batch)
            c_loss, c_acc = c_model.train_on_batch(Xsup_real, ysup_real)
            ''' update unsupervised discriminator (d) '''
            [X_real, _], y_real = self.generate_real_samples(unlabelled_dataset, half_batch)
            d_loss1 = d_model.train_on_batch(X_real, y_real)
            X_fake, y_fake = self.generate_fake_samples(g_model, latent_dim, half_batch)
            
            d_loss2 = d_model.train_on_batch(X_fake, y_fake)
            ''' update generator (g) '''
            X_gan, y_gan = self.generate_latent_points(latent_dim, n_batch), np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            #summarize loss on this batch
            print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (i+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss))
            #print('>%d, c[%.3f,%.0f]' % (i+1, c_loss, c_acc*100))
            # evaluate the model performance every so often
            if (i+1) % (bat_per_epo * 1) == 0:
                epoch_number+=1
                model_accuracy = self.summarize_performance(i, g_model, c_model, latent_dim, dataset, validation_dataset, epoch_number, model_accuracy)



