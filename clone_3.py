import cv2
import csv
import numpy as np
import sklearn
import os
import random

#empty array to store the data read from the csv file
lines=[]
#loading data from the csv file:
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line) #this creates a single list with all the data points appended

from sklearn.model_selection import train_test_split
train_samples, val_samples = train_test_split(lines[1:], test_size=0.2)

#creates empty list to store images and other measurements (this includes steering)

def generator(samples, batch_size=32, steering_correction=0.35):
    num_samples = len(samples)
    while 1:
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                #load in original images paths
                center_path = 'data/IMG/' + batch_sample[0].split('/')[-1]
                left_path = 'data/IMG/' + batch_sample[1].split('/')[-1]
                right_path = 'data/IMG/' + batch_sample[2].split('/')[-1]
                #read images
                center_image = cv2.imread(center_path)
                left_image = cv2.imread(left_path)
                right_image = cv2.imread(right_path)
                #create augmented images for the three images
                aug_center_image = cv2.flip(center_image,1)
                aug_left_image = cv2.flip(left_image,1)
                aug_right_image = cv2.flip(right_image,1)
                #load in measurements
                measurement_center = float(batch_sample[3])
                measurement_left = measurement_center + steering_correction
                measurement_right = measurement_center - steering_correction
                aug_measurement_center = measurement_center*-1.0
                aug_measurement_left = measurement_left*-1.0
                aug_measurement_right = measurement_right*-1.0
                #populating the empty array for images and measurements
                images.extend([center_image,left_image,right_image,aug_center_image,aug_left_image,aug_right_image])
                measurements.extend([measurement_center,measurement_left,measurement_right,aug_measurement_center,aug_measurement_left,aug_measurement_right])
            X_train = np.array(images)
            y_train = np.array(measurements)
            #spitting out one unit
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples)
val_generator = generator(val_samples)


#building the model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

model = Sequential()
#mdoel normalization and mean centering using Lambda
model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(160,320,3)))
#adding cropping layers
model.add(Cropping2D(cropping=((70,25),(0,0))))
#adding several convolutional layers
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(46,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))

#flattening output, does not affect batch size
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data = val_generator, nb_val_samples=len(val_samples), nb_epoch=7)

model.save('model_2.h5')
