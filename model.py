import csv
import cv2
import numpy as np
import os
import sklearn
from sklearn.preprocessing import LabelBinarizer
from zipfile import ZipFile

# extract data from Data.zip if not already there.
if(not os.path.isdir('data')):
   with ZipFile('Data.zip') as zipf:
        zipf.extractall('data')

samples = []
# read the data from driving log csv file. append it to samples list.
with open('./data/Data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        samples.append(line)

        
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# split train and validation data as 80% and 20% respectively.
train_samples, validation_samples = train_test_split(samples,test_size=0.2)


#since the data size is in GBs we need to use generators.
def generator(samples, batch_size=32):
    num_samples = len(samples)
   
    while 1: # loop forever so that generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                # there are three cameras, center, left and right.
                for i in range(0,3):
                    # file name
                    name = './data/Data/IMG/'+ batch_sample[i].split('\\')[-1]
                    center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                    center_angle = float(batch_sample[3])
                    images.append(center_image)
                    
                    if(i==0):
                        angles.append(center_angle)
                    elif(i==1):
                        angles.append(center_angle+0.2) # left angle
                    elif(i==2):
                        angles.append(center_angle-0.2) # right angle
                        
                    # Traing track is counter clockwise loop, so the data is baised towards left turns.
                    # To solve this we can have the data for counter clock and clock wise direction.
                    # and/or we can flip the images and take negative of steering angle.
                    images.append(cv2.flip(center_image,1))
                    if(i==0):
                        angles.append(center_angle*(-1))
                    elif(i==1):
                        angles.append((center_angle+0.2)*(-1))
                    elif(i==2):
                        angles.append((center_angle-0.2)*(-1))
                          
            # keras requires numpy format
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield sklearn.utils.shuffle(X_train, y_train)
            
# Set our batch size
batch_size=32
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)




from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda, Cropping2D

model = Sequential()

# Preprocess:- Nomalize the data by dividing max (225) 
# and mean center the data by subtracting 0.5.
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# trim image to only see section with road
model.add(Cropping2D(cropping=((70,25),(0,0))))           

#layer 1- Convolution, no of filters- 24, filter size= 5x5, stride= 2x2
model.add(Convolution2D(24,5,5,subsample=(2,2), activation="elu"))

#layer 2- Convolution, no of filters- 36, filter size= 5x5, stride= 2x2
model.add(Convolution2D(36,5,5,subsample=(2,2), activation="elu"))

#layer 3- Convolution, no of filters- 48, filter size= 5x5, stride= 2x2
model.add(Convolution2D(48,5,5,subsample=(2,2), activation="elu"))

#layer 4- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
model.add(Convolution2D(64,3,3, activation="elu"))

#layer 5- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
model.add(Convolution2D(64,3,3, activation="elu"))

model.add(Dropout(0.8))

#flatten image from 2D to side by side
model.add(Flatten())

#layer 6- fully connected layer 1
model.add(Dense(100, activation="elu"))

#layer 7- fully connected layer 1
model.add(Dense(50, activation="elu"))

#layer 8- fully connected layer 1
model.add(Dense(10, activation="elu"))

#layer 9- fully connected layer 1
model.add(Dense(1, activation="elu")) #here the final layer will contain one value as this is a regression problem and not classification


#now compile the model using mean squeared error(MSE), since its a regression network.
# we are minimizing the difference between steering angle that the network produces and ground trouth. 
model.compile(loss='mse',optimizer='adam')


from math import ceil
model.fit_generator(train_generator, steps_per_epoch=ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=ceil(len(validation_samples)/batch_size), epochs=5, verbose=1)

#In the end save the model
model.save('model.h5')

# how to run it
# python drive.py model.h5