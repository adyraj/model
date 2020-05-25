# importing required libraries

from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

model = Sequential()


# Add Layers

import random

model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
model.add(MaxPooling2D(pool_size=(2, 2)))

def architecture(option):
    if option == 1:
        model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
    elif option == 2:
        model.add(Convolution2D(filters=64, 
                        kernel_size=(2,2), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        
    elif option == 3:
        #two convolutional and 2 max pooling layers
        model.add(Convolution2D(filters=64, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    elif option == 4:
        model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
        model.add(Convolution2D(filters=64, 
                        kernel_size=(2,2), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
        model.add(MaxPooling2D(pool_size=(3, 3)))
    
    else:
        model.add(Convolution2D(filters=128, 
                        kernel_size=(2,2), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
        model.add(MaxPooling2D(pool_size=(3, 3)))        
        model.add(Convolution2D(filters=64, 
                        kernel_size=(2,2), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
        model.add(MaxPooling2D(pool_size=(3, 3)))        
architecture(random.randint(1,4))

model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())

def fullyconnected(option):
    if option == 1:
        model.add(Dense(units=128, activation='relu'))
    elif option == 2:
        model.add(Dense(units=128, activation='sigmoid'))
        model.add(Dense(units=64, activation='softmax'))
    elif option == 3:
        model.add(Dense(units=128, activation='sigmoid'))
        model.add(Dense(units=64, activation='softmax'))
        model.add(Dense(units=32, activation='relu'))
    elif option == 4:
        model.add(Dense(units=128, activation='softmax'))
        model.add(Dense(units=128, activation='sigmoid'))
        model.add(Dense(units=64, activation='softmax'))
        model.add(Dense(units=32, activation='relu'))
        
    else:
        model.add(Dense(units=128, activation='softmax'))
        model.add(Dense(units=128, activation='softmax'))
        model.add(Dense(units=64, activation='softmax'))
        model.add(Dense(units=64, activation='sigmoid'))
        model.add(Dense(units=32, activation='relu'))

fullyconnected(random.randint(1,5))

model.add(Dense(units=32, activation='relu'))

model.add(Dense(units=1,activation='sigmoid'))

print(model.summary())


# Compile the Model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'cnn_dataset/training_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'cnn_dataset/test_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
output = model.fit(training_set,
                steps_per_epoch=8000,
                epochs=1,
                validation_data=test_set,
                validation_steps=800)
output.history

print("Accuracy : ", output.history['accuracy'][0])
model =str(model.layers)
accuracy = str(output.history['accuracy'][0])

# Save accuracy and architecture in a file
print(accuracy, file=open("accuracy.txt", "a"))
print(model, file=open("model.txt", "a"))





