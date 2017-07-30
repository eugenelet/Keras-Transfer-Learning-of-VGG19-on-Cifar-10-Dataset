import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.optimizers import SGD
import cv2, numpy as np
from keras.datasets import cifar10
from keras.layers.normalization import BatchNormalization

BATCH_NORM = False

tbCallBack = keras.callbacks.TensorBoard(log_dir='./tensorflow/notrans-BN', histogram_freq=0, 
                            write_graph=True, write_images=False)
batch_size = 128
num_classes = 10
epochs = 164
data_augmentation = True

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#model = keras.applications.vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)


model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3), name='block1_conv1'))
model.add(BatchNormalization()) if BATCH_NORM else None
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3), padding='same', name='block1_conv2'))
model.add(BatchNormalization()) if BATCH_NORM else None
model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

model.add(Conv2D(128, (3, 3), padding='same', name='block2_conv1'))
model.add(BatchNormalization()) if BATCH_NORM else None
model.add(Activation('relu'))

model.add(Conv2D(128, (3, 3), padding='same', name='block2_conv2'))
model.add(BatchNormalization()) if BATCH_NORM else None
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))


model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv1'))
model.add(BatchNormalization()) if BATCH_NORM else None
model.add(Activation('relu'))

model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv2'))
model.add(BatchNormalization()) if BATCH_NORM else None
model.add(Activation('relu'))

model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv3'))
model.add(BatchNormalization()) if BATCH_NORM else None
model.add(Activation('relu'))

model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv4'))
model.add(BatchNormalization()) if BATCH_NORM else None
model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv1'))
model.add(BatchNormalization()) if BATCH_NORM else None
model.add(Activation('relu'))

model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv2'))
model.add(BatchNormalization()) if BATCH_NORM else None
model.add(Activation('relu'))

model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv3'))
model.add(BatchNormalization()) if BATCH_NORM else None
model.add(Activation('relu'))

model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv4'))
model.add(BatchNormalization()) if BATCH_NORM else None
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv1'))
model.add(BatchNormalization()) if BATCH_NORM else None
model.add(Activation('relu'))

model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv2'))
model.add(BatchNormalization()) if BATCH_NORM else None
model.add(Activation('relu'))

model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv3'))
model.add(BatchNormalization()) if BATCH_NORM else None
model.add(Activation('relu'))

model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv4'))
model.add(BatchNormalization()) if BATCH_NORM else None
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(4096))
model.add(BatchNormalization()) if BATCH_NORM else None
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(4096, name='fc2'))
model.add(BatchNormalization()) if BATCH_NORM else None
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(BatchNormalization()) if BATCH_NORM else None
model.add(Activation('softmax'))


# model.load_weights('vgg19_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)

'''
for layer in model.layers[:35]:
    layer.trainable = False
#model.layers[39].trainable = False
'''
# initiate RMSprop optimizer
opt = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0001, nesterov=True)
#opt = keras.optimizers.Adam(lr=0.001, decay=0.0001)

model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])#

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train[:,:,:,[0,1,2]] = x_train[:,:,:,[2,1,0]] 
x_train[:,:,:,0] -= 103.939
x_train[:,:,:,1] -= 116.779
x_train[:,:,:,2] -= 123.68
x_test[:,:,:,[0,1,2]] = x_test[:,:,:,[2,1,0]] 
x_test[:,:,:,0] -= 103.939
x_test[:,:,:,1] -= 116.779
x_test[:,:,:,2] -= 123.68

datagen = ImageDataGenerator(width_shift_range=0.125,  # randomly shift images horizontally (fraction of total width)
                             height_shift_range=0.125,  # randomly shift images vertically (fraction of total height)
                             horizontal_flip=True)

model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size+1,
                        epochs=epochs,callbacks=[tbCallBack],
                        validation_data=(x_test, y_test))

model.save_weights('vgg19_cifar10_retrain_weight_BN.h5')

