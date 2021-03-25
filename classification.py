'''
This is the net model of Environmental features recognition for lower limb prostheses toward predictive walking.
If you think this code is useful, please cite:
[1] K. Zhang, C. Xiong, W. Zhang, H. Liu, D. Lai, Y. Rong, and C. Fu, 
“Environmental features recognition for lower limb prostheses toward predictive walking,” 
IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 27, no. 3, pp. 465–476, Mar. 2019.

'''


from __future__ import print_function
import keras
import glob
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from zipfile import ZipFile



def load_data(path = 'data/', num_classes = 6, image_shape = (100, 100, 1)):
    file_vec = glob.glob(path + '/*/*.png')
    if 0 == len(file_vec):
        with ZipFile('data.zip', 'r') as zipObj:
           # Extract all the contents of zip file in current directory
           zipObj.extractall()
        file_vec = glob.glob(path + '/*/*.png')
    file_num = len(file_vec)
    X = np.zeros((file_num,) + image_shape)
    y = np.zeros(file_num)
    idx = 0
    for n in range(num_classes):
        for file in glob.glob(path + str(n+1) + '/*.png'):
            img = cv2.imread(file, -1)
            img = np.reshape(img, image_shape)
            X[idx,...] = img
            y[idx] = n
            idx += 1
    y = keras.utils.to_categorical(y, num_classes=num_classes)
    print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=1)
    return (X_train, y_train), (X_test, y_test)


def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

def calc_net_size():
    sess = K.get_session()
    graph = sess.graph
    stats_graph(graph)


batch_size = 128
num_classes = 5
epochs = 30

image_shape = (100, 100, 1)

# input image dimensions
img_rows, img_cols = image_shape[0], image_shape[1]

(x_train, y_train), (x_test, y_test) = load_data(image_shape = image_shape, 
                                                num_classes=num_classes)



if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# deep CNN
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


model_path = 'checkpoint/best_model.h5'
checkpoint = ModelCheckpoint(model_path, 
                             verbose=1, monitor='val_acc',
                             save_best_only=True, mode='auto')

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

is_train = True
if is_train:
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split=0.25, callbacks=[checkpoint])

# load the best model
# model.load_weights(model_path)
# calc_net_size()

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

