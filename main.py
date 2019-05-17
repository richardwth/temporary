import tensorflow as tf
from tensorflow import keras
import sys
sys.path.insert(0, '/home/richard/PycharmProjects/myNN')

print('TensorFlow version {}'.format(tf.__version__))
print('Keras version {}'.format(keras.__version__))

"""
This cell reads the train/vali/test tfrecords, and visualize one sample from
the batch.

If you want to run this code, 
1. copy this cell to a .py file and add a line to change sys.path
2. change FLAGS.DEFAULT_IN to:
/data/cephfs/punim0811/Datasets/iNaturalist/tfrecords_299/
or your local machine address accordingly if you decide to download 
some tfrecords file.
3. Note that on Spartan, im.show() would not work. However, you may save the 
example you want to visualize, e.g., im.save('test_image.jpg', 'JPEG')
"""
from GeneralTools.misc_fun import FLAGS
FLAGS.DEFAULT_IN = '/media/richard/ExtraStorage/Data/inaturalist_NHWC_299/'
FLAGS.IMAGE_FORMAT = 'channels_last'
FLAGS.IMAGE_FORMAT_ALIAS = 'NHWC'
from GeneralTools.inaturalist_func import ReadTFRecords
import os
import tensorflow as tf
from PIL import Image
import numpy as np

batch_size = 64
target_size = 299
key = 'train'
data_size = {'train': 265213, 'val': 3030, 'test': 35350}
data_label = {'train': 1, 'val': 1, 'test': 0}
num_images = data_size[key]
steps_per_epoch = num_images // batch_size
skip_count = num_images % batch_size
num_labels = data_label[key]
num_classes = 1010

filenames = os.listdir(FLAGS.DEFAULT_IN)
filenames = [filename.replace('.tfrecords', '') for filename in filenames if key in filename]
print(filenames)

dataset = ReadTFRecords(
    filenames, num_labels=num_labels, batch_size=batch_size, buffer_size=2000,
    skip_count=skip_count, num_threads=8, decode_jpeg=True,
    use_one_hot_label=True, num_classes=num_classes)
dataset.shape2image(3, target_size, target_size)
dataset.scheduler()
# data_batch = dataset.next_batch()

# with tf.Session() as sess:
#     if key == 'test':
#         x = sess.run(data_batch['x'])
#     else:
#         x, y = sess.run([data_batch['x'], data_batch['y']])

# visualize one sample from the batch
# x_im = x[0] * 255
# im = Image.fromarray(x_im.astype(np.uint8), 'RGB')
# im.show()

# if key in {'train', 'val'}:
#     print(y.shape)
#     print(y.dtype)

"""
This cell train a pre-trained model with extra layers

"""
# keras pretrained inception v3 model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, applications
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from tensorflow.keras import backend as K

# load the model
base_model = applications.InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(target_size, target_size, 3))
# Freeze some layers
for layer in base_model.layers[:-20]:
    layer.trainable = False
# Adding custom layers
# x = model.output
# x = Flatten()(x)
# x = Dense(1024, activation='relu')(x)
# x = Dropout(0.5)(x)
# predictions = Dense(num_classes, activation='linear')(x)
# mdl = Model(input = model.input, output = predictions)
mdl = Sequential([
    base_model, Flatten(), Dense(1024, activation='relu'),
    Dense(num_classes, activation='linear')])

mdl.compile(
    tf.keras.optimizers.Adam(lr=0.001),
    loss=tf.losses.softmax_cross_entropy, metrics=['accuracy'])
history = mdl.fit(
    dataset.dataset, epochs=1, callbacks=None, steps_per_epoch=20, verbose=1)
# print(mdl.summary())
