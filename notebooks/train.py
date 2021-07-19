import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os
import sys
import json
from PIL import Image
from sklearn.model_selection import train_test_split
from keras_unet.utils import get_augmented
from keras_unet.utils import plot_imgs
from keras_unet.models import custom_unet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.losses import jaccard_distance
from keras_unet.utils import plot_segm_history
import tensorflow as tf
import datetime
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# config.gpu_options.allow_growth = True
# tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

data_resize = 800
# Load Data
MODEL_PATH = './model_list_unet'
masks_01 = glob.glob("../../../data/facility/dam_crack_2/train/croppedgt/*.png")
orgs_01 = glob.glob("../../../data/facility/dam_crack_2/train/croppedimg/*.jpg")

masks_02 = glob.glob("../../../data/facility/dam_crack_2/val/croppedgt/*.png")
orgs_02 = glob.glob("../../../data/facility/dam_crack_2/val/croppedimg/*.jpg")

masks = masks_01 + masks_02
orgs = orgs_01 + orgs_02
# orgs = list(map(lambda x: x.replace(".png", ".jpg"), masks))

imgs_list = []
masks_list = []

for image, mask in zip(orgs, masks):
    imgs_list.append(np.array(Image.open(image).resize((data_resize,data_resize))))
    masks_list.append(np.array(Image.open(mask).resize((data_resize,data_resize))))
    # imgs_list.append(np.array(Image.open(image).resize((data_resize,data_resize))))
    # masks_list.append(np.array(Image.open(mask).resize((data_resize,data_resize))))

imgs_np = np.asarray(imgs_list)
masks_np = np.asarray(masks_list)

# Get data into correct shape, dtype and range (0.0-1.0)
x = np.asarray(imgs_np, dtype=np.float32)/255
y = np.asarray(masks_np, dtype=np.float32)/255
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)

# Train/Val Split
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)

# Prepare train generator with data augmentation

train_gen = get_augmented(
    x_train, y_train, batch_size=4,
    data_gen_args = dict(
        rotation_range=5.,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=40,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='constant'
    ))
    
sample_batch = next(train_gen)
xx, yy = sample_batch

input_shape = x_train[0].shape

model = custom_unet(
    input_shape,
    filters=32,
    use_batch_norm=True,
    dropout=0.3,
    dropout_change_per_layer=0.0,
    num_layers=4)

# Compile and Train
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
model_filename = 'seg_model_v3'
callback_checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_PATH, model_filename + '_epoch:{epoch:02d}_val_loss:{val_loss:.4f}_iou:{iou:.4f}.h5'),
    verbose=1, 
    monitor='val_loss', 
    save_best_only=False,)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model.compile(
    optimizer=Adam(), 
    #optimizer=SGD(lr=0.01, momentum=0.99),
    loss='binary_crossentropy',
    #loss=jaccard_distance,
    metrics=[iou, iou_thresholded])

history = model.fit_generator(
    train_gen,
    steps_per_epoch=100,
    epochs=150,
    
    validation_data=(x_val, y_val),
    callbacks=[callback_checkpoint, tensorboard_callback])

with open(MODEL_PATH + "/tensor_board.json", 'w') as f:
    json.dump(history.history, f)
    
plot_segm_history(history, metrics=['iou', 'val_iou'], losses=['loss', 'val_loss'])
plt.savefig(MODEL_PATH + "/tensor_board.png", dpi = 300)