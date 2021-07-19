import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os
import sys
from PIL import Image
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras_unet.utils import get_augmented
from keras_unet.utils import plot_imgs
from keras_unet.models import custom_unet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.losses import jaccard_distance
from keras_unet.utils import plot_segm_history
import skimage
from skimage import io
import cv2

# Load Data
masks_test = glob.glob("../input/vh/test/*.png")
weights_path = os.path.join("../notebooks", "seg_model_v3_vh.h5")

orgs_test = list(map(lambda x: x.replace(".png", ".jpg"), masks_test))


if not(os.path.isdir("../input/vh" + '/result')):
    os.makedirs(os.path.join("../input/vh", 'result'))

save_dir = []
for i in range(len(masks_test)):
    save_ids = masks_test[i].replace('test','result')
    save_dir.append(save_ids)

imgs_test_list = []
masks_test_list = []

for image, mask in zip(orgs_test, masks_test):
    imgs_test_list.append(np.array(Image.open(image).resize((384,384))))
    masks_test_list.append(np.array(Image.open(mask).resize((384,384))))
    
imgs_test_np = np.asarray(imgs_test_list)
masks_test_np = np.asarray(masks_test_list)
print(imgs_test_np.shape, masks_test_np.shape)

# Get data into correct shape, dtype and range (0.0-1.0)

print(imgs_test_np.max(), masks_test_np.max())
x_test = np.asarray(imgs_test_np, dtype=np.float32)/255
y_test = np.asarray(masks_test_np, dtype=np.float32)

print(x_test.max(), y_test.max())
print(x_test.shape, y_test.shape)

y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2], 1)
print(x_test.shape, y_test.shape)

model = load_model(weights_path, custom_objects={"iou":iou,"iou_thresholded":iou_thresholded})
model.load_weights(weights_path)
y_pred = model.predict(x_test)


print(save_dir)

# Change prediction value float 32 to uint8 

for i in range(y_pred.shape[0]):
    for j in range(y_pred[i].shape[0]):
        for k in range(y_pred[i].shape[1]):
            if y_pred[i,j,k] > 0.5:
                y_pred[i,j,k] = 255
                #y_pred[i,j,k] = y_pred[i,j,k].astype(np.uint8)
            else:
                y_pred[i,j,k] = 0
                #y_pred[i,j,k] = y_pred[i,j,k].astype(np.uint8)
                
y_pred = y_pred.astype(np.uint8)

# save mask image 

for i in range(y_pred.shape[0]):
    mask_ = y_pred[i,:,:,0]
    io.imsave(save_dir[i], mask_)
    #cv2.imwrite(result_dir + "/result/%s" %filename,mask_)
    
# vectorizing through contour method and calculate mask area

for j in range(y_pred.shape[0]):
    img =y_pred[j].astype(np.uint8)
    # imgray = cv2.cvtColor(img,cv2.COLOR_GRAY2GRAY)
    ret, thresh = cv2.threshold(img[:,:,0],127,255,0)
    image, contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    area_list = []
    for i in range(len(contours)):
        if len(contours[i]) > 1 and cv2.contourArea(contours[i]) > 200:
            area_list.append(cv2.contourArea(contours[i]))
            x0, y0 = zip(*np.squeeze(contours[i]))
            plt.plot(x0, y0, c="b")
            # io.imsave(save_dir[i], [x0, y0])

    print(area_list)
    plt.imshow(imgs_test_np[j,:,:])

    # plt.imshow(img[:,:,0])
    plt.xlim(0, 384)
    plt.ylim(384, 0)
    # plt.show()
    plt.savefig(save_dir[j].split('.pn')[0] + '.jpg', dpi = 300)
    plt.clf()
