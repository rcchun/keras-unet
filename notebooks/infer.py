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
from functools import reduce
import csv
# from libtiff import TIFF
# import shapefile
# from pyproj import Proj, transform
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

# calculate f1 score, precision, recall, iou, TP, FN, FP
def union_intersect(true, pred,threshold=100):
    # Predict matrix, GT matrix vectorize for Intersection 1d , Union 1d, setDiff 1d Calculation
    h,w = true.shape
    nflat=true.ravel().shape

    pred = pred.copy()
    true = true.copy()

    pred=pred.astype(int)
    true=true.astype(int)

    pred[pred<threshold]=0
    pred[pred>=threshold]=255
    true_ravel = true.ravel()
    pred_ravel = pred.ravel()

    # Find index 255. or 1. region
    true_ind = np.where(true_ravel == 1)
    pred_ind = np.where(pred_ravel == 255)

    # Intersection , Union , Diff Calculation
    TP_ind = np.intersect1d(true_ind, pred_ind)
    FN_ind = np.setdiff1d(true_ind, TP_ind)
    FP_ind = np.setdiff1d(pred_ind,TP_ind)
    union_ind = reduce(np.union1d,(TP_ind, FN_ind, FP_ind))

    # Intersection of Union(HED,GT)


    TP_count = TP_ind.shape[0]
    union_count=union_ind.shape[0]
    pred_count = pred_ind[0].shape[0]
    true_count = true_ind[0].shape[0]

    precision = 0
    iou = 0
    recall =0
    f1 = 0
    print('THRES({}) - TP : {}, UNION : {}, PRED : {}, TRUE : {}'.format(threshold, TP_count, union_count,pred_count, true_count))
    if TP_count==0 or pred_count==0 or true_count==0 or union_count==0:
        pass

    else :
        iou= TP_count / union_count
        precision = TP_count / pred_count
        recall = TP_count / true_count
        print(precision,recall)

        f1 = 2 * (precision * recall) / (precision + recall)

    # Create dummy array
    union = np.zeros(nflat)
    TP = np.zeros(nflat)
    FN = np.zeros(nflat)
    FP = np.zeros(nflat)

    # Write Array
    union[union_ind]=255
    TP[TP_ind]=255
    FN[FN_ind]=255
    FP[FP_ind]=255

    # return 2d arrays and iou
    return np.reshape(union,true.shape), np.reshape(TP,true.shape),np.reshape(FP,true.shape),np.reshape(FN,true.shape),precision,recall,iou ,f1

# slicing image
def get_patch(img_arr, size, stride):
    
    patches_list = []
    overlapping = 0

    ind_1 = image.shape[0] // size
    ind_2 = image.shape[1] // size
    k = 0
    l = 0
    for i in range(ind_1+1):
        
        for j in range(ind_2+1):
            if (i == ind_1) and (j != ind_2):
                
                pat = 255 * np.ones((size-(image.shape[0] - ind_1*size), size, 3))
                pat = np.asarray(pat, dtype=np.uint8)
                img_ = np.concatenate((image[ind_1 * size : image.shape[0], k * size : k * size +stride], pat), axis=0)
                patches_list.append(img_)
                k +=1

            elif (j == ind_2) and (i != ind_1):
        
                pat2 = 255 * np.ones((size, size-(image.shape[1] - ind_2*size), 3))
                pat2 = np.asarray(pat2, dtype=np.uint8)
                img_2 = np.concatenate((image[l * size : l * size +stride, ind_2 * size : image.shape[1]], pat2), axis =1)
                patches_list.append(img_2)
                l +=1

            elif (i == ind_1) and (j == ind_2):
                
                pat3 = 255 * np.ones((image.shape[0] - ind_1*size, size-(image.shape[1] - ind_2*size), 3))
                pat3 = np.asarray(pat3, dtype=np.uint8)
                img_3 = np.concatenate((image[ind_1 * size : image.shape[0], ind_2 * size : image.shape[1]], pat3), axis =1)
                pat4 = 255 * np.ones((size-(image.shape[0] - ind_1*size), size, 3))
                pat4 = np.asarray(pat4, dtype=np.uint8)
                img_4 = np.concatenate((img_3, pat4), axis =0)
                patches_list.append(img_4)
                
            else:
                patches_list.append(
                    img_arr[
                        i * stride : i * stride + size,
                        j * stride : j * stride + size
                    ]
                )
            
                        
    return np.stack(patches_list), ind_1, ind_2

# transform relative coordinate
def coord_transform(overall_length, ind_1, ind_2):
    index = []
    m = 0
    for i in range(ind_1+1):
        for j in range(ind_2+1):
            if m != overall_length:
                jj = i,j
                index.append(jj)
                m += 1
    return index

def generating_batchlist(quotient, remainder, batch_size):
    batch_list = []
    gen_batch = []
    baselist = range(quotient * batch_size + remainder)
    for i in range(quotient+1):
        batch_list.append(batch_size*i)
    for j in range(len(batch_list)):
        if j == len(batch_list)-1 and remainder != 0:
            gen_batch.append(baselist[batch_list[j]:])
        elif j != len(batch_list)-1 and remainder == 0:
            gen_batch.append(baselist[batch_list[j]:batch_list[j+1]])
    return gen_batch


# download coordinate system (prj file)
def getWKT_PRJ (epsg_code):
    import urllib.request
    wkt = urllib.request.urlopen("http://spatialreference.org/ref/sr-org/{0}/prj/".format(epsg_code))
    remove_spaces = wkt.read().decode("utf-8").replace(" ","")
    output = remove_spaces.replace("\n", "")
    return output

# batch_size
batch_size = 10
# data size
data_size_1 = 384
data_size_2 = 512
data_resize = 800
# Load Data
# weights_path = os.path.join("../notebooks/model_list_unet/0511_v3", "seg_model_v3_epoch:50_iou:0.8629.h5")
# weights_path = os.path.join("../notebooks/model_list_unet", "seg_model_v4_epoch:112_val_loss:0.2706_iou:0.8329.h5")
# weights_path = os.path.join("../notebooks/U-NET", "seg_check_v3130-0.16.h5")
weights_path = os.path.join("./model_list_unet", "seg_model_v3_epoch:148_val_loss:0.0607_iou:0.3625.h5")
# orgs_test = glob.glob("../../../data/facility/v4/test/croppedimg/*.jpg")
# orgs_test = glob.glob("../../../data/facility/BBOX_test/*.jpg")
orgs_test = glob.glob("../../../data/facility/dam_crack_2/test/croppedimg/*.jpg")
mask_test = glob.glob("../../../data/facility/dam_crack_2/test/croppedgt/*.png")
# mask_test = glob.glob("../../../data/facility/v4/test/croppedgt/*.png")
orgs_test.sort()
mask_test.sort()
# print(mask_test)

# reading tiff image and transform numpy array
# tif = TIFF.open(orgs_test, mode='r')
# image = tif.read_image()
# slicing tiff image
# x_crops, ind_1, ind_2 = get_patch(img_arr=image, size=data_resize, stride=data_resize)
# transform relative coordinate
# overall_length = len(x_crops)
# rel_coord = coord_transform(overall_length, ind_1, ind_2)
# print(rel_coord)
# generating result saving folder
# if not(os.path.isdir("../notebooks" + '/result')):
#     os.makedirs(os.path.join("../notebooks", 'result'))


save_dir = []
for i in range(len(orgs_test)):
    save_ids = orgs_test[i].split('/')[8]
    save_dir.append(save_ids)

imgs_test_list = []
# imgs_test_list = x_crops
mask_test_list = []

for image in orgs_test:
    # imgs_test_list.append(np.array(Image.open(image).resize((data_resize,data_resize))))
    imgs_test_list.append(np.array(Image.open(image).resize((data_resize,data_resize))))

for mask in mask_test:
    # mask_test_list.append(np.array(Image.open(mask).resize((data_resize,data_resize))))
    mask_test_list.append(np.array(Image.open(mask).resize((data_resize,data_resize))))
    
imgs_test_np = np.asarray(imgs_test_list)
mask_test_np = np.asarray(mask_test_list)
print(imgs_test_np.shape)
print(mask_test_np.shape)
# Get data into correct shape, dtype and range (0.0-1.0)

# x_test = np.asarray(x_crops, dtype=np.float32)/255
# y_test = np.asarray(mask_test_np, dtype=np.float32)/255
x_test = np.asarray(imgs_test_np, dtype=np.float32)/255
y_test = np.asarray(mask_test_np, dtype=np.float32)/255
# print(x_test.shape)
print(y_test.shape)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2], 1)
# print(x_test.shape, y_test.shape)

# batch list for prediction according to batch size 
# batch_ind, batch_ind_last = divmod(overall_length, batch_size)
# batchlist = generating_batchlist(batch_ind, batch_ind_last, batch_size)
# print(batchlist)

# model load and prediction
model = load_model(weights_path, custom_objects={"iou":iou,"iou_thresholded":iou_thresholded})
model.load_weights(weights_path)
y_pred = model.predict(x_test)

# y_pred = []
# for i in range(len(batchlist)):
#     y_pred_predict = model.predict(x_test[batchlist[i]])
#     y_pred.append(y_pred_predict)

# print(save_dir)
# print(len(y_pred))
# print(y_pred[0].shape)
# Change prediction value float 32 to uint8 
# for m in range(len(y_pred)):
#     for i in range(y_pred[m].shape[0]):
#        for j in range(y_pred[m][i].shape[0]):
#             for k in range(y_pred[m][i].shape[1]):
#                 if y_pred[m][i,j,k] > 0.5:
#                     y_pred[m][i,j,k] = 255
#
#                 else:
#                     y_pred[m][i,j,k] = 0
                
#     y_pred[m] = y_pred[m].astype(np.uint8)
for i in range(y_pred.shape[0]):
    for j in range(y_pred[i].shape[0]):
        for k in range(y_pred[i].shape[1]):
            if y_pred[i, j, k] > 0.5:
                y_pred[i, j, k] = 255

            else:
                y_pred[i, j, k] = 0

y_pred = y_pred.astype(np.uint8)
print(y_pred.shape)
# save mask image / F1 score calculate
f = open(os.path.join("../notebooks/result", 'output.csv'), 'a', encoding='utf-8', newline='')
wr = csv.writer(f)
wr.writerow(['FileName', 'f1', 'iou', 'precision', 'recall'])

for i in range(y_pred.shape[0]):
    mask_ = y_pred[i,:,:,0]
    gt = y_test[i,:,:,0]
    io.imsave('../notebooks/result/' + save_dir[i].split('.')[0] + '.png', mask_)
    union, TP, FP, FN, precision, recall, iou, f1 = union_intersect(gt,mask_,threshold=100)
    f = open(os.path.join("../notebooks/result", 'output.csv'), 'a', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow([save_dir[i].split('.')[0], f1, iou, precision, recall])
    f.close()
    io.imsave('../notebooks/result/' + save_dir[i].split('.')[0] + '_result.png', np.dstack((TP+FP, FN+FP, np.zeros(TP.shape))))

# for m in range(len(y_pred)):
#     for i in range(y_pred[m].shape[0]):
#         mask_ = y_pred[m][i,:,:,0]
        # gt = y_test[i,:,:,0]
        # io.imsave('../notebooks/result/' + save_dir[i].split('.')[0] + '.png', mask_)
#         io.imsave('../notebooks/result/' + 'shapefile_%d_%d'%(m,i) + '.png', mask_)
        #cv2.imwrite(result_dir + "/result/%s" %filename,mask_)
        # union, TP, FP, FN, precision, recall, iou, f1 = union_intersect(gt,mask_,threshold=120)
        # f = open(os.path.join("../notebooks/result", 'output.csv'), 'a', encoding='utf-8', newline='')
        # wr = csv.writer(f)
        # wr.writerow([save_dir[i].split('.')[0], f1, iou, precision, recall])
        # f.close()
        # io.imsave('../notebooks/result/' + save_dir[i].split('.')[0] + '_result.png', np.dstack((TP+FP, FN+FP, np.zeros(TP.shape))))

for j in range(y_pred.shape[0]):
    img = y_pred[j].astype(np.uint8)

    ret, thresh = cv2.threshold(img[:, :, 0], 127, 255, 0)
    contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area_list = []
    for i in range(len(contours)):
        if len(contours[i]) > 1 and cv2.contourArea(contours[i]) > 200:
            area_list.append(cv2.contourArea(contours[i]))
            x0, y0 = zip(*np.squeeze(contours[i]))
            plt.plot(x0, y0, c="b", linewidth=1.0)

    plt.axis('off')
    h = y_pred[j].shape[0]
    w = y_pred[j].shape[1]
    zeros = np.zeros((h, w))
    ones = y_pred[j].reshape(h, w)
    mask = np.stack((ones, zeros, zeros, ones), axis=-1)
    print(area_list)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.imshow(imgs_test_np[j, :, :])
    plt.imshow(mask, alpha=0.3)
    plt.xlim(0, data_resize)
    plt.ylim(data_resize, 0)
    plt.show()
    plt.savefig('../notebooks/result/' + save_dir[j], bbox_inches='tight', pad_inches=0, dpi=300)
    plt.clf()

# vectorizing through contour method and calculate mask area
# for n in range(len(y_pred)):
    #     for j in range(y_pred[n].shape[0]):
    #         print(n,j)
    #     img =y_pred[n][j].astype(np.uint8)
    #     ind_size = n * j
    #     # imgray = cv2.cvtColor(img,cv2.COLOR_GRAY2GRAY)
    #     ret, thresh = cv2.threshold(img[:,:,0],127,255,0)
    #     contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #     area_list = []
    #     polygon_list = []
    #     for i in range(len(contours)):
    #         if len(contours[i]) > 1 and cv2.contourArea(contours[i]) > 500:
    #             area_list.append(cv2.contourArea(contours[i]))
    #             x0, y0 = zip(*np.squeeze(contours[i]))
    #             polygon = []
    #
    #             for m in range(len(x0)):
    #                 polygon.append(
    #                 [345527 + ((347947 - 345527) / 9680) * data_resize* j + ((347947 - 345527) / 9680) * data_resize * (x0[m]/data_resize),
    #                 298865 - ((298865 - 295954) / 11644) * data_resize* n - ((298865 - 295954) / 11644) * data_resize * (y0[m]/data_resize)]
    #                 )
    #             polygon_list.append(polygon)
    #             #w = shapefile.Writer('../notebooks/result/' + 'shapefile_%d_%d'%(j,i))
    #             #w.field('region', 'N')
    #             #w.field('area', 'N')
    #             #w.poly([polygon])
    #             #w.record(region=k, area=int(cv2.contourArea(contours[i])))
    #             #w.close
    #             plt.plot(x0, y0, c="b", linewidth = 1.0)
    #             # io.imsave(save_dir[i], [x0, y0])
    #     if len(area_list) >= 1 :
    #         w = shapefile.Writer('../notebooks/result/' + 'shapefile_%d_%d'%(n,j))
    #         w.field('region', 'N')
    #         w.poly(polygon_list)
    #         w.record(region=k)
    #         w.close

    #     plt.axis('off')
    #     h = y_pred[n][j].shape[0]
    #     # h=data_resize
    #     w = y_pred[n][j].shape[1]
    #     # w=data_resize
    #     zeros = np.zeros((h, w))
    #     ones = y_pred[n][j].reshape(h, w)
    #     mask = np.stack((ones, zeros, zeros, ones), axis=-1)
    #     # io.imsave(save_dir[j], imgs_test_np[j,:,:]+mask)
    #     print(area_list)
    #     plt.axis('off')
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.tight_layout()
    #     plt.imshow(x_crops[j,:,:])
    #     plt.imshow(mask, alpha=0.3)
    #     # io.imsave(save_dir[j], imgs_test_np[j,:,:]+mask)
    #     # plt.imshow(img[:,:,0])
    #     plt.xlim(0, data_resize)
    #     plt.ylim(data_resize, 0)
    # #     # plt.show()
    #     # plt.savefig('../notebooks/result/' + save_dir[j], bbox_inches='tight', pad_inches=0, dpi=300)
    #     plt.savefig('../notebooks/result/' + 'shapefile_%d_%d.jpg'%(n, j), bbox_inches='tight', pad_inches=0, dpi=300)
    #     plt.clf()

    #     # writing coordinate system(generating project file)
    #     prj = open("../notebooks/result/shapefile_%d_%d.prj"%(n, j), "w")
    #     epsg = getWKT_PRJ("8862")
    #     prj.write(epsg)
    #     prj.close()