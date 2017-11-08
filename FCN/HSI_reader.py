import csv
import sys
import cv2
from shapely.wkt import loads as wkt_loads
import shapely.affinity
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
import scipy.misc as misc

class BatchDatset:
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0
    N_Cls = 10
    inDir = '../dstl'
    DF = pd.read_csv(inDir + '/train_wkt_v4.csv')
    GS = pd.read_csv(inDir + '/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
    SB = pd.read_csv(os.path.join(inDir, 'sample_submission.csv'))
    ISZ = 224
    smooth = 1e-12

    def __init__(self, image_options={}, num_classes = 10):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.image_options = image_options
        self.N_Cls = num_classes
        self.inDir = '../dstl'
        self.DF = pd.read_csv(self.inDir + '/train_wkt_v4.csv')
        self.GS = pd.read_csv(self.inDir + '/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
        self.SB = pd.read_csv(os.path.join(self.inDir, 'sample_submission.csv'))
        self.ISZ = 224
        self.smooth = 1e-12
        self._read_images()



    def _read_images(self):
        self.stick_all_train()
        _,_ = self.make_val()
        self.__channels = True
        img = np.load('../data/x_trn_%d.npy' % self.N_Cls)
        msk = np.load('../data/y_trn_%d.npy' % self.N_Cls)
        self.images , self.annotations= self.get_patches(img, msk, amt=4000)
        print (self.images.shape)
        print (self.annotations.shape)

    def _convert_coordinates_to_raster(self, coords, img_size, xymax):
        # __author__ = visoft
        # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
        Xmax, Ymax = xymax
        H, W = img_size
        W1 = 1.0 * W * W / (W + 1)
        H1 = 1.0 * H * H / (H + 1)
        xf = W1 / Xmax
        yf = H1 / Ymax
        coords[:, 1] *= yf
        coords[:, 0] *= xf
        coords_int = np.round(coords).astype(np.int32)
        return coords_int

    def _get_xmax_ymin(self, grid_sizes_panda, imageId):
        # __author__ = visoft
        # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
        xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0, 1:].astype(float)
        return (xmax, ymin)

    def _get_polygon_list(self, wkt_list_pandas, imageId, cType):
        # __author__ = visoft
        # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
        df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
        multipoly_def = df_image[df_image.ClassType == cType].MultipolygonWKT
        polygonList = None
        if len(multipoly_def) > 0:
            assert len(multipoly_def) == 1
            polygonList = wkt_loads(multipoly_def.values[0])
        return polygonList

    def _get_and_convert_contours(self, polygonList, raster_img_size, xymax):
        # __author__ = visoft
        # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
        perim_list = []
        interior_list = []
        if polygonList is None:
            return None
        for k in range(len(polygonList)):
            poly = polygonList[k]
            perim = np.array(list(poly.exterior.coords))
            perim_c = self._convert_coordinates_to_raster(perim, raster_img_size, xymax)
            perim_list.append(perim_c)
            for pi in poly.interiors:
                interior = np.array(list(pi.coords))
                interior_c = self._convert_coordinates_to_raster(interior, raster_img_size, xymax)
                interior_list.append(interior_c)
        return perim_list, interior_list

    def _plot_mask_from_contours(self, raster_img_size, contours, class_value=1):
        # __author__ = visoft
        # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
        img_mask = np.zeros(raster_img_size, np.uint8)
        if contours is None:
            return img_mask
        perim_list, interior_list = contours
        cv2.fillPoly(img_mask, perim_list, class_value)
        cv2.fillPoly(img_mask, interior_list, 0)
        return img_mask

    def generate_mask_for_image_and_class(self, raster_size, imageId, class_type, grid_sizes_panda=GS, wkt_list_pandas=DF):
        # __author__ = visoft
        # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
        xymax = self._get_xmax_ymin(grid_sizes_panda, imageId)
        polygon_list = self._get_polygon_list(wkt_list_pandas, imageId, class_type)
        contours = self._get_and_convert_contours(polygon_list, raster_size, xymax)
        mask = self._plot_mask_from_contours(raster_size, contours, 1)
        return mask

    def M(self, image_id):
        # __author__ = amaia
        # https://www.kaggle.com/aamaia/dstl-satellite-imagery-feature-detection/rgb-using-m-bands-example
        # filename = os.path.join(inDir, 'sixteen_band', '{}_M.tif'.format(image_id))
        filename = os.path.join(self.inDir, 'three_band', '{}.tif'.format(image_id))
        img = tiff.imread(filename)
        img = np.rollaxis(img, 0, 3)
        return img

    def stretch_n(self, bands, lower_percent=5, higher_percent=95):
        out = np.zeros_like(bands).astype(np.float32)
        n = bands.shape[2]
        for i in range(n):
            a = 0  # np.min(band)
            b = 255  # np.max(band)
            c = np.percentile(bands[:, :, i], lower_percent)
            d = np.percentile(bands[:, :, i], higher_percent)
            t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
            t[t < a] = a
            t[t > b] = b
            out[:, :, i] = t

        return out.astype(np.float32)

    def jaccard_coef(self, y_true, y_pred):
        # __author__ = Vladimir Iglovikov
        intersection = np.sum(y_true * y_pred)
        sum_ = np.sum(y_true + y_pred)

        jac = (intersection + self.smooth) / (sum_ - intersection + self.smooth)

        return np.mean(jac)

    def jaccard_coef_int(self, y_true, y_pred):
        # __author__ = Vladimir Iglovikov
        y_pred_pos = np.round(np.clip(y_pred, 0, 1))

        intersection = np.sum(y_true * y_pred_pos)
        sum_ = np.sum(y_true + y_pred_pos)
        jac = (intersection + self.smooth) / (sum_ - intersection + self.smooth)
        return np.mean(jac)

    def stick_all_train(self):
        print("let's stick all imgs together")
        s = 835

        x = np.zeros((5 * s, 5 * s, 3))
        y = np.zeros((5 * s, 5 * s, self.N_Cls))

        ids = sorted( self.DF.ImageId.unique())
        print(len(ids))
        for i in range(5):
            for j in range(5):
                id = ids[5 * i + j]

                img = self.M(id)
                img = self.stretch_n(img)
                print(img.shape, id, np.amax(img), np.amin(img))
                x[s * i:s * i + s, s * j:s * j + s, :] = img[:s, :s, :]
                for z in range(self.N_Cls):
                    y[s * i:s * i + s, s * j:s * j + s, z] = self.generate_mask_for_image_and_class(
                        (img.shape[0], img.shape[1]), id, z + 1)[:s, :s]
        #x = x * 255
        np.save('../data/x_trn_%d' % self.N_Cls, x)
        np.save('../data/y_trn_%d' % self.N_Cls, y)


    def get_patches(self, img, msk, amt=10000, aug=False):
        is2 = int(1.0 * self.ISZ)
        xm, ym = img.shape[0] - is2, img.shape[1] - is2

        x, y = [], []

        tr = [0.4, 0.1, 0.1, 0.15, 0.3, 0.95, 0.1, 0.05, 0.001, 0.005]
        for i in range(amt):
            xc = random.randint(0, xm)
            yc = random.randint(0, ym)

            im = img[xc:xc + is2, yc:yc + is2]
            ms = msk[xc:xc + is2, yc:yc + is2]

            for j in range(self.N_Cls):
                sm = np.sum(ms[:, :, j])
                if 1.0 * sm / is2 ** 2 > tr[j]:
                    if aug:
                        if random.uniform(0, 1) > 0.5:
                            im = im[::-1]
                            ms = ms[::-1]
                        if random.uniform(0, 1) > 0.5:
                            im = im[:, ::-1]
                            ms = ms[:, ::-1]

            x.append(im)
            y.append(ms)

        #x, y = 2 * np.transpose(x, (0, 3, 1, 2)) - 1, np.transpose(y, (0, 3, 1, 2))
        #print(x.shape, y.shape, np.amax(x), np.amin(x), np.amax(y), np.amin(y))
        x = np.reshape(x, (-1,self.ISZ,self.ISZ,3))
        y = np.reshape(y, (-1, self.ISZ, self.ISZ, self.N_Cls))
        return x, y

    def make_val(self):
        print("let's pick some samples for validation")
        img = np.load('../data/x_trn_%d.npy' % self.N_Cls)
        msk = np.load('../data/y_trn_%d.npy' % self.N_Cls)
        x, y = self.get_patches(img, msk, amt=100)

        np.save('../data/x_tmp_%d' % self.N_Cls, x)
        np.save('../data/y_tmp_%d' % self.N_Cls, y)
        return x, y

    def _transform(self, filename):
        image = misc.imread(filename)
        if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
            image = np.array([image for i in range(3)])

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(image,
                                         [resize_size, resize_size], interp='nearest')
        else:
            resize_image = image

        return np.array(resize_image)

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]












"""
make_val()
x_val, y_val = np.load('data/x_tmp_%d.npy' % N_Cls), np.load('data/y_tmp_%d.npy' % N_Cls)
print('xval')
print(x_val[0].shape)
print('yval')
print(y_val[0].shape)
plt.figure()
ax1 = plt.subplot(131)
ax1.set_title('image ID:')
ax1.imshow(x_val[0,5, :, :], cmap=plt.get_cmap('gist_ncar'))
ax2 = plt.subplot(132)
ax2.set_title('predict trees pixels')
ax2.imshow(y_val[0,4,:,:], cmap=plt.get_cmap('gray'))
ax2 = plt.subplot(133)
ax2.set_title('predict struct pixels')
ax2.imshow(y_val[0,1,:,:], cmap=plt.get_cmap('gray'))

plt.show()
"""


#img = np.load('data/x_trn_%d.npy' % N_Cls)
#msk = np.load('data/y_trn_%d.npy' % N_Cls)
#print(img[0].shape)
#print(msk[0].shape)
#x_trn, y_trn = get_patches(img, msk)