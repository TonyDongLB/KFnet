from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
from optparse import OptionParser

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

#from libtiff import TIFF

class myAugmentation(object):
    """
    A class used to augmentate image
    Firstly, read train image and label seperately, and then merge them together for the next process
    Secondly, use keras preprocessing to augmentate image
    Finally, seperate augmentated image apart into train image and label
    """

    def __init__(self, train_path, label_path, img_type, crop_box=None, need_elatic= False):

        """
        Using glob to get all .img_type form path
        """

        self.train_imgs = glob.glob(train_path + "/*." + img_type)  # 训练集
        self.label_imgs = glob.glob(label_path + "/*." + img_type)  # label
        self.train_path = train_path
        self.label_path = label_path

        dir = os.path.dirname(train_path)
        self.merge_path = os.path.join(dir, 'merge')
        self.aug_merge_path = os.path.join(dir, 'aug_merge')
        self.aug_train_path = os.path.join(dir, 'aug_train')
        self.aug_label_path = os.path.join(dir, 'aug_label')
        self.crop_box = crop_box
        self.need_elatic = need_elatic

        # for img_dir in [self.aug_merge_path, self.aug_train_path, self.aug_label_path, self.merge_path]:
        #     if not os.path.exists(img_dir):
        #         os.makedirs(img_dir)
        #     else:
        #         os.remove(img_dir)
        #         os.makedirs(img_dir)

        self.img_type = img_type
        self.slices = len(self.train_imgs)
        self.datagen = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.1,
            horizontal_flip=True,
            channel_shift_range=20,
            fill_mode='nearest')

    def Augmentation(self):
        # 读入3通道的train和label, 分别转换成矩阵, 然后将label的第一个通道放在train的第2个通处, 做数据增强
        print("Augmentation")
        """
        Start augmentation.....
        """
        trains = self.train_imgs
        labels = self.label_imgs
        path_train = self.train_path
        path_label = self.label_path
        path_merge = self.merge_path
        imgtype = self.img_type
        path_aug_merge = self.aug_merge_path
        crop_box = self.crop_box
        print('共有' + str(len(trains)) + '张照片。')
        if len(trains) != len(labels) or len(trains) == 0 or len(trains) == 0:
            print("trains can't match labels")
            return 0
        for i in range(len(trains)):
            filename = trains[i].split('/')[-1]
            pre = filename.split('.')[0]
            suf = filename.split('.')[1]
            img_t = load_img(path_train + '/' + filename)  # 读入train
            img_l = load_img(path_label + '/' + pre + '_mask.' + suf)  # 读入label

            if crop_box:
                img_t = img_t.crop(crop_box)
                img_l = img_l.crop(crop_box)

            x_t = img_to_array(img_t)                                    # 转换成矩阵
            x_l = img_to_array(img_l)
            x_t[:, :, 2] = x_l[:, :, 0]                                  # 把label当做train的第三个通道
            img_tmp = array_to_img(x_t)
            img_tmp.save(path_merge + "/" + filename)      # 保存合并后的图像
            img = x_t
            img = img.reshape((1,) + img.shape)                          # 改变shape(1, 512, 512, 3)
            savedir = path_aug_merge + "/" + filename.split('.')[0]                      # 存储合并增强后的图像
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            print('在处理第' + str(i) + '张照片。')

            self.doAugmentate(img, savedir, filename.split('.')[0])                      # 数据增强

    def doAugmentate(self, img, save_to_dir, save_prefix, batch_size=1, save_format=None, imgnum=30):
        print("doAugmenttaion")
        """
        augmentate one image
        """
        datagen = self.datagen
        if save_format is None:
            save_format = self.img_type
        i = 0
        for batch in datagen.flow(img,
                                  batch_size=batch_size,
                                  save_to_dir=save_to_dir,
                                  save_prefix=save_prefix,
                                  save_format=save_format):
            i += 1
            if i > imgnum:
                break

    def elastic_transform(self, image, alpha, sigma, alpha_affine, random_state=None):

        """Elastic deformation of images as described in [Simard2003]_ (with modifications).
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
             Convolutional Neural Networks applied to Visual Document Analysis", in
             Proc. of the International Conference on Document Analysis and
             Recognition, 2003.

         Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
        """

        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape
        shape_size = shape[:2]

        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32(
            [center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
             center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
        return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

    def splitMerge(self):
        # 读入合并增强之后的数据(aug_merge), 对其进行分离, 分别保存至 aug_train, aug_label
        print("splitMerge")
        """
        split merged image apart
        """
        path_merge = self.aug_merge_path       # 合并增强之后的图像
        path_train = self.aug_train_path       # 增强之后分离出来的train
        path_label = self.aug_label_path       # 增强之后分离出来的label
        all_path = glob.glob(path_merge + '/*')
        for path in all_path:
            print(path)
            train_imgs = glob.glob(path + "/*." + 'tif')
            i = path.split('/')[-1]
            savedir = path_train + "/" + i                   # 保存训练集的路径
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            savedir = path_label + "/" + i                   # 保存label的路径
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            for imgname in train_imgs:         # rindex("/") 是返回'/'在字符串中最后一次出现的索引
                midname = imgname[imgname.rindex("/") + 1:imgname.rindex("." + 'tif')] # 获得文件名(不包含后缀)
                img = cv2.imread(imgname)      # 读入训练图像
                img_train = img[:, :, 2]  # 训练集是第2个通道, label是第0个通道
                img_label = img[:, :, 0]
                cv2.imwrite(path_train + "/" + str(i) + "/" + midname + "_train" + "." + self.img_type, img_train) # 保存训练图像和label
                cv2.imwrite(path_label + "/" + str(i) + "/" + midname + "_label" + "." + self.img_type, img_label)



def get_args():
    parser = OptionParser()
    parser.add_option('--train_path', dest='train_path', default='/Users/apple/Documents/NN_Models/hand/train')
    parser.add_option('--label_path', dest='label_path', default='/Users/apple/Documents/NN_Models/hand/label')
    parser.add_option('--img_type', dest='img_type', default='jpg')

    (options, args) = parser.parse_args()
    return options

if __name__ == "__main__":
    args = get_args()
    aug = myAugmentation(train_path=args.train_path, label_path= args.label_path, img_type= args.img_type, need_elatic=True)
    aug.Augmentation()
    aug.splitMerge()
    print('!!!ALL DONE!!!')
