import os
from torch.utils import data
import numpy as np
import torch
from torchvision import transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader
import glob
import cv2

class Hand(data.Dataset):
    # # to load the hand picture
    def __init__(self, root, opt_frames=1,transforms=None, train=False, test=False, deploy=False, distrance4flow = 40):
        super(Hand, self).__init__()
        self.root = root
        self.transforms = transforms
        self.train = train
        self.test = test
        self.deploy = deploy
        self.distance4flow = distrance4flow
        self.opt_frames = opt_frames

        if test:
            self.imgs = glob.glob(root + '/test/0' + "/*." + 'jpg')
            self.imgs += glob.glob(root + '/test/1' + "/*." + 'jpg')
        elif train:
            self.imgs = glob.glob(root + '/train/0' + "/*." + 'jpg')
            self.imgs += glob.glob(root + '/train/1' + "/*." + 'jpg')
        else:
            self.imgs = glob.glob(root + '/deploy' + "/*." + 'jpg')
        if transforms is None:
            # normalize need to edit!
            normalize = T.Normalize(mean=[0.470, 0.521, 0.555], std=[0.324, 0.315, 0.300])
            self.transforms4img = T.Compose([
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        distance4flow = self.distance4flow
        opt_frames = self.opt_frames
        img_path = self.imgs[index]
        label = None
        filename = img_path.split('/')[-1].split('.')[0]
        num = filename.split("_")[-1]
        images = []
        for i in range(-opt_frames, opt_frames + 1):
            this_num = str(int(num) + 10 * i)
            this_img_path = img_path.replace(num, this_num)
            if not os.path.exists(this_img_path):
                images.append(None)
            else:
                images.append(cv2.resize(cv2.imread(this_img_path), (320, 240)))
        first_img = None
        last_img = None

        # 当有较多图像为空，需要填充的时候，此时设置为负样本
        num_of_img_is_none = 0
        for img in images:
            if img is None:
                num_of_img_is_none += 1
        if num_of_img_is_none >= 1:
            label = 0

        # 填充缺失的图像
        for i in range(len(images)):
            if images[i] is not None:
                last_img = images[i].copy()
            if images[i] is None and last_img is not None:
                images[i] = last_img.copy()
        for i in range(len(images)):
            i_rev = len(images) - i - 1
            if images[i_rev] is not None:
                first_img = images[i_rev].copy()
            if images[i_rev] is None and first_img is not None:
                images[i_rev] = first_img.copy()

        optical_flows = []
        for i in range(len(images) - 1):
            flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY),
                                                 cv2.cvtColor(images[+1], cv2.COLOR_BGR2GRAY),
                                                 flow=None,
                                                 pyr_scale=0.5, levels=3, winsize=15,
                                                 iterations=3, poly_n=5, poly_sigma=1.2,
                                                 flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            attach = np.zeros((flow.shape[0], flow.shape[1], 1), flow.dtype)
            flow = np.clip(flow, -distance4flow, distance4flow)
            flow /= distance4flow
            flow = np.concatenate((flow, attach), 2)
            optical_flows.append(flow)

        # # add a random flip
        img_1 = images[opt_frames]
        if np.random.random() > 0.5 and not self.deploy:
            img_1 = np.flip(img_1, 1).copy()
            for i in range(len(optical_flows)):
                optical_flows[i] = np.flip(optical_flows[i], 1).copy()

        # img_1 = self.transforms4img(img_1)
        # flow0 = np.transpose(flow0, (2, 0, 1))
        # flow1 = np.transpose(flow1, (2, 0, 1))
        # 补足为320*320
        img_1 = np.pad(img_1, [(40, 40), (0, 0), (0, 0)],
                       mode='constant', constant_values=0)
        for i in range(len(optical_flows)):
            optical_flows[i] = np.pad(optical_flows[i], [(40, 40), (0, 0), (0, 0)],
                                      mode='constant', constant_values=0)

        img_1 = self.transforms4img(img_1)
        for i in range(len(optical_flows)):
            optical_flows[i] = np.transpose(optical_flows[i], (2, 0, 1))

        if label is None:
            label = int(img_path.split('/')[-2])

        return img_1, optical_flows, label

    def __len__(self):
        return len(self.imgs)

