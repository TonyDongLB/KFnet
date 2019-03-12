import sys
import os
from optparse import OptionParser
import numpy as np
import glob
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import cv2
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from PIL import Image
from torchvision import transforms as T
from model import CamNet
from dataset import Hand
from loss import FocalLoss2d

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

normalize = T.Normalize(mean=[0.4, 0.4, 0.4], std=[0.4, 0.4, 0.4])
transforms4img = T.Compose([
                T.ToTensor(),
                normalize
            ])

def ave_score(ori_score):
    final_result = ori_score.copy()
    for i in range(len(ori_score)):
        if i < 4:
            continue
        if i + 4 >= len(ori_score):
            break
        score = 0.
        score += sum(ori_score[i -4: i +4 + 1]) / 7
        for i in range(i - 4,i + 4 + 1):
            if score > 0.8:
                final_result[i] = 1
            else:
                final_result[i] = 0
    return final_result

def get_new_data(images, flows, videoCapture, opt_frames, distrance4flow = 40):
    if len(images) < opt_frames * 2 + 1:
        while len(images) < opt_frames * 2 + 1:
            sucess, new_image = videoCapture.read()
            if not sucess:
                videoCapture.release()
                return sucess
            new_image = cv2.resize(new_image, (224, 224))
            images.append(new_image)
    else:
        for i in range(len(images) - 1):
            images[i] = images[i + 1].copy()
        sucess, new_image = videoCapture.read()
        if not sucess:
            videoCapture.release()
            return sucess
        new_image = cv2.resize(new_image, (224, 224))
        images[-1] = new_image
    if len(flows) < opt_frames * 2:
        while len(flows) < opt_frames * 2:
            for i in range(len(flows), opt_frames * 2):
                flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY),
                                                    cv2.cvtColor(images[i+1], cv2.COLOR_BGR2GRAY),
                                                    None,
                                                    pyr_scale=0.5, levels=3, winsize=15,
                                                    iterations=3, poly_n=5, poly_sigma=1.2,
                                                    flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
                attach = np.zeros((flow.shape[0], flow.shape[1], 1), np.float32)
                flow += distrance4flow
                flow /= distrance4flow * 2
                flow = np.concatenate((flow, attach), 2)
                flow = np.transpose(flow, (2, 0, 1))
                flow = torch.from_numpy(flow)
                flow = torch.unsqueeze(flow, dim=0)
                flows.append(flow)
    else:
        for i in range(len(flows) - 1):
            flows[i] = flows[i + 1]
        flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(images[-2], cv2.COLOR_BGR2GRAY),
                                                    cv2.cvtColor(images[-1], cv2.COLOR_BGR2GRAY),
                                                    None,
                                                    pyr_scale=0.5, levels=3, winsize=15,
                                                    iterations=3, poly_n=5, poly_sigma=1.2,
                                                    flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        attach = np.zeros((flow.shape[0], flow.shape[1], 1), np.float32)
        flow += distrance4flow
        flow /= distrance4flow * 2
        flow = np.concatenate((flow, attach), 2)
        flow = np.transpose(flow, (2, 0, 1))
        flow = torch.from_numpy(flow)
        flow = torch.unsqueeze(flow, dim=0)
        flows[-1] = flow
    return (images, flows)


def deploy(model, gpu, video_path, opt_frames, distrance4flow = 40):
    videoCapture = cv2.VideoCapture(video_path)
    images = []
    flows = []
    result_pre = []
    index = 0

    while True:
        index += 1
        #  设置数据
        # print(index)
        result = get_new_data(images, flows, videoCapture, opt_frames, distrance4flow)
        if not result:
            break
        images = result[0]
        flows = result[1]
        img_1 = images[opt_frames]
        displayImg = cv2.resize(img_1, (640, 480))
        img_1_to_predict = transforms4img(img_1)
        img_1_to_predict = torch.unsqueeze(img_1_to_predict, dim=0)

        # 推测
        img_1_to_predict = Variable(img_1_to_predict)
        for i in range(len(flows)):
            flows[i] = Variable(flows[i])

        if gpu:
            img_1_to_predict = img_1_to_predict.cuda()
            for i in range(len(flows)):
                flows[i] = flows[i].cuda()

        output = model(img_1_to_predict, flows)


        pred = output.data.max(1)[1]
        classed = int(pred.cpu().numpy()[0])
        result_pre.append(classed)

    result_pre = ave_score(result_pre)
    videoCapture.release()
    videoCapture = cv2.VideoCapture(video_path)
    for i in range(opt_frames):
        sucess, new_image = videoCapture.read()
    i = 0
    while True:
        sucess, new_image = videoCapture.read()
        if not sucess:
            break
        if result_pre[i] == 1:
            cv2.putText(new_image, "SOMEONE IS WRITING", (30, 30), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 4)
        else:
            cv2.putText(new_image, "NULL", (30, 30), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 4)
        i += 1
        cv2.imshow("test Video", new_image)
        cv2.waitKey(200)

    cv2.destroyWindow('test Video')



def get_args():
    parser = OptionParser()
    parser.add_option('-b', '--batch-size', dest='batchsize', default=1,
                      type='int', help='batch size')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('--model', '-m', default='/home/dl/Work/HandNet/checkpoints/epoch_40_0.925531914893617_dist=60_0308_04:13:33.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_option('--distance4flow', dest='distance4flow', default=60,
                      type='int', help='distance for flow')
    parser.add_option('--opt_frames', dest='opt_frames', default=2,
                      type='int',
                      help='recent opt flow frames(before and after, so the total opt_frames is double this value')
    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()
    root_data = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'data')
    videos = glob.glob(root_data + '/deploy/videos' + "/*." + 'avi')

    model = CamNet(is_predict=True, opt_frames=args.opt_frames)

    for video in videos:
        if args.gpu:
            model.cuda()
            # cudnn.benchmark = True # faster convolutions, but more memory
            print("Using ", torch.cuda.device_count(), " GPUs!")
        if torch.cuda.device_count() > 1:
            print("Using ", torch.cuda.device_count(), " GPUs!")
            model = nn.DataParallel(model)

        model.load(path=args.model, gpu=args.gpu)
        model.eval()
        print('Processing!')
        deploy(model=model, gpu=args.gpu, video_path=video, distrance4flow=args.distance4flow, opt_frames=args.opt_frames)
        print('DONE')
