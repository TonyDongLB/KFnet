import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from PIL import Image
from torchvision import transforms as T
from model import CamNet
from dataset import Hand
from loss import FocalLoss2d
from utils import *


def train_model(model, opt_frames=1, epochs=200, batch_size=16, lr=0.01, gpu=True, distance4flow=40):
    # #设置路径
    root_data = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'data')
    dir_checkpoint = 'checkpoints/0130/'
    writer = SummaryWriter('log/0130/')

    # # 设计数据
    train_set = Hand(root_data, opt_frames=opt_frames, train=True, distrance4flow=distance4flow)
    test_set = Hand(root_data, opt_frames=opt_frames, test=True, distrance4flow=distance4flow)
    train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                            num_workers=6)
    test_data = DataLoader(test_set, batch_size=1, shuffle=True,
                           num_workers=6)

    # # setting optimizer and criterion
    optimizer = torch.optim.Adam(
        [
            {'params': model.img_feature.parameters(), 'lr': 1e-2 / 2, 'weight_decay': 1e-3},
            {'params': model.flow_feature.parameters()},
            {'params': model.fusion4flow.parameters()},
            {'params': model.reductionFlowFea.parameters()},
            {'params': model.fusion4img.parameters()},
            {'params': model.layer4fusion.parameters()},
            {'params': model.layer4flow.parameters()},
            {'params': model.fc_fusion.parameters()},
            {'params': model.fc_flow.parameters()},
        ],
        lr=lr,
        weight_decay=1e-3)

    # optimizer = optim.SGD([
    #             {'params': model.img_feature.parameters(), 'lr': 1e-2 / 2, 'weight_decay': 1e-3},
    #             {'params': model.flow_feature.parameters()},
    #             {'params': model.fusion.parameters()},
    #             {'params': model.layer4fusion.parameters()},
    #             {'params': model.layer4flow.parameters()},
    #             {'params': model.fc_fusion.parameters()},
    #             {'params': model.fc_flow.parameters()},
    #         ],
    #         lr=lr,
    #         momentum=0.9,
    #         weight_decay=0.0005)

    # # to use CEloss with weight
    # ??? when i add this fucking weight???
    weight = torch.Tensor([1, 1])
    if gpu:
        weight = weight.cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    # # to use CrossEntropyLoss
    # criterion = FocalLoss2d(gamma=2)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        CUDA: {}
        Trainset size: {}
        Testset  size: {}
        opt_frames : {}
        distance4flow : {}
    '''.format(epochs, batch_size, lr, str(gpu), len(train_set), len(test_set), opt_frames, distance4flow))

    processed_batch = 0
    last_accuracy = 0

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        epoch_loss = 0
        num_i = 0

        for ii, (img1, flows, label) in enumerate(train_data):
            num_i += 1
            processed_batch += 1

            img1 = Variable(img1)
            for i in range(len(flows)):
                flows[i] = Variable(flows[i])
            label = Variable(label).long()

            if gpu:
                img1 = img1.cuda()
                for i in range(len(flows)):
                    flows[i] = flows[i].cuda()
                label = label.cuda()

            optimizer.zero_grad()

            prediction = model(img1, flows)

            loss = criterion(prediction, label)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / num_i))
        writer.add_scalar('image_train_loss_epoch', epoch_loss / num_i, epoch + 1)

        # # 测试网络
        model.eval()

        val_accuracy = eval(model, test_data, gpu=gpu, criterion=criterion)
        writer.add_scalar('val_accuracy', val_accuracy, epoch + 1)
        print('val_accuracy: {}'.format(val_accuracy))

        model.train()

        # # 修改学习率
        if last_accuracy == 0:
            last_accuracy = val_accuracy
        elif val_accuracy < last_accuracy:
            for param_group in optimizer.param_groups:
                if param_group['lr'] <= 1e-6:
                    continue
                param_group['lr'] = param_group['lr'] * 0.1
                print('learn rate is ' + str(param_group['lr'] * 0.1))
        last_accuracy = val_accuracy

        if val_accuracy > 0.90 or epoch > 10:
            if isinstance(model, nn.DataParallel):
                model.module.save(epoch=epoch, eval=val_accuracy)
            else:
                model.save(distance4flow=distance4flow, epoch=epoch, eval=val_accuracy)
            print('Checkpoint {} saved !'.format(epoch + 1))


def eval(model, dataset, gpu=False, criterion=None):
    right = 0

    for ii, (img1, flows, label) in enumerate(dataset):
        img1 = Variable(img1)
        for i in range(len(flows)):
            flows[i] = Variable(flows[i])
        label = Variable(label).long()

        if gpu:
            img1 = img1.cuda()
            for i in range(len(flows)):
                flows[i] = flows[i].cuda()
            label = label.cuda()

        score = model(img1, flows)
        pred = score.max(1)[1]
        right += pred.eq(label).sum().item()

    accuracy = right / len(dataset)

    return accuracy


def get_args():
    parser = OptionParser()
    parser.add_option('--opt_frames', dest='opt_frames', default=2,
                      type='int',
                      help='recent opt flow frames(before and after, so the total opt_frames is double this value')
    parser.add_option('-e', '--epochs', dest='epochs', default=40, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=16,
                      type='int', help='batch size')
    parser.add_option('--distance4flow', dest='distance4flow', default=60,
                      type='int', help='distance for flow')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.01,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    model = CamNet(opt_frames=args.opt_frames)
    model = loadPretrain(model)

    if args.load:
        model.load(args.load)
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        model.cuda()
        cudnn.benchmark = True # faster convolutions, but more memory
    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), " GPUs!")
        model = nn.DataParallel(model)

    try:
        train_model(model=model,
                    opt_frames=args.opt_frames,
                    epochs=args.epochs,
                    batch_size=args.batchsize,
                    lr=args.lr,
                    gpu=args.gpu,
                    distance4flow=args.distance4flow,)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(1)
