#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# Written by Chao CHEN (chaochancs@gmail.com)
# Created On: 2017-08-11
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.

    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        # print('inputs.size(0) == ' + str(inputs.size()))
        # N = inputs.size(0)
        # # print(N)
        # C = inputs.size(1)
        if inputs.dim() == 4:
            inputs = inputs.contiguous().view(inputs.size(0), inputs.size(1), -1)
            inputs = inputs.transpose(1, 2)
            inputs = inputs.contiguous().view(-1, inputs.size(2)).squeeze()

        if targets.dim() == 4:
            targets = targets.contiguous().view(targets.size(0), targets.size(1), -1)
            targets = targets.transpose(1,2)
            targets = targets.contiguous().view(-1, targets.size(2)).squeeze()
        else:
            targets = targets.view(-1)

        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(inputs.size()).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        ids = ids.type(torch.LongTensor)
        src = Variable(torch.ones(class_mask.size()))
        if inputs.is_cuda:
            src = src.cuda()
            if not ids.is_cuda:
                ids = ids.cuda()
        class_mask.scatter_(1, ids, src)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


if __name__ == "__main__":
    FL = FocalLoss(class_num=2, gamma=0)
    CE = nn.CrossEntropyLoss()
    N = 4
    C = 2
    inputs = torch.rand(N, C)
    targets = torch.LongTensor(N).random_(C)
    inputs_fl = Variable(inputs.clone(), requires_grad=True)
    targets_fl = Variable(targets.clone())

    inputs_ce = Variable(inputs.clone(), requires_grad=True)
    targets_ce = Variable(targets.clone())
    print('----inputs----')
    print(inputs)
    print('---target-----')
    print(targets)

    fl_loss = FL(inputs_fl, targets_fl)
    ce_loss = CE(inputs_ce, targets_ce)
    print('ce = {}, fl ={}'.format(ce_loss.data[0], fl_loss.data[0]))
    fl_loss.backward()
    ce_loss.backward()
    # print(inputs_fl.grad.data)
    print(inputs_ce.grad.data)

    # N = 5
    # C = 4
    # inputs = torch.rand(3, 2, N, C)
    # targets = torch.LongTensor(3, 1, N, C, ).random_(2)
    # inputs_fl = Variable(inputs.clone(), requires_grad=True)
    # targets_fl = Variable(targets.clone())
    #
    # inputs_ce = Variable(inputs.clone(), requires_grad=True)
    # targets_ce = Variable(targets.clone())
    # print('----inputs----')
    # print(inputs_fl)
    # print('---target-----')
    # print(targets_fl)
    # # print('----trans----')
    # # trans = inputs_fl.transpose(0, 2)
    # # trans = trans.contiguous().view(-1, trans.size(2)).squeeze().shape
    # # print(trans)
    #
    # fl_loss = FL(inputs_fl, targets_fl)
    # # ce_loss = CE(inputs_ce, targets_ce)
    # # print('ce = {}'.format(ce_loss.data[0]))
    # print('fl ={}'.format(fl_loss.data[0]))
    # fl_loss.backward()
    # # ce_loss.backward()
    # print(inputs_fl.grad.data)
    # # print(inputs_ce.grad.data)



