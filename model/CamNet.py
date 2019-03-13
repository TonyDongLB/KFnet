from torch import nn
import torch as t
import torch
import math
import time

from model.modules import *
from torchvision.models import ResNet
from collections import OrderedDict

# 参考论文《Convolutional Two-Stream Network Fusion for Video Action Recognition》与
#     《Beyond Short Snippets: Deep Networks for Video Classification》

class CamNet(nn.Module):
    """
    基于SE_ResNet50结构，去掉的maxpool和全连接层，增加了fusion模块。
    注：GitHub上的SENET有权重文件有问题，换回标准的ResNet
    注意在特征提取阶段，对img的特征提取反向传播两次，注意学习率。
    opt_frames : 表示两边各有多少个光流图片，即总光流图片为opt_frames * 2。
    """
    def __init__(self, opt_frames=1, block=Bottleneck, layers=(3, 4, 23, 3), pretrained=False, is_predict = False):
        super(CamNet, self).__init__()
        self.is_predict = is_predict
        self.opt_frames = opt_frames

        self.img_inplanes = 64
        self.flow_inplanes = 64

        self.img_feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(block, 64, layers[0], img_path=True),
            self._make_layer(block, 128, layers[1], stride=2, img_path=True),
            self._make_layer(block, 256, layers[2], stride=2, img_path=True),
        )

        self.flow_feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(block, 64, layers[0], img_path=False),
            self._make_layer(block, 128, layers[1], stride=2, img_path=False),
            self._make_layer(block, 256, layers[2], stride=2, img_path=False),
        )

        self.fusion4flow = nn.Sequential(
            nn.Conv3d(256 * block.expansion, 256 * block.expansion // 2, kernel_size=(1, 1, 1), padding=(0, 0, 0),
                      bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(256 * block.expansion // 2, 256 * block.expansion // 2, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(256 * block.expansion // 2, 256 * block.expansion // 2, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )

        self.reductionFlowFea = nn.Sequential(
            nn.Conv2d(256 * block.expansion * opt_frames, 256 * block.expansion, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(256 * block.expansion, 256 * block.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.fusion4img = nn.Sequential(
            nn.Conv2d(256 * block.expansion * 2, 256 * block.expansion, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(256 * block.expansion, 256 * block.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.layer4fusion = self._make_layer(block, 512, layers[3], stride=2, img_path=True)
        self.layer4flow = nn.Sequential(
            nn.Conv3d(256 * block.expansion // 2, 256 * block.expansion, kernel_size=(3, 3, 3), stride=(1, 2, 2),
                      padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(1024),
            nn.ReLU(inplace=True),
            nn.Conv3d(256 * block.expansion, 256 * block.expansion, kernel_size=(3, 3, 3), padding=(1, 1, 1),
                      bias=False),
            nn.BatchNorm3d(1024),
            nn.ReLU(inplace=True),
        )

        self.avgpool2d = nn.AvgPool2d((10, 10), stride=1)
        self.avgpool3d = nn.AvgPool3d((1, 10, 10), stride=1)

        self.fc_fusion = nn.Sequential(
            nn.Linear(512 * block.expansion, 2),
            nn.Dropout(0.4),
        )
        self.fc_flow = nn.Sequential(
            nn.Linear(512 * block.expansion * 2, 2),
            nn.Dropout(0.4),
        )

        # # 初始化
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for mm in m:
                    if isinstance(mm, nn.Conv2d):
                        n = mm.kernel_size[0] * mm.kernel_size[1] * mm.out_channels
                        mm.weight.data.normal_(0, math.sqrt(2. / n))
                    elif isinstance(mm, nn.BatchNorm2d):
                        mm.weight.data.fill_(1)
                        mm.bias.data.zero_()
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, img_path=True):
        downsample = None
        if img_path:
            inplanes = self.img_inplanes
        else:
            inplanes = self.flow_inplanes
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        if img_path:
            inplanes = planes * block.expansion
            self.img_inplanes = inplanes
        else:
            inplanes = planes * block.expansion
            self.flow_inplanes = inplanes
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, img1, flows):
        img1_fea = self.img_feature(img1)
        flow_fea = []
        for i in range(len(flows)):
            flow_fea.append(self.flow_feature(flows[i]))
        for i in range(len(flows)):
            flow_fea[i] = torch.unsqueeze(flow_fea[i], 2)
        flow_fea = torch.cat(flow_fea, dim=2)
        flow_fea = self.fusion4flow(flow_fea)
        flow_fea_list = torch.split(flow_fea, 1, dim=2)
        flow_fea_list = list(map(torch.squeeze, flow_fea_list))
        # # 当batch_size为1时候需要补回
        if len(flow_fea_list[0].size()) < 4:
            for i in range(len(flows)):
                flow_fea_list[i] = torch.unsqueeze(flow_fea_list[i], 0)
        flow_fea_list = torch.cat(flow_fea_list, dim=1)
        flow_fea_list_reduc = self.reductionFlowFea(flow_fea_list)

        # TODO 此次测试取消了光流向图像特征的流向，后续可以尝试拿回来
        flow_fea_list_reduc_detach = flow_fea_list_reduc.detach() # detach for not bachward the image loss to flow path
        fusioned_fea = torch.cat((img1_fea, flow_fea_list_reduc_detach), dim=1)
        fusioned_fea = self.fusion4img(fusioned_fea)
        # 直接把图像特征当做融合特征继续传播
        # fusioned_fea = img1_fea

        fusioned_fea = self.layer4fusion(fusioned_fea)
        flow_fea = self.layer4flow(flow_fea)

        fusioned_fea = self.avgpool2d(fusioned_fea)
        fusioned_fea = fusioned_fea.view(fusioned_fea.size(0), -1)
        flow_fea = self.avgpool3d(flow_fea)
        flow_fea = flow_fea.view(flow_fea.size(0), -1)

        result_image = self.fc_fusion(fusioned_fea)
        result_flow = self.fc_flow(flow_fea)

        # TODO 在测试的时候增大了光流的比重, 在测试的时候，使用各自计算各自的loss
        # return result1.div(2) + result2.div(2)
        if self.is_predict:
            return result_image / 2 + result_flow / 2
        else:
            return result_image / 2 + result_flow / 2

    def save(self,distance4flow, name=None, epoch=0, batch_size=128, eval=0,):
        '''
        保存模型，默认使用“模型名字+种类数+时间”作为文件名
        '''
        if name is None:
            prefix = 'checkpoints/epoch_' + str(epoch + 1) + '_' + str(eval) + '_' + 'dist=' + str(distance4flow) + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name

    def load(self, path, gpu=True):
        '''
        可加载指定路径的模型，针对是在多GPU训练的模型。
        '''
        if gpu:
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location='cpu')

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # print(k)
            name = k#[7:]  # remove `module.`
            new_state_dict[name] = v
        self.load_state_dict(new_state_dict)


if __name__ == "__main__":
    test = CamNet()
    dic = test.state_dict()
    print(len(dic))
    print(list(dic.keys()))
