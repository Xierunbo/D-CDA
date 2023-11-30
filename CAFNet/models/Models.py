import torch.nn as nn
import torch
from .layer import MultiSpectralAttentionLayer
from test_bottleneck import ACmix
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True): 
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out




class RepresentativeBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(RepresentativeBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.num_features = num_features
        ### weights for affine transformation in BatchNorm ###
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
            self.bias = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
            self.weight.data.fill_(1)
            self.bias.data.fill_(0)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        ### weights for centering calibration ###        
        self.center_weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.center_weight.data.fill_(0)
        ### weights for scaling calibration ###            
        self.scale_weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.scale_bias = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.scale_weight.data.fill_(0)
        self.scale_bias.data.fill_(1)
        ### calculate statistics ###
        self.stas = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, input):
        self._check_input_dim(input)

        ####### centering calibration begin ####### 
        input += self.center_weight.view(1,self.num_features,1,1)*self.stas(input)
        ####### centering calibration end ####### 

        ####### BatchNorm begin #######         
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else: 
                    exponential_average_factor = self.momentum
        output = F.batch_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        ####### BatchNorm end #######

        ####### scaling calibration begin ####### 
        scale_factor = torch.sigmoid(self.scale_weight*self.stas(output)+self.scale_bias)
        ####### scaling calibration end ####### 
        if self.affine:
            return self.weight*scale_factor*output + self.bias
        else:
            return scale_factor*output
            
class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output


class up(nn.Module):
    def __init__(self, in_ch, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):

        x = self.up(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)



class SNUNet_ECAM(nn.Module):
    # SNUNet-CD with ECAM
    def __init__(self, in_ch=3, out_ch=2):
        super(SNUNet_ECAM, self).__init__()
        torch.nn.Module.dump_patches = True
        c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
        self.att_32 = CoordAtt(32,32,reduction=32)
        self.att_64 = CoordAtt(64,64,reduction=32)
        self.att_128 = CoordAtt(128,128,reduction=32)
        self.att_256 = CoordAtt(256,256,reduction=32)
        n1 = 32     # the initial number of channels of feature map
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.Up1_0 = up(filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.Up2_0 = up(filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.Up3_0 = up(filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
        self.Up4_0 = up(filters[4])

        self.conv0_1 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.Up1_1 = up(filters[1])
        self.conv2_1 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])
        self.Up2_1 = up(filters[2])
        self.conv3_1 = conv_block_nested(filters[3] * 2 + filters[4], filters[3], filters[3])
        self.Up3_1 = up(filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])
        self.Up1_2 = up(filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 3 + filters[3], filters[2], filters[2])
        self.Up2_2 = up(filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 4 + filters[2], filters[1], filters[1])
        self.Up1_3 = up(filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 5 + filters[1], filters[0], filters[0])

        self.ca = ChannelAttention(filters[0] * 4, ratio=16)
        self.ca1 = ChannelAttention(filters[0], ratio=16 // 4)

        self.conv_final = nn.Conv2d(filters[0] * 4, out_ch, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, xA, xB):
        '''xA'''
        x0_0A = self.conv0_0(xA)
        out_0_0A = self.att_32(x0_0A)
        x1_0A = self.conv1_0(self.pool(out_0_0A))
        out_1_0A = self.att_64(x1_0A)
        x2_0A = self.conv2_0(self.pool(out_1_0A))
        out_2_0A = self.att_128(x2_0A)
        x3_0A = self.conv3_0(self.pool(out_2_0A))
        # x4_0A = self.conv4_0(self.pool(x3_0A))
        '''xB'''
        x0_0B = self.conv0_0(xB)
        out_0_0B = self.att_32(x0_0B)
        x1_0B = self.conv1_0(self.pool(out_0_0B))
        out_1_0B = self.att_64(x1_0B)
        x2_0B = self.conv2_0(self.pool(out_1_0B))
        out_2_0B = self.att_128(x2_0B)
        x3_0B = self.conv3_0(self.pool(out_2_0B))
        x4_0B = self.conv4_0(self.pool(x3_0B))

        x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))
        # out_1_1 = self.att_64(x1_1)
        x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1(x1_1)], 1))

        
        x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
        # out_2_1 = self.att_128(x2_1)
        x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(x2_1)], 1))
        # out_1_2 = self.att_64(x1_2)
        x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, self.Up4_0(x4_0B)], 1))
        # out_3_1 = self.att_256(x3_1)
        x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(x3_1)], 1))
        # out_2_2 = self.att_128(x2_2)
        x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))

        # out_1_3 = self.att_64(x1_3)

        x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))

        out = torch.cat([x0_1, x0_2, x0_3, x0_4], 1)

        intra = torch.sum(torch.stack((x0_1, x0_2, x0_3, x0_4)), dim=0)
        ca1 = self.ca1(intra)
        out = self.ca(out) * (out + ca1.repeat(1, 4, 1, 1))
        out = self.conv_final(out)

        return (out, )


class Siam_NestedUNet_Conc(nn.Module):
    # SNUNet-CD without Attention
    def __init__(self, in_ch=3, out_ch=2):
        super(Siam_NestedUNet_Conc, self).__init__()
        torch.nn.Module.dump_patches = True
        n1 = 32     # the initial number of channels of feature map
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.att = MultiSpectralAttentionLayer(planes * 4, c2wh[planes], c2wh[planes],  reduction=reduction, freq_sel_method = 'top16')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.Up1_0 = up(filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.Up2_0 = up(filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.Up3_0 = up(filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
        self.Up4_0 = up(filters[4])

        self.conv0_1 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.Up1_1 = up(filters[1])
        self.conv2_1 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])
        self.Up2_1 = up(filters[2])
        self.conv3_1 = conv_block_nested(filters[3] * 2 + filters[4], filters[3], filters[3])
        self.Up3_1 = up(filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])
        self.Up1_2 = up(filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 3 + filters[3], filters[2], filters[2])
        self.Up2_2 = up(filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 4 + filters[2], filters[1], filters[1])
        self.Up1_3 = up(filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 5 + filters[1], filters[0], filters[0])

        self.final1 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.final2 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.final3 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.final4 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.conv_final = nn.Conv2d(out_ch * 4, out_ch, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, xA, xB):
        '''xA'''
        x0_0A = self.conv0_0(xA)
        x1_0A = self.conv1_0(self.pool(x0_0A))
        x2_0A = self.conv2_0(self.pool(x1_0A))
        x3_0A = self.conv3_0(self.pool(x2_0A))
        # x4_0A = self.conv4_0(self.pool(x3_0A))
        '''xB'''
        x0_0B = self.conv0_0(xB)
        x1_0B = self.conv1_0(self.pool(x0_0B))
        x2_0B = self.conv2_0(self.pool(x1_0B))
        x3_0B = self.conv3_0(self.pool(x2_0B))
        x4_0B = self.conv4_0(self.pool(x3_0B))

        x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1(x1_1)], 1))


        x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, self.Up4_0(x4_0B)], 1))
        out = self.att(x3_1)
        x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(out)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))


        output1 = self.final1(x0_1)
        output2 = self.final2(x0_2)
        output3 = self.final3(x0_3)
        output4 = self.final4(x0_4)
        output = self.conv_final(torch.cat([output1, output2, output3, output4], 1))
        return (output1, output2, output3, output4, output)
