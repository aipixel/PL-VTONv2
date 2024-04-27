import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from GRU import ConvGRU

class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)				# 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Decoder(nn.Module):
  def __init__(self, in_channels, middle_channels, out_channels):
    super(Decoder, self).__init__()
    self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    self.conv_relu = nn.Sequential(
        nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
        )
  def forward(self, x1, x2):
    x1 = self.up(x1)
    x1 = torch.cat((x1, x2), dim=1)
    x1 = self.conv_relu(x1)
    return x1

class STNNet(nn.Module):
    def __init__(self, input_channels=28):
        super(STNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20*61*45, 50)
        self.fc2 = nn.Linear(50, 6)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 60 * 44, 32),
            nn.ReLU(True),
            nn.Linear(32, 1 * 2)
        )
        self.fc_loc2 = nn.Sequential(
            nn.Linear(10 * 60 * 44, 32),
            nn.ReLU(True),
            nn.Linear(32, 1 * 2)
        )

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 60 * 44)
        theta1 = self.fc_loc(xs)
        theta2 = self.fc_loc2(xs)
        theta1 = theta1.view(-1, 2, 1)
        theta2 = theta2.view(-1, 2, 1)
        return theta1, theta2

    def forward(self, cloth, pose_map18, parse7_occ):
        x = torch.cat((cloth, pose_map18, parse7_occ), axis=1)    # [b, 3+18+7(28), 256, 192]
        theta1, theta2 = self.stn(x)
        theta1 = self.sigmoid(theta1) + 1
        theta2 = self.tanh(theta2)
        theta11 = theta1[:,0,:].unsqueeze(1)
        theta22 = theta1[:,1,:].unsqueeze(1)
        theta_zero = torch.zeros_like(theta11)
        theta_up = torch.cat((theta11, theta_zero), axis=2)
        theta_down = torch.cat((theta_zero, theta22), axis=2)
        theta1 = torch.cat((theta_up, theta_down), axis=1)
        theta = torch.cat((theta1, theta2), axis=2)
        return theta

class FlowModel_GRU(nn.Module):
    def __init__(self, input_A_channels=31):
        super(FlowModel_GRU, self).__init__()
        #------------#
        # get a flow #
        #------------#
        self.base_model = torchvision.models.resnet34(True)
        self.base_layers = list(self.base_model.children())
        self.encode1 = nn.Sequential(
            nn.Conv2d(input_A_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2],
        )   # [b, 64, 128, 96]
        self.encode2 = nn.Sequential(*self.base_layers[3:5]) # [b, 64, 64, 48]
        self.encode3 = self.base_layers[5]   # [b, 128, 32, 24]
        self.encode4 = self.base_layers[6]   # [b, 256, 16, 12]
        self.encode5 = self.base_layers[7]   # [b, 512, 8, 6]

        self.decode5 = Decoder(in_channels=512, middle_channels=256+256, out_channels=256)
        self.decode4 = Decoder(in_channels=256, middle_channels=128+128, out_channels=128)
        self.decode3 = Decoder(in_channels=128, middle_channels=64+64, out_channels=64)
        self.decode2 = Decoder(in_channels=64, middle_channels=64+64, out_channels=64)
        self.decode1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
            )
        self.conv_last = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)

        # flow op
        self.conv_flow1 = nn.Conv2d(256, 2, 1, 1)
        self.conv_flow2 = nn.Conv2d(128, 2, 1, 1)
        self.conv_flow3 = nn.Conv2d(64, 2, 1, 1)
        self.conv_flow4 = nn.Conv2d(64, 2, 1, 1)

        self.conv_gru = ConvGRU(input_size=(256, 192),
                                input_dim=2,
                                hidden_dim=[32, 64, 2],
                                kernel_size=(3,3),
                                num_layers=3,
                                dtype=torch.cuda.FloatTensor,
                                batch_first=True,
                                bias = True,
                                return_all_layers = False)
        self.tanh = nn.Tanh()

        self.se0 = SE_Block(ch_in=input_A_channels)
        self.se1 = SE_Block(ch_in=64)
        self.se2 = SE_Block(ch_in=64)
        self.se3 = SE_Block(ch_in=128)
        self.se4 = SE_Block(ch_in=256)
        self.se5 = SE_Block(ch_in=512)
      
    def forward(self, pre_cloth, pose_map18, parse7_occ, image_occ):
        input = torch.cat((pre_cloth, pose_map18, parse7_occ, image_occ), axis=1)    # [b, 3+18+7+3 (31), 256, 192]
        #------------#
        # get a flow #
        #------------#
        # input = self.se0(input)
        e1 = self.encode1(input)     # [b,64,128,96]
        # e1 = self.se1(e1)
        e2 = self.encode2(e1)        # [b,64,64,48]
        # e2 = self.se2(e2)
        e3 = self.encode3(e2)        # [b,128,32,24]
        # e3 = self.se3(e3)
        e4 = self.encode4(e3)        # [b,256,16,12]
        # e4 = self.se4(e4)
        f = self.encode5(e4)         # [b,512,8,6]
        # f = self.se5(f)

        d4 = self.decode5(f, e4)     # [b,256,16,12]  --->  flow1
        d3 = self.decode4(d4, e3)    # [b,128,32,24]  --->  flow2
        d2 = self.decode3(d3, e2)    # [b,64,64,48]   --->  flow3
        d1 = self.decode2(d2, e1)    # [b,64,128,96]  --->  flow4
        d0 = self.decode1(d1)        # [b,64,256,192] 
        flow = self.conv_last(d0)    # [b,2,256,192]  --->  flow5

        flow1 = torch.nn.functional.interpolate(d4, scale_factor=16, mode='bilinear', align_corners=True)   # [b,256,256,192]
        flow1 = self.conv_flow1(flow1)  # [b,2,256,192]

        flow2 = torch.nn.functional.interpolate(d3, scale_factor=8, mode='bilinear', align_corners=True)   # [b,128,256,192]
        flow2 = self.conv_flow2(flow2)  # [b,2,256,192]

        flow3 = torch.nn.functional.interpolate(d2, scale_factor=4, mode='bilinear', align_corners=True)   # [b,64,256,192]
        flow3 = self.conv_flow3(flow3)  # [b,2,256,192]

        flow4 = torch.nn.functional.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=True)   # [b,64,256,192]
        flow4 = self.conv_flow4(flow4)  # [b,2,256,192]

        flow5 = flow

        flow_all = torch.cat((flow1.unsqueeze(1), flow2.unsqueeze(1), flow3.unsqueeze(1), flow4.unsqueeze(1), flow5.unsqueeze(1)), axis=1)  # [b, 5, 2, 256, 192]
        layer_output_list, last_state_list = self.conv_gru(flow_all)

        gru_flow = last_state_list[0][0]            # (b, 2, 256, 192)
        flow_all = flow_all.permute(0,1,3,4,2)      # [b,5,256,192,2]
        gru_flow = gru_flow.permute(0,2,3,1)        # [b,256,192,2]
        gru_flow = self.tanh(gru_flow)

        gridY = torch.linspace(-1, 1, steps = 256).view(1, -1, 1, 1).expand(1, 256, 192, 1)
        gridX = torch.linspace(-1, 1, steps = 192).view(1, 1, -1, 1).expand(1, 256, 192, 1)
        grid = torch.cat((gridX, gridY), dim=3).type(gru_flow.type())

        grid = torch.repeat_interleave(grid, repeats=gru_flow.shape[0], dim=0)
        gru_flow = torch.clamp(gru_flow + grid, min=-1, max=1)

        #---------------------------------#
        # get the result through the flow #
        #---------------------------------#
        warp_cloth = F.grid_sample(pre_cloth, gru_flow, mode='bilinear', padding_mode='border')
        
        return gru_flow, warp_cloth

class ParseModel(nn.Module):
    def __init__(self, input_channels=32):
        super(ParseModel, self).__init__()
        self.base_model = torchvision.models.resnet34(True)
        self.base_layers = list(self.base_model.children())
        self.encode1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2],
        )   # [b, 64, 128, 96]
        self.encode2 = nn.Sequential(*self.base_layers[3:5])  # [b, 64, 64, 48]
        self.encode3 = self.base_layers[5]                    # [b, 128, 32, 24]
        self.encode4 = self.base_layers[6]                    # [b, 256, 16, 12]
        self.encode5 = self.base_layers[7]                    # [b, 512, 8, 6]

        self.decode5 = Decoder(in_channels=512, middle_channels=256+256, out_channels=256)
        self.decode4 = Decoder(in_channels=256, middle_channels=128+128, out_channels=128)
        self.decode3 = Decoder(in_channels=128, middle_channels=64+64, out_channels=64)
        self.decode2 = Decoder(in_channels=64, middle_channels=64+64, out_channels=64)
        self.decode1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
            )
        self.conv_last = nn.Conv2d(in_channels=64, out_channels=7, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.se0 = SE_Block(ch_in=input_channels)
        self.se1 = SE_Block(ch_in=64)
        self.se2 = SE_Block(ch_in=64)
        self.se3 = SE_Block(ch_in=128)
        self.se4 = SE_Block(ch_in=256)
        self.se5 = SE_Block(ch_in=512)

    def forward(self, warp_cloth, pose_map18, parse7_occ, image_occ, mis_parse):
        input = torch.cat((warp_cloth, pose_map18, parse7_occ, image_occ, mis_parse), axis=1)    # [b, 3+18+7+3+1 (32), 256, 192]
        input = self.se0(input)
        e1 = self.encode1(input)     # [b,64,128,96]
        e1 = self.se1(e1)
        e2 = self.encode2(e1)        # [b,64,64,48]
        e2 = self.se2(e2)
        e3 = self.encode3(e2)        # [b,128,32,24]
        e3 = self.se3(e3)
        e4 = self.encode4(e3)        # [b,256,16,12]
        e4 = self.se4(e4)
        f = self.encode5(e4)         # [b,512,8,6]
        f = self.se5(f)

        d4 = self.decode5(f, e4)     # [b,256,16,12]  
        d3 = self.decode4(d4, e3)    # [b,128,32,24]  
        d2 = self.decode3(d3, e2)    # [b,64,64,48]   
        d1 = self.decode2(d2, e1)    # [b,64,128,96]  
        d0 = self.decode1(d1)       
        parse = self.conv_last(d0)    # [b,20,256,192]
        parse = self.sigmoid(parse)   # [b,20,256,192]
        return parse

# coarse去掉肢体部分（结果只保留肢体外的部分），因为mask遮挡了几乎全部的肢体信息，模型只能靠猜想 (我们没有设置判别器，会增加计算成本和模型大小)，没有所依据的辅助信息 / 或者loss不关注
class TryOnModel(nn.Module):
    def __init__(self, input_channels=47):
        super(TryOnModel, self).__init__()
        self.base_model = torchvision.models.resnet34(True)
        self.base_layers = list(self.base_model.children())
        self.encode1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2],
        )   # [b, 64, 128, 96]
        self.encode2 = nn.Sequential(*self.base_layers[3:5]) # [b, 64, 64, 48]
        self.encode3 = self.base_layers[5]   # [b, 128, 32, 24]
        self.encode4 = self.base_layers[6]   # [b, 256, 16, 12]
        self.encode5 = self.base_layers[7]   # [b, 512, 8, 6]

        self.decode5 = Decoder(in_channels=512, middle_channels=256+256, out_channels=256)
        self.decode4 = Decoder(in_channels=256, middle_channels=128+128, out_channels=128)
        self.decode3 = Decoder(in_channels=128, middle_channels=64+64, out_channels=64)
        self.decode2 = Decoder(in_channels=64, middle_channels=64+64, out_channels=64)
        self.decode1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
            )
        self.conv_last = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, warp_cloth, pose_map18, parse7_t, img_preserve):
        input = torch.cat((warp_cloth, pose_map18, parse7_t, img_preserve), axis=1)    # [b, 3+18+20+3+3 (47), 256, 192]
        #------------#
        # get a flow #
        #------------#
        e1 = self.encode1(input)     # [b,64,128,96]
        e2 = self.encode2(e1)        # [b,64,64,48]
        e3 = self.encode3(e2)        # [b,128,32,24]
        e4 = self.encode4(e3)        # [b,256,16,12]
        f = self.encode5(e4)         # [b,512,8,6]

        d4 = self.decode5(f, e4)     # [b,256,16,12]  
        d3 = self.decode4(d4, e3)    # [b,128,32,24]  
        d2 = self.decode3(d3, e2)    # [b,64,64,48]   
        d1 = self.decode2(d2, e1)    # [b,64,128,96]  
        d0 = self.decode1(d1)       
        try_on = self.conv_last(d0)    # [b,3,256,192]
        try_on = self.sigmoid(try_on)
        return try_on

# https://github.com/ignacio-rocco/cnngeometric_pytorch/blob/master/model/cnn_geometric_model.py
# I. Rocco, R. Arandjelović and J. Sivic. Convolutional neural network architecture for geometric matching. CVPR 2017
class FeatureCorrelation(nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()
    
    def forward(self, feature_A, feature_B):
        b,c,h,w = feature_A.size()
        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2,3).contiguous().view(b,c,h*w)
        feature_B = feature_B.view(b,c,h*w).transpose(1,2)
        # perform matrix mult.
        feature_mul = torch.bmm(feature_B,feature_A)
        '''
        torch.bmm(input, mat2, out=None) → Tensor
        Performs a batch matrix-matrix product of matrices stored in input and mat2.
        input and mat2 must be 3-D tensors each containing the same number of matrices.
        If input is a (b \times n \times m)(b×n×m) tensor, mat2 is a (b \times m \times p)(b×m×p) tensor,
        out will be a (b \times n \times p)(b×n×p) tensor.
​	    '''
        correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
        return correlation_tensor

# 肢体信息融合的时候可以借鉴VITON的feature matching
class LimbModel(nn.Module):
    def __init__(self, input_channels=28, limb_channels=192):
        super(LimbModel, self).__init__()
        self.base_model = torchvision.models.resnet34(True)
        self.base_layers = list(self.base_model.children())
        self.encode1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2],
        )   # [b, 64, 128, 96]
        self.encode2 = nn.Sequential(*self.base_layers[3:5]) # [b, 64, 64, 48]
        self.encode3 = self.base_layers[5]   # [b, 128, 32, 24]
        self.encode4 = self.base_layers[6]   # [b, 256, 16, 12]
        self.encode5 = self.base_layers[7]   # [b, 512, 8, 6]

        self.decode5 = Decoder(in_channels=1024, middle_channels=512+256, out_channels=512)
        self.decode4 = Decoder(in_channels=512, middle_channels=128+128, out_channels=128)
        self.decode3 = Decoder(in_channels=128, middle_channels=64+64, out_channels=64)
        self.decode2 = Decoder(in_channels=64, middle_channels=64+64, out_channels=64)
        self.decode1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
            )
        self.conv_last = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.limb_conv1 = nn.Sequential(
            nn.Conv2d(limb_channels,256,3,2,1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.limb_conv2 = nn.Sequential(
            nn.Conv2d(256,512,3,2,1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.correlation = FeatureCorrelation()
        self.add_channel = nn.Sequential(
            nn.Conv2d(48,1024,1,1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True)
        )

    def forward(self, limb, try_on_coarse, pose_map18, parse7_t):
        limb_feature1 = self.limb_conv1(limb)             # [b, 256, 16, 12]
        limb_feature2 = self.limb_conv2(limb_feature1)    # [b, 512, 8, 6]

        input = torch.cat((try_on_coarse, pose_map18, parse7_t), axis=1)    # [b, 3+18+7 (28), 256, 192]
        #------------#
        # get a flow #
        #------------#
        e1 = self.encode1(input)     # [b,64,128,96]
        e2 = self.encode2(e1)        # [b,64,64,48]
        e3 = self.encode3(e2)        # [b,128,32,24]
        e4 = self.encode4(e3)        # [b,256,16,12]
        f = self.encode5(e4)         # [b,512,8,6]

        # e4 = torch.cat((e4, limb_feature1), axis=1)  # [b,512,16,12]
        # f = torch.cat((f, limb_feature2), axis=1)    # [b,1024,8,6]
        f = self.correlation(f, limb_feature2)
        f = self.add_channel(f)

        d4 = self.decode5(f, e4)     # [b,512,16,12] 
        d3 = self.decode4(d4, e3)    # [b,128,32,24]  
        d2 = self.decode3(d3, e2)    # [b,64,64,48]   
        d1 = self.decode2(d2, e1)    # [b,64,128,96]  
        d0 = self.decode1(d1)       

        try_on_fine = self.conv_last(d0)    # [b,3,256,192]
        try_on_fine = self.sigmoid(try_on_fine)
        return try_on_fine

class Network(nn.Module):
    def __init__(self, istrain=True): 
        super(Network, self).__init__()
        self.istrain = istrain
        self.try_on_model = TryOnModel(input_channels=31)
        self.limb_model = LimbModel(input_channels=28, limb_channels=192)
      
    def forward(self, limb, warp_cloth, pose_map18, parse7_t, img_preserve):
        tmp = self.try_on_model(warp_cloth, pose_map18, parse7_t, img_preserve)        # [b, 3+18+7+3, 256, 192]
        try_on = self.limb_model(limb, tmp, pose_map18, parse7_t)                 # [b, 3+18+7+3, 256, 192]
        return tmp, try_on

if __name__ == '__main__':
    cloth = torch.from_numpy(np.zeros((2,3,256,192)).astype(np.float32)).cuda()
    pose_map18 = torch.from_numpy(np.zeros((2,18,256,192)).astype(np.float32)).cuda()
    parse7_occ = torch.from_numpy(np.zeros((2,7,256,192)).astype(np.float32)).cuda()
    # image_occ = torch.from_numpy(np.zeros((2,3,256,192)).astype(np.float32)).cuda()

    # model = PAModel().cuda()
    # res = model(cloth, pose_map18, parse7_occ, image_occ)
    # print("res:", res.shape)
    # cloth = torch.from_numpy(np.zeros((1,28,256,192)).astype(np.float32)).cuda()
    # img = torch.from_numpy(np.array(Image.open("1.jpg")).transpose(2,0,1)).unsqueeze(0).cuda()/255
    model = STNNet2().cuda()
    res = model(cloth, pose_map18, parse7_occ)





