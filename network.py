import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear, MetaLayerNorm)
import numpy as np

def calcHW(height, width, kernel_size, stride, padding=0):
    height = (height - kernel_size + 2 * padding)//stride + 1 
    width = (width - kernel_size + 2 * padding)//stride + 1 
    return height, width

class SpatialSoftmax(nn.Module):
    """
    C, W, H => C*2 # for each channel, get good point [x,y]
    """
    def __init__(self, n_rows, n_cols):
        super(SpatialSoftmax, self).__init__()

        x_map = np.zeros((n_rows, n_cols))
        y_map = np.zeros((n_rows, n_cols))
        for i in range(n_rows):
            for j in range(n_cols):
                x_map[i,j] = (i - n_rows / 2) / n_rows
                y_map[i,j] = (j - n_cols / 2) / n_cols
        x_map = torch.from_numpy(np.array(x_map.reshape((-1)), np.float32)).cuda()
        y_map = torch.from_numpy(np.array(y_map.reshape((-1)), np.float32)).cuda()
        
        self.x_map = x_map
        self.y_map = y_map
        
    def forward(self, x): 
        x = x.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3]) # batch, C, W*H
        x = F.softmax(x, 2) # batch, C, W*H
        fp_x = torch.matmul(x, self.x_map) # batch, C
        fp_y = torch.matmul(x, self.y_map) # batch, C
        x = torch.cat((fp_x, fp_y), 1)
        return x # batch, C*2

class MIL(MetaModule):
    def __init__(self):
        super(MIL, self).__init__()
        h1, w1 = 200, 200
        h2, w2 = calcHW(h1, w1, kernel_size=8, stride=2)
        h3, w3 = calcHW(h2, w2, kernel_size=4, stride=2)
        h4, w4 = calcHW(h3, w3, kernel_size=4, stride=2)

        self.features = MetaSequential(
            MetaConv2d(in_channels=3, out_channels=32, kernel_size=8, stride=2),
            MetaLayerNorm([32, h2, w2]),
            nn.ReLU(),

            MetaConv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            MetaLayerNorm([64, h3, w3]),
            nn.ReLU(),

            MetaConv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2),
            MetaLayerNorm([64, h4, w4]),
            nn.ReLU(),

            SpatialSoftmax(h4, w4)
        )

        self.policy = MetaSequential(
            MetaLinear(2*64 + 3, 128),
            nn.ReLU(),
            MetaLinear(128, 128),
            nn.ReLU(),            
            MetaLinear(128, 128),
            nn.ReLU(),            
            MetaLinear(128, 4),
        )

    def forward(self, image, robot_config, params=None):
        features = self.features(image, params=self.get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        features = torch.cat((features, robot_config), -1) 
        action = self.policy(features, params=self.get_subdict(params, 'policy'))
        return action