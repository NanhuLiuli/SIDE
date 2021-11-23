from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

class PointNetfeat_strAM(nn.Module):
    def __init__(self, input_c):
        super(PointNetfeat_strAM, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_c, 256, 1)
        self.conv2 = torch.nn.Conv1d(256, 512, 1)
        self.conv3 = torch.nn.Conv1d(512, 1024, 1)
        self.conv4 = torch.nn.Conv1d(1024, 1024, 1)
        # self.isp=nn.Sequential(
        #     nn.Linear(1000, 1000),
        #     nn.BatchNorm1d(1000)
        #     )
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.strAM_2D=torch.nn.Conv2d(1024,1024,3,1,1)

    def forward(self, x, res):
        x = F.relu(self.bn1((self.conv1(x))))
        x = F.relu((self.bn2(self.conv2(x))))
        x = self.bn3(self.conv3(x))

        isp_cube=x.view(x.size(0),x.size(1), res, res, res)
        isp=torch.mean(isp_cube,dim=3)
        isp=torch.sigmoid(self.strAM_2D(isp)).unsqueeze(3)
        isp=isp.expand_as(isp_cube)
        isp=isp*isp_cube
        isp=isp.view(x.size(0), x.size(1), res*res*res)

        x = F.relu(self.bn4(self.conv4(isp)))+x
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        return x



class PointNetDetector(nn.Module):
    def __init__(self, input_c):
        super(PointNetDetector, self).__init__()
        self.feat_all = PointNetfeat_strAM(input_c)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        # self.dim = nn.Linear(256, 3)
        # self.pos = nn.Linear(256, 3)
        # self.ori = nn.Linear(256, 1)
        self.depth = nn.Linear(256, 1)
        # self.conf = nn.Linear(256, 1)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    
    def forward(self, input_data, res):
        xa = self.feat_all(input_data, res)
        x = self.fc1(xa)
        x = x if x.shape[0] <= 1 else self.bn1(x)
        x = F.relu(x)

        
        x = self.dropout(self.fc2(x))
        x = x if x.shape[0] <= 1 else self.bn2(x)
        x = F.relu(x)
        depth = self.depth(x)

        # x = F.relu(self.bn1(self.fc1(xa)))
        # x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        # pos = self.pos(x)
        # dim = self.dim(x)
        # ori = self.ori(x)
        # conf = self.conf(x)
        # x = torch.cat([pos,dim,ori,conf],dim=1)
        # x = torch.cat([pos, dim, ori],dim=1)

        return depth



