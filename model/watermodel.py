import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model():
	return MainModel()


class DoubleConv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(DoubleConv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False, padding_mode='reflect'),
			nn.GroupNorm(num_channels=out_ch, num_groups=8, affine=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False, padding_mode='reflect'),
			nn.GroupNorm(num_channels=out_ch, num_groups=8, affine=True),
			nn.ReLU(inplace=True)
		)
	def forward(self, x):
		x = self.conv(x)
		return x


class InDoubleConv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(InDoubleConv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 9, stride=4, padding=4, bias=False, padding_mode='reflect'),
			nn.GroupNorm(num_channels=out_ch, num_groups=8, affine=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False, padding_mode='reflect'),
			nn.GroupNorm(num_channels=out_ch, num_groups=8, affine=True),
			nn.ReLU(inplace=True)
		)
	def forward(self, x):
		x = self.conv(x)
		return x


class InConv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(InConv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(1, 64, 7, stride = 4, padding=3,  bias=False, padding_mode='reflect'),
			nn.GroupNorm(num_channels=64, num_groups=8, affine=True),
			nn.ReLU(inplace=True)
		)
		self.convf = nn.Sequential(
			nn.Conv2d(64, 64, 3, padding=1, bias=False, padding_mode='reflect'),
			nn.GroupNorm(num_channels=64, num_groups=8, affine=True),
			nn.ReLU(inplace=False)
		)
	def forward(self, x):
		R = x[:, 0:1, :, :]
		G = x[:, 1:2, :, :]
		B = x[:, 2:3, :, :]
		xR = torch.unsqueeze(self.conv(R), 1)
		xG = torch.unsqueeze(self.conv(G), 1)
		xB = torch.unsqueeze(self.conv(B), 1)
		x = torch.cat([xR, xG, xB], 1)
		x, _ = torch.min(x, dim=1)
		return self.convf(x)

class SKConv(nn.Module):
    def __init__(self, outfeatures=64, infeatures=1, M=4 ,L=32):

        super(SKConv, self).__init__()
        self.M = M
        self.convs = nn.ModuleList([])
        in_conv = InConv(in_ch=infeatures, out_ch=outfeatures)
        for i in range(M):
        	if i==0:
        		self.convs.append(in_conv)
        	else:
	            self.convs.append(nn.Sequential(
	            	nn.Upsample(scale_factor=1/(2**i), mode='bilinear', align_corners=True),
	                in_conv,
	                nn.Upsample(scale_factor=2**i, mode='bilinear', align_corners=True)
	            ))
        self.fc = nn.Linear(outfeatures, L)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(L, outfeatures)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v

class estimation(nn.Module):
	def __init__(self):
		super(estimation, self).__init__()
				
		self.InConv = SKConv(outfeatures=64, infeatures=1, M=3 ,L=32)

		self.convt = DoubleConv(64, 64)
		self.OutConv = nn.Conv2d(64, 3, 3, padding = 1, stride=1, bias=False, padding_mode='reflect')
		self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

		self.conv1 = InDoubleConv(3, 64)
		self.conv1X = InDoubleConv(3, 64)
		self.conv2 = DoubleConv(64, 64)
		self.maxpool = nn.MaxPool2d(15, 7)
		
		self.pool = nn.AdaptiveAvgPool2d(1)
		self.dense = nn.Linear(64, 3, bias=False)
		
		
	def forward(self, x):

		xmin = self.InConv(x)

		trans = self.conv1X(x)
		trans = torch.mul(trans, xmin)
		trans = self.OutConv(self.up(self.convt(xmin)))
		trans = torch.sigmoid(trans) + 1e-12

		atm = self.conv1(x)
		atm = torch.mul(atm, xmin)
		atm = self.pool(self.conv2(self.maxpool(atm)))
		atm = atm.view(-1, 64)
		atm = torch.sigmoid(self.dense(atm))
		
		return trans, atm


class MainModel(nn.Module):
	def __init__(self):
		super().__init__()
				
		self.estimation = estimation()		
		
	def forward(self, x):

		trans, atm = self.estimation(x)

		atm = torch.unsqueeze(torch.unsqueeze(atm, 2), 2)
		atm = atm.expand_as(x)

		out = (x - (1 - trans)*atm)/trans
		
		return trans, atm, out



