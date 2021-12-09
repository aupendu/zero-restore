import torch
import torch.nn as nn
import torchvision
import imageio
import numpy as np

import model
import argparse

import os
import math
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1)
random.seed(1)

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv2d') != -1: #0.02
		m.weight.data.normal_(0.0, 0.001)
	if classname.find('Linear') != -1: #0.02
		m.weight.data.normal_(0.0, 0.001)

parser = argparse.ArgumentParser(description='Lowlight Image Enhancement')
parser.add_argument('--TestFolderPath', type=str, default='data/Lowlight/data/LOLTest', help='Lowlight Image folder name') 
parser.add_argument('--SavePath', type=str, default='lowresults/LOLTest', help='SavePath Name')
args = parser.parse_args()

def _np2Tensor(img):
	np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
	tensor = torch.from_numpy(np_transpose).float()
	return torch.unsqueeze(tensor, 0)

def psnr(imgS, imgG):
	diff = imgS - imgG
	mse = diff.pow(2).mean()
	return -10 * math.log10(mse)

class I_TV(nn.Module):
	def __init__(self):
		super(I_TV,self).__init__()
		pass

	def forward(self,x):
		batch_size, h_x, w_x = x.size()[0], x.size()[2], x.size()[3]
		count_h, count_w = (h_x-1) * w_x, h_x * (w_x - 1)
		h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
		w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
		return (h_tv/count_h+w_tv/count_w)/batch_size


def randomSelect(_min, _max, _div):
	px = random.randint(_min, _max)
	px = float(px)/_div
	return px

def _augment(_image):
	it = random.randint(0, 7)
	if it==1: _image = _image.rot90(1, [2, 3])
	if it==2: _image = _image.rot90(2, [2, 3])
	if it==3: _image = _image.rot90(3, [2, 3])
	if it==4: _image = _image.flip(2).rot90(1, [2, 3])
	if it==5: _image = _image.flip(3).rot90(1, [2, 3])
	if it==6: _image = _image.flip(2)
	if it==7: _image = _image.flip(3)
	return _image



_img_TV = I_TV()
itr_no = 10000

def test(args):
	InputImages = os.listdir(args.TestFolderPath+'/Input/')

	os.makedirs(args.SavePath+'/', exist_ok=True)

	for i in range(len(InputImages)):
		print("Images Processed: %d/ %d  \r" % (i+1, len(InputImages)))

		_model = model.Model('lowmodel')
		_model.apply(weights_init)
		_model.cuda()
		optimizer = torch.optim.Adam(_model.parameters(), lr=1e-3, betas=(0.99, 0.999), eps=1e-08, weight_decay=1e-2)

		Input = imageio.imread(args.TestFolderPath+'/Input/'+InputImages[i])
		Input = _np2Tensor(Input)
		Input = (Input/255.).cuda()

		H, W = Input.shape[2], Input.shape[3]
		H, W = H - H%32, W - W%32
		Input = Input[:, :, 0:H, 0:W]


		for k in tqdm(range(itr_no), desc="Loading..."):
			Inputmage = _augment(Input)

			_model.train()
			optimizer.zero_grad()
			trans_map, atm_map, HazefreeImage = _model(Inputmage)

			px = 0.9
			_trans_map = px
			InputmageX = Inputmage*_trans_map + (1 - _trans_map)*atm_map
			trans_mapX, atm_mapX, HazefreeImageX = _model(InputmageX)
	

			otensor = torch.ones(HazefreeImage[:, 0:1, :, :].shape).cuda()
			ztensor = torch.zeros(HazefreeImage[:, 0:1, :, :].shape).cuda()

			lossMxR = torch.sum(torch.max(HazefreeImage[:, 0:1, :, :], otensor)) + torch.sum(torch.max(HazefreeImageX[:, 0:1, :, :], otensor)) - 2*torch.sum(otensor)
			lossMxG = torch.sum(torch.max(HazefreeImage[:, 1:2, :, :], otensor)) + torch.sum(torch.max(HazefreeImageX[:, 1:2, :, :], otensor)) - 2*torch.sum(otensor)
			lossMxB = torch.sum(torch.max(HazefreeImage[:, 2:3, :, :], otensor)) + torch.sum(torch.max(HazefreeImageX[:, 2:3, :, :], otensor)) - 2*torch.sum(otensor)
			lossMx = lossMxR + lossMxG + 10*lossMxB
			

			lossMnR = -torch.sum(torch.min(HazefreeImage[:, 0:1, :, :], ztensor)) - torch.sum(torch.min(HazefreeImageX[:, 0:1, :, :], ztensor)) 
			lossMnG = -torch.sum(torch.min(HazefreeImage[:, 1:2, :, :], ztensor)) - torch.sum(torch.min(HazefreeImageX[:, 1:2, :, :], ztensor)) 
			lossMnB = -torch.sum(torch.min(HazefreeImage[:, 2:3, :, :], ztensor)) - torch.sum(torch.min(HazefreeImageX[:, 2:3, :, :], ztensor)) 
			lossMn = lossMnR + lossMnG + 10*lossMnB

			lossT = torch.sum(torch.abs(trans_mapX - px*trans_map)) 
			lossA =  torch.sum((atm_map - atm_mapX)**2) 

			lossTV = _img_TV(HazefreeImage)

			loss =  lossT + lossA + 0.001*lossMx + 0.01*lossMn + 0.001*lossTV 

			loss.backward()
			optimizer.step()


		_model.eval()
		with torch.no_grad():
			_trans, _atm, _GT = _model(Input)
			_GT = torch.clamp(_GT, 0, 1)

		torchvision.utils.save_image(_GT, args.SavePath+'/'+InputImages[i][:-4]+'.png')



test(args) 