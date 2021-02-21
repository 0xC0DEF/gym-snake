import torch as tc
import torch.nn as nn
import torch.nn.init as init
from config import GameOption
from core import DEVICE
	
class FFC(nn.Module):
	def __init__(self, chn_in, chn_out, ker_sz):
		super().__init__()
		self.c=nn.Conv2d(chn_in,chn_out,ker_sz,padding=ker_sz//2,padding_mode="circular",bias=False)
		#self.d=nn.Dropout2d(0.5)
		#self.b=nn.BatchNorm2d(chn_out)
		self.a=nn.LeakyReLU(0.1)

	def forward(self, x):
		return self.a(self.c(x))
	
class CTR(nn.Module):
	def __init__(self, N_in, N_out, drop_out=False):
		super().__init__()
		self.l=nn.Linear(N_in,N_out)
		self.drop_out=drop_out
		if drop_out:
			self.d=nn.Dropout(0.5)
		self.a=nn.LeakyReLU(0.1)

	def forward(self, x):
		x=self.l(x)
		if self.drop_out:
			x=self.d(x)
		return self.a(x)

class NN(nn.Module):
	def __init__(self):
		super(NN,self).__init__()
		self.chn_in=8
		self.chn_out=32
		self.ch_adjuster=nn.Conv2d(2,self.chn_in,1,padding=0,bias=False)
		self.ffcs=nn.ModuleList([
			FFC(self.chn_in,12,3),
			#FFC(12,12,3),
			FFC(12,16,3),
			FFC(16,20,3),
			FFC(20,24,3),
			FFC(24,self.chn_out,3)])
		
		self.dense=nn.Sequential(
			CTR(self.chn_out*GameOption.ROW*GameOption.COL+1,512),
			CTR(512,1024),
			CTR(1024,256),
			CTR(256,128),
			CTR(128,4),
		)
		def weight_init(x):
			if type(x)==nn.Linear or type(x)==nn.Conv2d:
				init.xavier_uniform_(x.weight.data)
				if x.bias != None:
					init.zeros_(x.bias)
		self.ffcs.apply(weight_init)
		self.dense.apply(weight_init)
				
	def forward(self,x,starvation):
		#x=x.reshape((-1,self.chn_in,GameOption.ROW,GameOption.COL))
		x=x.reshape((-1,2,GameOption.ROW,GameOption.COL))
		xa=self.ch_adjuster(x)
		x=xa
		for ffc in self.ffcs:
			x=ffc(x)#+xa #residual training
		x=x.reshape(-1,self.chn_out*GameOption.ROW*GameOption.COL)
		starvation=starvation.reshape(-1,1)
		x=tc.cat((x,tc.tensor(starvation).to(DEVICE)),1)
		return self.dense(x)