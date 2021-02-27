import torch as tc
import torch.nn as nn
import torch.nn.init as init
from config import Option
from core import DEVICE

class Conv(nn.Module):
	def __init__(self, chn_in, chn_out, ker_sz):
		super().__init__()
		self.c=nn.Conv2d(chn_in,chn_out,ker_sz,padding=ker_sz//2,padding_mode="circular",bias=False)
		#self.d=nn.Dropout2d(0.5)
		#self.b=nn.BatchNorm2d(chn_out)
		self.a=nn.LeakyReLU(0.1)

	def forward(self, x):
		return self.a(self.c(x))
	
class Full(nn.Module):
	def __init__(self, N_in, N_out, drop_out=False):
		super().__init__()
		self.l=nn.Linear(N_in,N_out)
		self.drop_out=drop_out
		if self.drop_out: self.d=nn.Dropout(0.5)
		self.a=nn.LeakyReLU(0.1)

	def forward(self, x):
		x=self.l(x)
		if self.drop_out: x=self.d(x)
		return self.a(x)

class NN(nn.Module):
	def __init__(self):
		super(NN,self).__init__()
		self.chn_in=2
		self.chn_out=64
		self.conv=nn.Sequential(
			Conv(self.chn_in,16,5),
			Conv(16,32,3),
			Conv(32,self.chn_out,3),
		)
		self.full_adv=nn.Sequential(
			Full(self.chn_out*Option.ROW*Option.COL+1,512),
			Full(512,256),
			Full(256,4),
		)
		self.full_stval=nn.Sequential(
			Full(self.chn_out*Option.ROW*Option.COL+1,512),
			Full(512,256),
			Full(256,4),
		)
		for x in self.modules():
			if isinstance(x,nn.Conv2d) or isinstance(x,nn.Linear):
				init.xavier_uniform_(x.weight.data)
				if x.bias != None:
					init.zeros_(x.bias)

	def forward(self,x,starvation):
		x = x.reshape((-1,self.chn_in,Option.ROW,Option.COL))
		x = self.conv(x)
		x = x.reshape(-1,self.chn_out*Option.ROW*Option.COL)
		starvation = starvation.reshape(-1,1)
		x = tc.cat((x,tc.tensor(starvation).to(DEVICE)),1)
		stval = self.full_stval(x)
		adv = self.full_adv(x)
		qval = stval+(adv-adv.mean())
		return qval