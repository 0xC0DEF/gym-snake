import torch.nn as nn
import torch.nn.init as init
from config import GameOption
    
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
        self.chn_mid=8
        self.chn_out=8
        self.ch_adjuster=nn.Conv2d(3,self.chn_in,1,padding=0,bias=False)
        self.ffcs=nn.ModuleList([
            FFC(self.chn_in,self.chn_mid,3),
            FFC(self.chn_mid,self.chn_mid,3),
            FFC(self.chn_mid,self.chn_mid,3),
            FFC(self.chn_mid,self.chn_mid,3),
            FFC(self.chn_mid,self.chn_out,3)])
        
        self.dense=nn.Sequential(
            nn.Linear(self.chn_out*GameOption.ROW*GameOption.COL,256),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            
            CTR(256,128,True),
            CTR(128,32),
            
            nn.Linear(32,3),
        )
        def weight_init(x):
            if type(x)==nn.Linear or type(x)==nn.Conv2d:
                init.xavier_uniform_(x.weight.data)
                if x.bias != None:
                    init.zeros_(x.bias)
        self.ffcs.apply(weight_init)
        self.dense.apply(weight_init)
                
    def forward(self,x):
        #x=x.reshape((-1,self.chn_in,GameOption.ROW,GameOption.COL))
        x=x.reshape((-1,3,GameOption.ROW,GameOption.COL))
        xa=self.ch_adjuster(x)
        x=xa
        for ffc in self.ffcs:
            x=ffc(x)+xa #residual training
        x=x.reshape(-1,self.chn_out*GameOption.ROW*GameOption.COL)
        return self.dense(x)