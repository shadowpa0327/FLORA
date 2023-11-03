import torch.nn as nn
import torch.nn.functional as F

class LRLinear(nn.Module):
    def __init__(self, ratio, in_channel, out_channel, bias=True):
        super().__init__()
        self.bias = bias
        self.lr = (ratio != None) and (ratio != 1.0)
        self.sample_ratio = ratio if self.lr else 1.0
        self.num_components = int(round(self.sample_ratio * min(in_channel, out_channel)))
        if self.lr:
            self.VT = nn.Linear(in_channel, self.num_components, bias=False)
            self.U = nn.Linear(self.num_components, out_channel, bias=bias)
        else:
            self.fc = nn.Linear(in_channel, out_channel, bias=bias)

    def forward(self,x,scaling_factor=None):
        if self.lr:
            x = self.VT(x)
            x = self.U(x)
        else:
            x = self.fc(x)

        return x

class LRLinearSuper(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, fused = False):
        super().__init__()
        self.bias = bias
        self.fused = fused
        self.num_components = min(in_channel, out_channel)
        self.in_features = in_channel
        self.out_features = out_channel
        self.full_rank = min(in_channel, out_channel)
        self.VT = nn.Linear(in_channel, self.num_components, bias=False)
        self.U = nn.Linear(self.num_components, out_channel, bias=bias)
        self.samples = {}
        self.set_sample_config(1.0)
        
        
    def set_sample_config(self, sample, normalized = True):
        if not normalized:
            sample = sample / self.full_rank
        self.sample_ratio = sample
        self._sample_parameters()
        
    def _sample_parameters(self):
        sample_dim = int(round(self.sample_ratio * self.num_components))
        if self.fused:
            self.samples['weight'] = self.U.weight[:,:sample_dim].clone() @ self.VT.weight[:sample_dim,:].clone() 
        else:
            self.samples['VT_weight'] = self.VT.weight[:sample_dim,:]
            self.samples['U_weight'] = self.U.weight[:,:sample_dim]
        if self.bias:
            self.samples['bias'] = self.U.bias

    def forward(self,x,scaling_factor=None):
        if self.fused:
            if self.bias:
                x = F.linear(x, self.samples['weight'], self.samples['bias'])
            else:
                x = F.linear(x, self.samples['weight'])
        else:
            x = F.linear(x, self.samples['VT_weight'])
            if self.bias:
                x = F.linear(x, self.samples['U_weight'], self.samples['bias'])
            else:
                x = F.linear(x, self.samples['U_weight'])
        return x

    def __repr__(self):
        return f'LRLinearSuper(in_features={self.in_features}, out_features={self.out_features}, rank_ratio={self.sample_ratio})'




class LRLinearSuperV2(nn.Module):
    def __init__(self, choices_block_config, in_channel, out_channel, bias=True, fused = False):
        super().__init__()
        self.bias = bias
        self.fused = fused
        self.num_components = min(in_channel, out_channel)
        self.in_features = in_channel
        self.out_features = out_channel
        self.full_rank = min(in_channel, out_channel)
        self.block_size = choices_block_config
        
        if len(self.block_size) > 2:
            raise ValueError(f"currently only support 2 seperated choices block (current value: {len(self.block_size)})")
        self.VT = nn.ModuleList([nn.Linear(in_channel, int(round(self.num_components*ratio)), bias=False)
                                 for ratio in self.block_size])
        self.U = nn.ModuleList([nn.Linear( int(round(self.num_components*ratio)), out_channel, bias=bias)
                                for ratio in self.block_size]) 

        self.samples = {}
        self.set_sample_config(1.0)
        
        
    def set_sample_config(self, sample, normalized = True):
        if not normalized:
            sample = sample / self.full_rank
        self.sample_ratio = sample
        if self.sample_ratio > self.block_size[0]: #TODO(brian1009): make it configurable
            self._sample_parameters(1)
        else:
            self._sample_parameters(0)
        
    def _sample_parameters(self, i):
        sample_dim = int(round(self.sample_ratio * self.num_components))
        if self.fused:
            self.samples['weight'] = self.U[i].weight[:,:sample_dim].clone() @ self.VT[i].weight[:sample_dim,:].clone() 
        else:
            self.samples['VT_weight'] = self.VT[i].weight[:sample_dim,:]
            self.samples['U_weight'] = self.U[i].weight[:,:sample_dim]
        if self.bias:
            self.samples['bias'] = self.U[i].bias

    def forward(self,x,scaling_factor=None):
        if self.fused:
            if self.bias:
                x = F.linear(x, self.samples['weight'], self.samples['bias'])
            else:
                x = F.linear(x, self.samples['weight'])
        else:
            x = F.linear(x, self.samples['VT_weight'])
            if self.bias:
                x = F.linear(x, self.samples['U_weight'], self.samples['bias'])
            else:
                x = F.linear(x, self.samples['U_weight'])
        return x

    def __repr__(self):
        return f'LRLinearSuper(in_features={self.in_features}, out_features={self.out_features}, rank_ratio={self.sample_ratio}, choices_block_config={self.block_size})'
