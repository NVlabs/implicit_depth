import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


class IMNet(nn.Module):
    def __init__(self, inp_dim, out_dim, gf_dim=64, use_sigmoid=False):
        super(IMNet, self).__init__()
        self.inp_dim = inp_dim
        self.gf_dim = gf_dim
        self.use_sigmoid = use_sigmoid
        self.linear_1 = nn.Linear(self.inp_dim, self.gf_dim*4, bias=True)
        self.linear_2 = nn.Linear(self.gf_dim*4, self.gf_dim*2, bias=True)
        self.linear_3 = nn.Linear(self.gf_dim*2, self.gf_dim*1, bias=True)
        self.linear_4 = nn.Linear(self.gf_dim*1, out_dim, bias=True)
        if self.use_sigmoid:
            self.sigmoid = nn.Sigmoid()
        nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_1.bias,0)
        nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_2.bias,0)
        nn.init.normal_(self.linear_3.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_3.bias,0)
        nn.init.normal_(self.linear_4.weight, mean=1e-5, std=0.02)
        nn.init.constant_(self.linear_4.bias,0)

    def forward(self, inp_feat):
        l1 = self.linear_1(inp_feat)
        l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)

        l2 = self.linear_2(l1)
        l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)

        l3 = self.linear_3(l2)
        l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)

        l4 = self.linear_4(l3)
        
        if self.use_sigmoid:
            l4 = self.sigmoid(l4)
        else:
            l4 = torch.max(torch.min(l4, l4*0.01+0.99), l4*0.01)
        
        return l4

class IEF(nn.Module):
    def __init__(self, device, inp_dim, out_dim, gf_dim=64, n_iter=3, use_sigmoid=False):
        super(IEF, self).__init__()
        self.device = device
        self.init_offset = torch.Tensor([0.001]).float().to(self.device)
        self.inp_dim = inp_dim
        self.gf_dim = gf_dim
        self.n_iter = n_iter
        self.use_sigmoid = use_sigmoid
        self.offset_enc = nn.Linear(1, 16, bias=True)
        self.linear_1 = nn.Linear(self.inp_dim+16, self.gf_dim*4, bias=True)
        self.linear_2 = nn.Linear(self.gf_dim*4, self.gf_dim*2, bias=True)
        self.linear_3 = nn.Linear(self.gf_dim*2, self.gf_dim*1, bias=True)
        self.linear_4 = nn.Linear(self.gf_dim*1, out_dim, bias=True)
        if self.use_sigmoid:
            self.sigmoid = nn.Sigmoid()

        nn.init.normal_(self.offset_enc.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.offset_enc.bias,0)
        nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_1.bias,0)
        nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_2.bias,0)
        nn.init.normal_(self.linear_3.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_3.bias,0)
        nn.init.normal_(self.linear_4.weight, mean=1e-5, std=0.02)
        nn.init.constant_(self.linear_4.bias,0)


    def forward(self, inp_feat):
        batch_size = inp_feat.shape[0]
        # iterative update
        pred_offset = self.init_offset.expand(batch_size, -1)
        for i in range(self.n_iter):
            offset_feat = self.offset_enc(pred_offset)
            xc = torch.cat([inp_feat,offset_feat],1)
            l1 = self.linear_1(xc)
            l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)

            l2 = self.linear_2(l1)
            l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)

            l3 = self.linear_3(l2)
            l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)

            l4 = self.linear_4(l3)
            pred_offset = pred_offset + l4
        
        if self.use_sigmoid:
            pred_offset = self.sigmoid(pred_offset)
        else:
            pred_offset = torch.max(torch.min(pred_offset, pred_offset*0.01+0.99), pred_offset*0.01)
        return pred_offset
