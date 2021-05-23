import torch
import torch.nn as nn
import torch.optim as optim

def Adam(params, lr, **kwargs):
	return optim.Adam(params, lr = lr, **kwargs)

def RMSprop(params, lr, **kwargs):
	return optim.RMSprop(params, lr = lr, **kwargs)

def SGD(params, lr, **kwargs):
	return optim.SGD(params, lr = lr, **kwargs)

def LBFGS(params, lr, **kwargs):
	optim.LBFGS(params, lr = lr, **kwargs)