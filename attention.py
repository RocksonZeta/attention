import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np


class Bottle(nn.Module):
	''' Perform the reshape routine before and after an operation '''
	def forward(self,x):
		if len(x.size())<=2:
			return super(Bottle,self).forward(x)
		size = x.size()[:2]
		out = super(Bottle,self).forward(x.view(size[0]*size[1] , -1))
		return out.view(size[0],size[1],-1)
class BottleLinear(Bottle,nn.Linear):
	''' Perform the reshape routine before and after a linear projection '''
	pass
class BottleSoftmax(Bottle,nn.Softmax):
	''' Perform the reshape routine before and after a softmax projection '''
	def __init__(self,*args, **kwargs):
		super(BottleSoftmax,self).__init__(*args, **kwargs)

class ScaledDotProductAttention(nn.Module):
	def __init__(self , d_model , dropout=0.1):
		super(ScaledDotProductAttention,self).__init__()
		self.temper = np.sqrt(d_model)
		self.dropout = nn.Dropout(dropout)
		self.softmax = BottleSoftmax(1)

	def forward(self,q,k,v,att_mask=None):
		'''
		att(q,k,v) = softmax(qk'/temper)v
		'''
		att = torch.bmm(q,k.transpose(1,2))/self.temper
		if att_mask is not None:
			# (mask_fill ,value ): mask should be bytetensor(eg,[0,1]) , 0:retain raw,1:replace by value
			# att_mask use to not consider the masked data ,exp(-inf) = 0
			att.data.masked_fill_(att_mask,-float('inf'))
		att = self.softmax(att)
		att = self.dropout(att)
		output = torch.bmm(att,v)
		return output,att

class MultiheadAttention(nn.Module):
	def __init__(self ,n_head,d_model,d_k,d_v,dropout=0.1):
		'''
		d_model is embedding dim
		'''
		super(MultiheadAttention,self).__init__()
		self.n_head = n_head
		self.d_model= d_model
		self.d_k = d_k
		self.d_v = d_v
		self.dropout = dropout
		
		self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model,d_k))
		self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model,d_k))
		self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model,d_v))

		self.attention = ScaledDotProductAttention(d_model,dropout)
		self.dropout = nn.Dropout(dropout)
		self.proj = nn.Linear(n_head*d_v ,d_model)

		init.xavier_normal(self.w_qs)
		init.xavier_normal(self.w_ks)
		init.xavier_normal(self.w_vs)

		

	def forward(self,q,k,v,att_mask=None):
		'''
		q : batch_size * seq_len * x_dim (model_dim or embeding dim)
		'''
		batch_size ,len_q,d_model = q.size()
		batch_size ,len_k,d_model = k.size()
		batch_size ,len_v,d_model = v.size()

		# d_k,d_v,n_head = self.d_k,self.d_v,self.n_head
		
		qs,ks,vs = q.repeat(self.n_head,1,1),k.repeat(self.n_head,1,1),v.repeat(self.n_head,1,1)
		n_head,d_k,d_v = self.n_head,self.d_k,self.d_v

		# qs : n_head * (batch*seq) * d_model
		# wq : n_head * d_model * d_q
		# out: n_head * (batch*seq) * d_q
		# view: (n_head*batch) * seq * dq
		qs = torch.bmm(qs.view(n_head,-1,d_model),self.w_qs).view(-1 ,len_q ,d_k)
		ks = torch.bmm(ks.view(n_head,-1,d_model),self.w_ks).view(-1 ,len_k ,d_k)
		vs = torch.bmm(vs.view(n_head,-1,d_model),self.w_vs).view(-1 ,len_v ,d_v)
		
		att_masks = None
		if att_mask is not None :
			att_masks = att_mask.repeat(self.n_head,1,1)

		outputs , attns = self.attention(qs,ks,vs,att_masks)
		outputs = torch.cat(outputs.split(batch_size , dim=0),dim=-1)
		outputs = self.proj(outputs)
		outputs = self.dropout(outputs)
		return outputs , attns

