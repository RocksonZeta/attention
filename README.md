# Attention

## Paper
[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## Attention
Attention can view as a query from values. `Attention = <Q,K>V`   
Basic Formula :   
```
Attention(Q,K,V) = softmax(QK'/sqrt(dk))V
```


### Two type attentions: 
#### ScaledDotProductAttention
```python
def forward(self,q,k,v,att_mask=None):
	'''
	att(q,k,v) = softmax(qk'/temper)v
	'''
	att = torch.bmm(q,k.transpose(1,2))/self.temper
	if att_mask is not None:
		att.data.masked_fill_(att_mask,-float('inf'))
	att = self.softmax(att)
	att = self.dropout(att)
	output = torch.bmm(att,v)
	return output,att

```

#### MultiheadAttention
```python
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


```