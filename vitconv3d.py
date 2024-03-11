import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange
import pdb

class Residual(nn.Module):
	def __init__(self, fn):
		super().__init__()
		self.fn = fn

	def forward(self, x, **kwargs):
		return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
	def __init__(self, dim, fn):
		super().__init__()
		self.norm = nn.LayerNorm(dim)
		self.fn = fn

	def forward(self, x, **kwargs):
		return self.fn(self.norm(x), **kwargs)


class PreNormLocal(nn.Module):
	def __init__(self, dim, fn):
		super().__init__()
		self.norm = nn.LayerNorm(dim)
		self.fn = fn

	def forward(self, x, **kwargs):
		x = x.permute(0, 2, 3, 1)
		x = self.norm(x)
		x = x.permute(0, 3, 1, 2)
		# print('before fn: ', x.shape)
		x = self.fn(x, **kwargs)
		# print('after fn: ', x.shape)
		return x

class FeedForward(nn.Module):
	def __init__(self, dim, hidden_dim, dropout=0.):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(dim, hidden_dim),
			#nn.GELU(),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, dim),
			nn.Dropout(dropout)
		)

	def forward(self, x):
		return self.net(x)


class Attention(nn.Module):
	def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
		super().__init__()
		inner_dim = dim_head * heads
		project_out = not (heads == 1 and dim_head == dim)

		self.heads = heads
		self.scale = dim_head ** -0.5
		self.temperature = 1

		self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

		self.to_out = nn.Sequential(
			nn.Linear(inner_dim, dim),
			nn.Dropout(dropout)
		) if project_out else nn.Identity()

	def forward(self, x):
		# print(x.shape)
		#pdb.set_trace()
		b, n, _, h = *x.shape, self.heads
		qkv = self.to_qkv(x).chunk(3, dim=-1)
		q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
		# print(q.shape, k.shape, v.shape)
		#dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
		attn = torch.matmul(q / self.temperature, k.transpose(2, 3))*self.scale #torch.Size([1, 32, 751, 751])
		attn = F.softmax(attn, dim=-1) ##torch.Size([1, 32, 751, 751])
		out = torch.matmul(attn, v) #torch.Size([1, 32, 751, 8])
		#attn = dots.softmax(dim=-1)

		#out = einsum('b h i j, b h j d -> b h i d', attn, v)
		out = rearrange(out, 'b h n d -> b n (h d)')
		out = self.to_out(out)
		return out

class Transformer(nn.Module):
	def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
		super().__init__()
		self.layers = nn.ModuleList([])
		self.norm = nn.LayerNorm(dim)
		# for _ in range(depth):
		# 	self.layers.append(nn.ModuleList([
		# 		PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
		# 		PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
		# 	]))
		for _ in range(depth):
			self.layers.append(nn.ModuleList([
				PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
				PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
			]))
	def forward(self, x):
		for attn, ff in self.layers:
			x = attn(x) + x
			x = ff(x) + x
		return self.norm(x)

class TSViTcls_conv3d(nn.Module):
	"""
	Temporal-Spatial ViT for uSkin
	"""
	def __init__(self, image_size_w, image_size_h, patch_size_w, patch_size_h, patch_size_d, num_classes,
	      		max_seq_len, out_channels, temporal_depth, depth, spatial_depth, heads, dropout, emb_dropout,
				num_channels, scale_dim):
		super().__init__()
		self.image_size_w = image_size_w
		self.image_size_h = image_size_h 
		self.patch_size_w = patch_size_w
		self.patch_size_h = patch_size_h
		self.patch_size_d = patch_size_d
		self.num_patches = self.image_size_w * self.image_size_h // self.patch_size_w // self.patch_size_h
		self.num_classes = num_classes #regression target
		self.num_frames = max_seq_len
		#self.dim = dim
		self.out_channels = out_channels
		self.dim = self.num_patches*self.out_channels
		self.scale_dim = scale_dim
		if temporal_depth:
			self.temporal_depth = temporal_depth
		else:
			self.temporal_depth = depth
		if spatial_depth:
			self.spatial_depth = spatial_depth
		else:
			self.spatial_depth = depth
		self.heads = heads
		self.dim_head = int(self.dim/self.heads)
		self.dropout = dropout
		self.emb_dropout = emb_dropout 
		num_patches = self.num_patches
		patch_dim = (num_channels) * self.patch_size_h * self.patch_size_w * self.patch_size_d
		#pdb.set_trace
		self.to_patch_embedding_conv3d = nn.Conv3d(6, self.out_channels, kernel_size=(patch_size_d, patch_size_w, patch_size_h), 
		 								stride=(patch_size_d, patch_size_w, patch_size_h))
		self.to_temporal_embedding_input = nn.Linear(int(self.num_frames/self.patch_size_d), self.dim)
		self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim)).cuda()
		self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
												self.dim * self.scale_dim, self.dropout)

		self.to_space_pos_embedding = nn.Linear(self.num_patches, int(self.dim/self.num_patches))
		self.space_transformer = Transformer(int(self.dim/self.num_patches), self.spatial_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)
		self.dropout = nn.Dropout(self.emb_dropout)

		self.mlp_head = nn.Sequential(
			#nn.LayerNorm(self.dim),
			nn.Linear(int(self.dim*self.num_frames/self.patch_size_d),int(self.dim*self.num_frames/self.patch_size_d/10)),
			nn.GELU(),
			nn.Linear(int(self.dim*self.num_frames/self.patch_size_d/10), 2)
			)
		#self.mlp_head1 = nn.Linear(self.dim, 1)
		#self.mlp_head2 = nn.Linear(int((self.num_frames/self.patch_size_d+self.num_classes)*(self.num_patches+1)), self.num_classes) 

	def forward(self, x):

		B, T, C, W, H = x.shape
		#temporal position
		#pdb.set_trace()
	
		xt = torch.arange(0, int(self.num_frames/self.patch_size_d)).to(torch.int64)
		xt = F.one_hot(xt, num_classes=int(self.num_frames/self.patch_size_d)).to(torch.float32)
		x = self.to_patch_embedding_conv3d(x.permute(0, 2, 1, 3, 4)).permute(0,2,1,3,4) #torch.Size([128, 75, 16, 2, 3])
		x = x.reshape(x.size(0), x.size(1), -1) #torch.Size([128, 75, 96])
		#pdb.set_trace()

		temporal_pos_embedding = self.to_temporal_embedding_input(xt.cuda()).reshape(1, int(T/self.patch_size_d), self.dim)
		x += temporal_pos_embedding
		x = self.temporal_transformer(x)
		
		#pdb.set_trace()
		x = x.reshape(B, self.num_patches, x.size(1), int(self.dim/self.num_patches)).permute(0, 2, 1, 3).reshape(B*x.size(1), self.num_patches, int(self.dim/self.num_patches))
		#space position
		xs = torch.arange(0, int(self.num_patches)).to(torch.int64)
		xs = F.one_hot(xs, num_classes=int(self.num_patches)).to(torch.float32)
		self.space_pos_embedding = self.to_space_pos_embedding(xs.cuda())
		x += self.space_pos_embedding#[:, :, :(n + 1)]
		x = self.space_transformer(x)
		#pdb.set_trace()

		x = self.mlp_head(x.reshape(B, -1))
		
		return x
	
if __name__ == "__main__":
	# res = 24
	# model_config = {'img_res': res, 'patch_size': 3, 'patch_size_time': 1, 'patch_time': 4, 'num_classes': 20,
	# 				'max_seq_len': 16, 'dim': 128, 'temporal_depth': 10, 'spatial_depth': 4, 'depth': 4,
	# 				'heads': 3, 'pool': 'cls', 'num_channels': 14, 'dim_head': 64, 'dropout': 0., 'emb_dropout': 0.,
	# 				'scale_dim': 4}
	model_config = {'img_size_w': 4, 'img_size_h': 6, 'patch_size_w': 1, 'patch_size_h': 2, 'patch_size_time': 1, 'patch_time': 4, 'num_classes': 2,
					'max_seq_len': 750, 'dim': 128, 'temporal_depth': 6, 'spatial_depth': 4, 'depth': 4,
					'heads': 3, 'pool': 'cls', 'num_channels': 3, 'dim_head': 64, 'dropout': 0., 'emb_dropout': 0.,
					'scale_dim': 4}
	# train_config = {'dataset': "psetae_repl_2018_100_3", 'label_map': "labels_20k2k", 'max_seq_len': 16, 'batch_size': 5,
	# 				'extra_data': [], 'num_workers': 4}

	model = TSViTcls_uSkin(model_config)#.cuda()
	parameters = filter(lambda p: p.requires_grad, model.parameters())
	parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
	print('Trainable Parameters: %.3fM' % parameters)
	img = torch.rand((2, 750, 3, 4, 6))#.cuda()
	out = model(img)
	print("Shape of out :", out.shape)  # [B, num_classes]
	pdb.set_trace()