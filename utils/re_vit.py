import torch
import numpy as np
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from pos_embed import get_2d_sincos_pos_embed #位置编码

def pair(t):
    '''如果 t 不是元祖则返回 (t,t) 构成的元组
    这里其实是获取图像尺寸，如果传入参数 img_size 是 (h,w) 之间返回
    否则默认是正方形的图像，进行转换
    '''
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    '''前馈神经网络
    '''
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(), #高斯误差线性单元函数
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    '''注意力层
    '''
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads #这里指q/k/v输入给 attend 的维数 
        project_out = not (heads == 1 and dim_head == dim) 

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1) # dim=-1 表示对最高维进行Softmax
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False) #一次性线性变换出 q,k,v. 待后续拆分

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity() #单头且不进行维度变换时 to_out() 不进行任何操作(nn.Identity是不进行操作原样输出的占位层).

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1) #拆分为三个分块，对应 q,k,v。但是每个的尺寸都是 batch, nums, (heads*dim_head)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) #修改q,k,v张量尺寸为 batch, heads, nums, dim_head

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out) # 变为输入维数

class Transformer(nn.Module):
    '''Transformer Encoder
    堆叠 depth 个 含跳接(残差)的Attention+FFN
    '''
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x 
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, channels, num_classes = 1, 
                 dim, depth, heads, mlp_dim, pool = 'cls',  dim_head = 64, 
                 dropout = 0., emb_dropout = 0.):

        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean','all'}, 'pool type must be cls (cls token), mean (mean pooling) or all (keep all)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width), #变换后 h*w 就是 num_patches, p1*p2*c 就是 patch_dim
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim), #线性变换到 Transformer 的输入维数 dim
        )

        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, dim),requires_grad=False)  # fixed sin-cos embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

        self.set_pos_emd()

        self.apply(self._init_weights)

    def set_pos_emd(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embed[:, 1:, :] #增加位置编码
        cls_tokens = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = repeat(cls_tokens, '1 1 d -> b 1 d', b = b) #按batch增加 cls_tokens 的维数，它们共享权重
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.dropout(x)
        x = self.transformer(x)

        if self.pool == 'mean':
            x = x.mean(dim = 1)
            return self.mlp_head(x) #降维 dim -> num_classes，其实就是为了做分类
        if self.pool == 'cls':
            x[:, 0] 
            return self.mlp_head(x)
        # 对 patch_nums 个输出结果进行平均或截取第一个，此时 x.shape = (b,dim)，用作下游任务

        # 否则保留每一 patch 的潜在向量，用于重建
        return x, self.pos_embed

class DeViT(nn.Module):
    def __init__(self, *, image_size, patch_size, channels = 3,
                 dim, depth, heads, mlp_dim, dim_head = 64,
                 dropout = 0., emb_dropout = 0.):

        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        self.to_restructe = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(dim, patch_dim),
            nn.Linear(patch_dim, patch_dim),
            # nn.LayerNorm(patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=(image_height//patch_height), w=(image_width//patch_width), c=channels, p1=patch_height)
            #将展平的patch变为image
        )

    def forward(self, x, pos):

        x += pos # 摘除 pos_embedding

        x = self.dropout(x)
        x = self.transformer(x)
        
        x = x[:,1:,:] # 摘除 cls_tokens
        x = self.to_restructe(x)

        return x