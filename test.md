@XiaXin-Aloys 
Hi. I've spent some time experimenting with your model since release and find it to be really-really great in terms of accuracy and speed. But I have a couple of modification to make it even faster & better, if you're interested. 

As I mentioned in [this issue](https://github.com/bytedance/Next-ViT/issues/2) you are using strange average pooling. I understand your concern about extra reshape to BCHW making the model slower, so I've reimplemented your E_MHSA to work on BCHW tensors directly (no `rearrange` needed) and retrained it. The accuracy didn't increase, but the models gets noticeably faster on larger resolutions. 

Here is the code of the module

<details>
  <summary>E_MHSA for BCHW + AvgPool2d </summary>
  
  ```python
  class E_MHSA(nn.Module):
    """
    Efficient Multi-Head Self Attention for BCHW input and proper AvgPool 
    """
    def __init__(self, dim, out_dim=None, head_dim=32, qkv_bias=True, qk_scale=None,
                 attn_drop=0, proj_drop=0., sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.num_heads = self.dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Conv2d(dim, self.dim, kernel_size=1, bias=qkv_bias)
        self.k = nn.Conv2d(dim, self.dim, kernel_size=1, bias=qkv_bias)
        self.v = nn.Conv2d(dim, self.dim, kernel_size=1, bias=qkv_bias)
        self.proj = nn.Conv2d(self.dim, self.out_dim, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.AvgPool2d(kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.BatchNorm2d(dim, eps=NORM_EPS)
        self.is_bn_merge = False
    def merge_bn(self, pre_bn):
        merge_pre_bn(self.q, pre_bn)
        if self.sr_ratio > 1:
            merge_pre_bn(self.k, pre_bn, self.norm)
            merge_pre_bn(self.v, pre_bn, self.norm)
        else:
            merge_pre_bn(self.k, pre_bn)
            merge_pre_bn(self.v, pre_bn)
        self.is_bn_merge = True
    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x)
        # -> [B, Hd, C', N] -> [B, Hd, N, C']
        q = q.reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)

        if self.sr_ratio > 1:
            x_ = self.sr(x)
            if not torch.onnx.is_in_onnx_export() and not self.is_bn_merge:
                x_ = self.norm(x_)
            k = self.k(x_)
            # -> [B, Hd, C', N]
            k = k.reshape(B, self.num_heads, C // self.num_heads, -1)
            v = self.v(x_)
            # -> [B, Hd, C', N]
            v = v.reshape(B, self.num_heads, C // self.num_heads, -1)
        else:
            k = self.k(x)
            k = k.reshape(B, self.num_heads, C // self.num_heads, -1)
            v = self.v(x)
            v = v.reshape(B, self.num_heads, C // self.num_heads, -1)
        # [B, Hd, N, C'] @ [B, Hd, C', Npool] -> [B, Hd, N, Npool]
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # [B, Hd, C', Npool] @ [B, Hd, Npool, N] -> [B, Hd, C', N]
        x = (v @ attn.transpose(-1, -2)).reshape(B, C, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
  ```
  
</details>


I've also experimented with linear attention from [Xcit: Cross-covariance image transformers](https://arxiv.org/abs/2106.09681) paper, with extra AvgPooling as in your attention. It works much faster on larger resolutions and slightly better. Tables below. 

<details>
  <summary>XCA module with support for down-scaling </summary>

  ```python
  class XCA_mod(nn.Module):
    """Cross-Covariance Attention (XCA)
    Operation where the channels are updated using a weighted sum. The weights are obtained from the (softmax
    normalized) Cross-covariance matrix (Q^T \\cdot K \\in d_h \\times d_h)
    This could be viewed as dynamic 1x1 convolution
    """

    def __init__(self, dim, head_dim=32, qkv_bias=True, downscale_factor: int = 1):
        super().__init__()
        self.num_heads = dim // head_dim
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.qk = conv1x1(dim, dim * 2, bias=qkv_bias)
        self.v = conv1x1(dim, dim, bias=qkv_bias)
        self.proj = nn.Sequential(conv1x1(dim, dim, bias=True))
        self.downscale_factor = downscale_factor
        if downscale_factor > 1:
            self.down = nn.AvgPool2d(kernel_size=downscale_factor)
            self.norm = nn.BatchNorm2d(dim, eps=NORM_EPS)

    def forward(self, x):
        B, C, H, W = x.shape
        # C` == channels per head, Hd == num heads
        # -> x B x Hd x C` x N
        v = self.v(x).reshape(B, self.num_heads, C // self.num_heads, -1)

        x_ = self.norm(self.down(x))  if self.downscale_factor > 1 else x
        # -> x B x Hd x C` x N_small
        q, k = self.qk(x_).reshape(B, 2, self.num_heads, C // self.num_heads, -1).unbind(dim=1)

        # Paper section 3.2 l2-Normalization and temperature scaling
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        # -> B x Hd x C` x C`
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # B x Hd x C` x C` @ B x Hd x C` x H*W -> B x C x H x W
        x_out = (attn @ v).reshape(B, C, H, W)
        x_out = self.proj(x_out)
        return x_out

    def merge_bn(self, pre_bn):
        raise NotImplemented

  ```
  
</details>

Here are the speed benchmarks. Tested on V100 with AMP16 + batch-size 8 (defaults in your deployment scripts)
|                        | Avg. Time (ms) | Median Time (ms) |
|------------------------|----------------|------------------|
| Original @ 224         | 3.4698         | 3.4509           |
| BCHW + AvgPool2d @ 224 | 3.4613         | 3.4365           |
| BCHW + XCA             | 3.4288         | 3.4161           |
| Original @ 384         | 7.7770         | 7.7455           |
| BCHW + AvgPool2d       | 6.9375         | 6.9192           |
| BCHW + XCA             | 6.0936         | 6.0682           |


I've also compared your provided weights with my self-trained XCA version. The acc@1 is almost identical, but loss on validation is significantly lower, not sure how to interpret this. The numbers are slightly different from yours, because I've evaluated on 1 GPU.

|                           | Trained @ | Eval @ | Acc@1  | Loss  |
|---------------------------|-----------|--------|--------|-------|
| Original model            | 224       | 224    | 82.484 | 0.958 |
|                           |           | 288    | 82.950 | 0.966 |
|                           |           | 384    | 82.256 | 1.043 |
| Original + Finetune @ 384 |           | 384    | 83.578 | 0.900 |
| XCA model                 | 224       | 224    | 82.524 | 0.881 |
|                           |           | 288    | 82.880 | 0.860 |
|                           |           | 384    | 82.238 | 0.900 |


Testing model trained with maxpool

@224
* Acc@1 82.340 Acc@5 96.168 loss 0.917
@384
* Acc@1 82.312 Acc@5 96.146 loss 0.939
@448 - всего на процент меньше, кмк хороший результат
* Acc@1 81.332 Acc@5 95.672 loss 0.988
@448 + 2x scale_ratio. Как видно качество только сильно падает от такого, хотя ожидалось ровно обратного. 
* Acc@1 77.068 Acc@5 93.140 loss 1.284


AvgPool + LS + Ema + BN in XCA
@224
* Acc@1 82.576 Acc@5 96.256 loss 0.922

Качество падает лишь чуть-чуть. Как оно будет работать при увеличение размера в 2 раза и даунскейла в 2 раз? Ожидаю что лучше чем без увеличения даунскейла



--- 


Код для запусков скриптов на vast.ai