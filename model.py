import math
import logging
import torch
import torch.fft
import torch.nn.functional as F
from torch.autograd import Variable
from timm.models.layers import to_2tuple, trunc_normal_
from einops import rearrange
from torch import einsum, nn

_logger = logging.getLogger(__name__)

pi = 3.1415926535

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class ParallelTransformerBlock(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.norm = LayerNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim * 2))

        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)

        self.ff_out = nn.Sequential(
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        )

        self.mask = None
        self.pos_emb = None

    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n].to(device)

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.mask = mask
        return mask

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n].to(device)

        pos_emb = self.rotary_emb(n, device=device)
        self.pos_emb = pos_emb
        return pos_emb

    def forward(self, x, attn_mask=None):
        n, device, h = x.shape[1], x.device, self.heads
        x = self.norm(x)
        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)
        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))
        q = q * self.scale
        sim = einsum("b h i d, b j d -> b h i j", q, k)
        causal_mask = self.get_mask(n, device)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
        if exists(attn_mask):
            attn_mask = rearrange(attn_mask, 'b i j -> b 1 i j')
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.attn_out(out) + self.ff_out(ff)

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim=None,
        dim_head=64,
        heads=8,
        parallel_ff=False,
        ff_mult=4,
        norm_context=False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(context_dim) if norm_context else nn.Identity()

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether to have parallel feedforward

        ff_inner_dim = ff_mult * dim

        self.ff = nn.Sequential(
            nn.Linear(dim, ff_inner_dim * 2, bias=False),
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        ) if parallel_ff else None

    def forward(self, x, context):
        x = self.norm(x)
        context = self.context_norm(context)
        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        q = q * self.scale
        k, v = self.to_kv(context).chunk(2, dim=-1)
        sim = einsum('b h i d, b j d -> b h i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        if exists(self.ff):
            out = out + self.ff(x)
        return out

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class TextPositionEmbed(nn.Module):
    def __init__(self, seq_len, d_model=128, dropout=0.):
        super(TextPositionEmbed, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, text):
        text = text + Variable(self.pe[:,:text.size(1)], requires_grad=False)

        return self.dropout(text)

class ImagePatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, d_model=128, in_channels=3):
        super(ImagePatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)  # (img_size, img_size)
        patch_size = to_2tuple(patch_size)  # (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.conv_layer = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, image):
        B, C, H, W = image.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        image = self.conv_layer(image).flatten(2).transpose(1, 2)  # (B, H*W, D)
        return image

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.):
        super(FeedForward, self).__init__()
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff),
                                          nn.Dropout(dropout),
                                          nn.GELU(),
                                          nn.Linear(d_ff, d_model),
                                          nn.Dropout(dropout))

    def forward(self, x):
        return self.feed_forward(x)

class Image2TextGate(nn.Module):
    def __init__(self, n, d_model):
        super(Image2TextGate, self).__init__()
        self.n = n
        self.avg_pool = nn.AvgPool1d(kernel_size=n)
        self.conv_layer = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.select_para = nn.Parameter(torch.randn(n, d_model, 2, dtype=torch.float32))

    def forward(self, image):
        B, N, C = image.shape
        assert N == self.n
        image = image * torch.view_as_complex(self.select_para)
        image = image.permute(0, 2, 1)  # (B, C, N)
        image = self.avg_pool(image.real)  # (B, C, 1)
        image = image.permute(0, 2, 1)  # (B, 1, C)
        return image

class Text2ImageGate(nn.Module):
    def __init__(self, s, d_model):
        super(Text2ImageGate, self).__init__()
        self.s = s
        self.avg_pool = nn.AvgPool1d(kernel_size=s)
        self.conv_layer = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.select_para = nn.Parameter(torch.randn(s, d_model, 2, dtype=torch.float32))

    def forward(self, text):
        text = text * torch.view_as_complex(self.select_para)  # (B, S, C)
        text = text.permute(0, 2, 1)
        text = self.avg_pool(text.real)  # (B, C, 1)
        text = text.permute(0, 2, 1)  # (B, 1, C)
        return text

class ImageFrequencySelection(nn.Module):
    def __init__(self, s, d_model):
        super(ImageFrequencySelection, self).__init__()

        self.text_gate = Text2ImageGate(s, d_model)

    def forward(self, image, text):
        text_gate = self.text_gate(text)
        image = image * text_gate
        return image

class TextFrequencySelection(nn.Module):
    def __init__(self, n, d_model):
        super(TextFrequencySelection, self).__init__()

        self.image_gate = Image2TextGate(n, d_model)

    def forward(self, text, image):
        image_gate = self.image_gate(image)
        text = text * image_gate
        return text

class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.):
        super(AddNorm, self).__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = FeedForward(d_model, d_model, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x)
        x_ = x
        x = self.dropout(x)
        x = self.feed_forward(x) + x_
        x = self.norm2(x)
        return x

class FtLayer(nn.Module):
    def __init__(self, d_model, s, n, num_filter=2, dropout=0.,use_bank=True):
        super(FtLayer, self).__init__()
        self.s = s
        self.n = n
        self.use_bank = use_bank
        self.num_filter = num_filter

        self.text_weight = nn.Parameter(torch.randn(s, d_model, 2, dtype=torch.float32))
        self.text_filter_bank = nn.Parameter(torch.randn(num_filter, s, d_model, 2, dtype=torch.float32))

        self.image_weight = nn.Parameter(torch.randn(n, d_model, 2, dtype=torch.float32))
        self.image_filter_bank = nn.Parameter(torch.randn(num_filter, n, d_model, 2, dtype=torch.float32))

        self.text_frequency_select = TextFrequencySelection(n, d_model)
        self.image_frenquency_select = ImageFrequencySelection(s, d_model)

        self.text_add_norm = AddNorm(d_model, dropout)
        self.image_add_norm = AddNorm(d_model, dropout)

    def filter(self, x, length, filter_bank, weight):
        if self.use_bank:
            power = (x * x) / length
            Y = []
            for k in range(self.num_filter):
                cos = torch.cos(torch.as_tensor((2 * (k + 1) - 1) * pi / 2 * self.num_filter))
                Y.append(power * filter_bank[k] * cos)
            C = torch.stack(Y)  # (filter, batch, s, dim)
            x = torch.sum(C, dim=0)  # (batch, s, dim)
        else:
            x = x * weight

        return x

    def forward(self, text, image, spatial_size=None):
        x_text = text
        B, S, D = text.shape
        assert S // 2 + 1 == self.s

        x_image = image
        B, N, C = image.shape

        # fft
        _text = torch.fft.rfft(text, dim=1, norm='ortho')
        _image = torch.fft.rfft(image, dim=1, norm='ortho')

        # frequency filter
        _text = self.filter(_text, self.s, torch.view_as_complex(self.text_filter_bank),
                            torch.view_as_complex(self.text_weight))
        _image = self.filter(_image, self.n, torch.view_as_complex(self.image_filter_bank),
                             torch.view_as_complex(self.image_weight))


        # frequency select
        _text = self.text_frequency_select(_text, _image)
        _image = self.image_frenquency_select(_image, _text)

        # ifft
        text = torch.fft.irfft(_text, n=S, dim=1, norm='ortho')
        image = torch.fft.irfft(_image, n=N, dim=1, norm='ortho')

        text = self.text_add_norm(text + x_text)
        image = self.image_add_norm(image + x_image)

        return text, image

class FtBlock(nn.Module):
    def __init__(self, d_model, s, n, num_layer=1, num_filter=2, dropout=0.):
        super(FtBlock, self).__init__()
        self.ft = nn.ModuleList([FtLayer(d_model, s, n, num_filter, dropout) for _ in range(num_layer)])

    def forward(self, text, image):
        for ft_layer in self.ft:
            text, image = ft_layer(text, image)

        return text, image

class Fusion(nn.Module):
    def __init__(self, d_model, act_layer=torch.tanh):
        super(Fusion, self).__init__()

        self.text_weight = nn.Parameter(torch.randn(d_model, d_model, dtype=torch.float32))
        self.image_weight = nn.Parameter(torch.randn(d_model, d_model, dtype=torch.float32))
        self.fusion_weight = nn.Parameter(torch.randn(d_model, d_model, dtype=torch.float32))
        self.act_layer = act_layer

    def forward(self, text, image):
        alpha = self.js_div(text, image)

        fusion = torch.matmul(text, self.text_weight) + torch.matmul(image, self.image_weight)
        f = (1-alpha) * fusion + alpha * text + alpha * image

        return f

    @staticmethod
    def js_div(p, q):
        """
        Function that measures JS divergence between target and output logits:
        """
        M = (p + q) / 2
        kl1 = F.kl_div(F.log_softmax(M, dim=-1), F.softmax(p, dim=-1), reduction='batchmean')
        kl2 = F.kl_div(F.log_softmax(M, dim=-1), F.softmax(q, dim=-1), reduction='batchmean')
        gamma = 0.5 * kl1 + 0.5 * kl2
        return gamma

class MLP(nn.Module):
    def __init__(self, inputs_dim, hidden_dim, outputs_dim, num_class, act_layer=nn.ReLU, dropout=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(inputs_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act_layer = act_layer()
        self.fc2 = nn.Linear(hidden_dim, outputs_dim)
        self.norm2 = nn.LayerNorm(outputs_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(outputs_dim, num_class)

    def forward(self, x, need_tsne_data=False):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act_layer(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.act_layer(x)
        features = None
        if need_tsne_data:
            features = x.detach().cpu().numpy()
        x = self.fc3(x)
        return x, features

class TraffiCFUS(nn.Module):
    def __init__(self, W, vocab_size, d_text, seq_len, img_size, patch_size, d_model,
                 num_filter, num_class, num_layer, dropout=0., mlp_ratio=4.):
        super(TraffiCFUS, self).__init__()

        self.text_embed = nn.Embedding(vocab_size, d_text)
        self.text_embed.weight = nn.Parameter(torch.from_numpy(W),)
        self.text_encoder = nn.Sequential(nn.Linear(d_text, d_model),
                                          nn.LayerNorm(d_model),
                                          TextPositionEmbed(seq_len, d_model, dropout))
        s = seq_len // 2 + 1

        self.img_patch_embed = ImagePatchEmbed(img_size, patch_size, d_model)
        num_img_patches = self.img_patch_embed.num_patches
        self.img_pos_embed = nn.Parameter(torch.zeros(1, num_img_patches, d_model))
        self.img_pos_drop = nn.Dropout(p=dropout)
        img_len = (img_size // patch_size) * (img_size // patch_size)
        n = img_len // 2 + 1

        self.FourierTransormer = FtBlock(d_model, s, n, num_layer, num_filter, dropout)

        self.fusion = Fusion(d_model)
        self.mlp = MLP(d_model, int(mlp_ratio * d_model), d_model, num_class, dropout=dropout)

        self.multimodal_layers = nn.ModuleList([
            nn.ModuleList([
                Residual(ParallelTransformerBlock(dim=d_model, dim_head=64, heads=8, ff_mult=4)),
                Residual(CrossAttention(dim=d_model, dim_head=64, heads=8, parallel_ff=True, ff_mult=4))
            ]) for _ in range(num_layer)
        ])
        self.to_logits = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, W.shape[0])
        )

        trunc_normal_(self.img_pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.)

    def forward(self, text, image, need_tsne_data=False):
        text = text.long()
        text = self.text_embed(text)  # (batch, seq, dim)
        text_embed = text
        text = self.text_encoder(text)

        image = image.to(torch.float32)
        image = self.img_patch_embed(image)
        image = image + self.img_pos_embed
        image = self.img_pos_drop(image)

        text, image = self.FourierTransormer(text, image)

        text_fusion = torch.max(text, dim=1)[0]
        image_fusion = torch.max(image, dim=1)[0]
        f = self.fusion(text_fusion, image_fusion)  # (batch, d_model)
        outputs, features = self.mlp(f, need_tsne_data)

        if need_tsne_data:
            return text_fusion, image_fusion, outputs, f, features

        for attn_ff, cross_attn in self.multimodal_layers:
            text = attn_ff(text)
            text = cross_attn(text, image)
        logits = self.to_logits(text)

        return text_fusion, image_fusion, outputs, f, logits