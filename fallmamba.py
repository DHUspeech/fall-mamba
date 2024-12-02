import os
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint
import torchvision.models as models
from einops import rearrange
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

MODEL_PATH = 'your_model_path'
_MODELS = {
    "videomamba_t16_in1k": os.path.join(MODEL_PATH, "videomamba_t16_in1k_res224.pth"),
    "videomamba_s16_in1k": os.path.join(MODEL_PATH, "videomamba_s16_in1k_res224.pth"),
    "videomamba_m16_in1k": os.path.join(MODEL_PATH, "videomamba_m16_in1k_res224.pth"),
}

class VideoFeatureExtractor(nn.Module):
    def __init__(self, embed_dim):
        super(VideoFeatureExtractor, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.fc = nn.Linear(resnet.fc.in_features, embed_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)  
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class AudioFeatureExtractor(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(AudioFeatureExtractor, self).__init__()
        resnet = (models.resnet50
                  (pretrained=True))
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(resnet.fc.in_features, embed_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x) 
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout_rate=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.depth = embed_dim // num_heads

        self.query_audio = nn.Linear(embed_dim, embed_dim)
        self.key_video = nn.Linear(embed_dim, embed_dim)
        self.value_video = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, video_features, audio_features):
        batch_size = video_features.size(0)
        seq_length = video_features.size(1)

        query = self.query_audio(audio_features).view(batch_size, seq_length, self.num_heads, self.depth).transpose(1, 2)
        key = self.key_video(video_features).view(batch_size, seq_length, self.num_heads, self.depth).transpose(1, 2)
        value = self.value_video(video_features).view(batch_size, seq_length, self.num_heads, self.depth).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.depth)
        attn = self.softmax(scores)

        out = torch.matmul(attn, value).transpose(1, 2).contiguous().view(batch_size, seq_length, self.num_heads * self.depth)
        out = self.norm(out)
        return out

class MultiHeadTemporalAttention(nn.Module):
    def __init__(self, in_dim, num_heads=8, dropout_rate=0.1, max_len=5000):
        super().__init__()
        assert in_dim % num_heads == 0, "in_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.depth = in_dim // num_heads

        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_rate)
        self.scale = nn.Parameter(torch.sqrt(torch.tensor(self.depth, dtype=torch.float32)))
        self.norm = nn.LayerNorm(in_dim)

        nn.init.kaiming_uniform_(self.query.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.key.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.value.weight, mode='fan_in', nonlinearity='relu')

        self.time_pos_encoding = self._generate_time_pos_encoding(max_len, in_dim)

    def _generate_time_pos_encoding(self, max_len, d_model):
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pos_encoding = torch.zeros(max_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0)
        return pos_encoding

    def forward(self, x, seq_len=None):
        batch_size, seq_len, _ = x.size()

       
        time_pos_encoding = self.time_pos_encoding[:, :seq_len, :].to(x.device)
        x = x + time_pos_encoding

        query = self.query(x).view(batch_size, seq_len, self.num_heads, self.depth).transpose(1, 2)
        key = self.key(x).view(batch_size, seq_len, self.num_heads, self.depth).transpose(1, 2)
        value = self.value(x).view(batch_size, seq_len, self.num_heads, self.depth).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        attn = self.softmax(scores)
        attn = self.dropout(attn)

        out = torch.matmul(attn, value).transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.depth)
        out = self.norm(out)
        return out


class Block(nn.Module):
    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False,
                 residual_in_fp32=False, drop_path=0., apply_temporal_attention=False,
                 num_heads=8, mlp_ratio=4.0, max_len=5000):
        super().__init__()
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32
        self.apply_temporal_attention = apply_temporal_attention
        if apply_temporal_attention:
            self.temporal_attention = MultiHeadTemporalAttention(dim, num_heads=num_heads, max_len=max_len)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, dim),
                nn.Dropout(drop_path)
            )

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,
                use_checkpoint=False):
        if residual is None:
            residual = hidden_states

        if self.apply_temporal_attention:
            temp_output = self.temporal_attention(hidden_states)
            hidden_states = temp_output + residual
            hidden_states = self.mlp(hidden_states)

        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)

        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, inference_params)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)

        if not self.apply_temporal_attention:
            if not self.fused_add_norm:
                residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
                hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states if residual is None else self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        drop_path=0.,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        layer_idx=None,
        bimamba=True,
        device=None,
        dtype=None,
        apply_temporal_attention=False,
        num_heads=8,
        max_len=5000
):
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba=bimamba, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        apply_temporal_attention=apply_temporal_attention,
        num_heads=num_heads,
        max_len=max_len
    )
    block.layer_idx = layer_idx
    return block

def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,
        rescale_prenorm_residual=True,
        n_residuals_per_layer=2,
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, kernel_size=1, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.tubelet_size = kernel_size

        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(kernel_size, patch_size[0], patch_size[1]),
            stride=(kernel_size, patch_size[0], patch_size[1])
        )

    def forward(self, x):
        x = self.proj(x)
        return x

class VisionMamba(nn.Module):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            depth=24,
            embed_dim=192,
            channels=3,
            num_classes=2,
            drop_rate=0.,
            drop_path_rate=0.1,
            ssm_cfg=None,
            norm_epsilon=1e-5,
            initializer_cfg=None,
            fused_add_norm=True,
            rms_norm=True,
            residual_in_fp32=True,
            bimamba=True,
            kernel_size=1,
            num_frames=8,
            fc_drop_rate=0.,
            device=None,
            dtype=None,
            use_checkpoint=False,
            checkpoint_num=0,
            max_len=5000  # 添加 max_len 参数
    ):
        factory_kwargs = {"device": device, "dtype": dtype}  # follow MambaLMHeadModel
        super().__init__()
        self.num_frames = num_frames
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        print(f'Use checkpoint: {use_checkpoint}')
        print(f'Checkpoint number: {checkpoint_num}')

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            kernel_size=kernel_size,
            in_chans=channels, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        self.video_feature_extractor = VideoFeatureExtractor(embed_dim)
        self.audio_feature_extractor = AudioFeatureExtractor(input_dim=3, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, (num_frames // kernel_size) * 200, embed_dim))  # 乘以200
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.head_drop = nn.Dropout(fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.layers = nn.ModuleList()
        for i in range(depth):
            apply_temporal_attention = i < 16
            max_len = num_frames * 200 
            block = create_block(
                embed_dim,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                bimamba=bimamba,
                drop_path=inter_dpr[i],
                apply_temporal_attention=apply_temporal_attention,
                max_len=max_len,  
                **factory_kwargs,
            )
            self.layers.append(block)

        self.cross_attention = CrossAttention(embed_dim=embed_dim)

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)

        # original init
        self.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "temporal_pos_embedding"}

    def get_num_layers(self):
        return len(self.layers)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, video, audio=None, inference_params=None):
        B, C, T, H, W = video.shape
        # print(f"Initial video shape: {video.shape}")

        # 处理视频特征
        video = video.view(B * T, C, H, W)
        #print(f"Reshaped video shape: {video.shape}")
        video_features = self.video_feature_extractor(video)
        video_features = video_features.view(B, T, -1)
        # print(f"Video features shape: {video_features.shape}")

        if audio is not None:
            if audio.dim() == 5:
                audio = audio.view(B * T, 3, H, W)
            # print(f"Reshaped audio shape: {audio.shape}")
            audio_features = self.audio_feature_extractor(audio)
            audio_features = audio_features.view(B, T, -1)
            # print(f"Audio features shape: {audio_features.shape}")

            # 融合特征
            fused_features = self.cross_attention(video_features, audio_features)
            fused_features = fused_features.view(B, T, self.embed_dim).permute(0, 2, 1)
            fused_features = fused_features.unsqueeze(2).unsqueeze(2)
            fused_features = fused_features.expand(-1, -1, H, W, -1)
            fused_features = fused_features.permute(0, 1, 4, 2, 3)

            if fused_features.shape[1] > 3:
                fused_features = fused_features[:, :3, :, :, :]
            # print(f"Fused features shape before PatchEmbed: {fused_features.shape}")
            x = self.patch_embed(fused_features)
        else:
            video_features = video_features.view(B, T, self.embed_dim).permute(0, 2, 1)
            video_features = video_features.unsqueeze(2).unsqueeze(2)
            video_features = video_features.expand(-1, -1, H, W, -1)
            video_features = video_features.permute(0, 1, 4, 2, 3)

            if video_features.shape[1] > 3:
                video_features = video_features[:, :3, :, :, :]
            # print(f"Video features shape before PatchEmbed: {video_features.shape}")
            x = self.patch_embed(video_features)

        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((x[:, :x.size(1) // 2, :], cls_token, x[:, x.size(1) // 2:, :]), dim=1)
        x = x + self.pos_embed

        cls_tokens = x[:B, :1, :]
        x = x[:, 1:]
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        x = x + self.temporal_pos_embedding[:, :T, :]
        x = rearrange(x, '(b n) t m -> b (t n) m', b=B, t=T)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)
        residual = None
        hidden_states = x
        for idx, layer in enumerate(self.layers):
            if self.use_checkpoint and idx < self.checkpoint_num:
                hidden_states, residual = layer(hidden_states, residual, inference_params=inference_params,
                                                use_checkpoint=True)
            else:
                hidden_states, residual = layer(hidden_states, residual, inference_params=inference_params)

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(self.drop_path(hidden_states), self.norm_f.weight, self.norm_f.bias,
                                              eps=self.norm_f.eps, residual=residual, prenorm=False,
                                              residual_in_fp32=self.residual_in_fp32)

        return hidden_states[:, 0, :]

    def forward(self, video, audio=None, inference_params=None):
        x = self.forward_features(video, audio, inference_params)
        x = self.head(self.head_drop(x))
        return x


def inflate_weight(weight_2d, time_dim, center=True):
    print(f'Init center: {center}')
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim
    return weight_3d

def load_state_dict(model, state_dict, center=True):
    state_dict_3d = model.state_dict()
    for k in state_dict.keys():
        if k in state_dict_3d.keys() and state_dict[k].shape != state_dict_3d[k].shape:
            if len(state_dict_3d[k].shape) <= 3:
                print(f'Ignore: {k}')
                continue
            print(f'Inflate: {k}, {state_dict[k].shape} => {state_dict_3d[k].shape}')
            time_dim = state_dict_3d[k].shape[2]
            state_dict[k] = inflate_weight(state_dict[k], time_dim, center=center)

    del state_dict['head.weight']
    del state_dict['head.bias']
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

@register_model
def videomamba_tiny(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16,
        embed_dim=192,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_t16_in1k"], map_location='cpu')
        load_state_dict(model, state_dict, center=True)
    return model

@register_model
def videomamba_small(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16,
        embed_dim=384,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_s16_in1k"], map_location='cpu')
        load_state_dict(model, state_dict, center=True)
    return model

@register_model
def videomamba_middle(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16,
        embed_dim=576,
        depth=32,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_m16_in1k"], map_location='cpu')
        load_state_dict(model, state_dict, center=True)
    return model

if __name__ == '__main__':
    import time
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    import numpy as np

    seed = 4217
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    num_frames = 8
    img_size = 224

    model = videomamba_middle(num_frames=num_frames).cuda()
    flops = FlopCountAnalysis(model, torch.rand(1, 3, num_frames, img_size, img_size).cuda())
    s = time.time()
    print(flop_count_table(flops, max_depth=1))
    print(time.time() - s)
