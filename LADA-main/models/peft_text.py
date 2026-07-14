import torch
import torch.nn as nn
import torch.nn.functional as F

from clip.model import CLIP

from .peft_modules import *


class Text_Tuner(nn.Module):
    def __init__(self, cfg, clip_model:CLIP, num_classes):
        super().__init__()
        # dtype = clip_model.text_projection.dtype
        # feat_dim = clip_model.text_projection.shape[1]
        # self.adaptformer = AdaptFormer(feat_dim, feat_dim // 4, dtype=dtype)

        n_layers = len(clip_model.transformer.resblocks)
        emb_dim = clip_model.text_projection.shape[1]
        dtype = clip_model.text_projection.dtype

        blocks = clip_model.transformer.resblocks

        get_attn_in_weight = lambda i: blocks[i].attn.in_proj_weight
        get_attn_in_bias = lambda i: blocks[i].attn.in_proj_bias
        get_attn_out_weight = lambda i: blocks[i].attn.out_proj.weight
        get_attn_out_bias = lambda i: blocks[i].attn.out_proj.bias
        get_mlp_in_weight = lambda i: blocks[i].mlp[0].weight
        get_mlp_in_bias = lambda i: blocks[i].mlp[0].bias
        get_mlp_out_weight = lambda i: blocks[i].mlp[2].weight
        get_mlp_out_bias = lambda i: blocks[i].mlp[2].bias

        attn_in_dim = get_attn_in_bias(0).shape[0]
        attn_out_dim = get_attn_out_bias(0).shape[0]
        mlp_in_dim = get_mlp_in_bias(0).shape[0]
        mlp_out_dim = get_mlp_out_bias(0).shape[0]


        use_full_tuning = cfg.t_full_tuning
        use_bias_tuning = cfg.t_bias_tuning
        use_ln_tuning = cfg.t_ln_tuning
        use_adapter = cfg.t_adapter
        use_adaptformer = cfg.t_adaptformer
        use_lora = cfg.t_lora
        use_lora_mlp = cfg.t_lora_mlp
        use_ssf_attn = cfg.t_ssf_attn
        use_ssf_mlp = cfg.t_ssf_mlp
        use_ssf_ln = cfg.t_ssf_ln
        use_mask = cfg.t_mask
        partial = cfg.t_partial
        adapter_dim = cfg.t_adapter_dim
        mask_ratio = cfg.t_mask_ratio
        mask_seed = cfg.t_mask_seed

        if partial is None:
            _start, _end = 0, n_layers
        elif isinstance(partial, int):
            _start, _end = n_layers - partial, n_layers
        elif isinstance(partial, list):
            _start, _end = partial[0], partial[1]
        
        if (use_adapter or use_adaptformer or use_lora or use_lora_mlp) and (adapter_dim is None):
            adapter_dim = 2 ** max(0, int(math.log2(num_classes / (n_layers * 2))))
            # adapter_dim = max(1, num_classes // (n_layers * 2))
            print("Text 'Adapter' bottle dimension set to {}".format(adapter_dim))

        if use_mask and mask_ratio is None:
            mask_ratio = num_classes / (12 * n_layers * emb_dim)
            mask_ratio = max(0.001, mask_ratio // 0.001 * 0.001)
            print("Mask ratio set to {}".format(mask_ratio))
        
        if use_mask and mask_seed is None:
            mask_seed = 0

        if use_full_tuning:
            block_tuned = blocks[_start: _end]
        else:
            block_tuned = None

        if use_bias_tuning:
            bias_tuned = nn.ParameterList([
                param for name, param in blocks.named_parameters()
                if name.endswith("bias")
            ])
        else:
            bias_tuned = None
        
        if use_ln_tuning:
            ln_tuned = nn.ModuleList([
                mod for name, mod in blocks.named_modules()
                if isinstance(mod, nn.LayerNorm)
            ])
        else:
            ln_tuned = None
        
        if use_adapter:
            adapter_list = nn.ModuleList([
                *[None] * (_start),
                *[Adapter(in_dim=emb_dim, bottle_dim=adapter_dim, dtype=dtype) for _ in range(_start, _end)],
                *[None] * (n_layers - _end)
            ])
        else:
            adapter_list = nn.ModuleList([None] * n_layers)

        if use_adaptformer:
            adaptformer_list = nn.ModuleList([
                *[None] * (_start),
                *[AdaptFormer(in_dim=emb_dim, bottle_dim=adapter_dim, dtype=dtype) for _ in range(_start, _end)],
                *[None] * (n_layers - _end)
            ])
        else:
            adaptformer_list = nn.ModuleList([None] * n_layers)

        if use_lora:
            lora_list = nn.ModuleList([
                *[None] * (_start),
                *[nn.ModuleDict({
                    "q": LoRA(in_dim=emb_dim, bottle_dim=adapter_dim, dtype=dtype),
                    "v": LoRA(in_dim=emb_dim, bottle_dim=adapter_dim, dtype=dtype),
                }) for _ in range(_start, _end)],
                *[None] * (n_layers - _end)
            ])
        else:
            lora_list = nn.ModuleList([None] * n_layers)

        if use_lora_mlp:
            lora_mlp_list = nn.ModuleList([
                *[None] * (_start),
                *[nn.ModuleDict({
                    "1": LoRA(in_dim=emb_dim, bottle_dim=adapter_dim, out_dim=mlp_in_dim, dtype=dtype),
                    "2": LoRA(in_dim=mlp_in_dim, bottle_dim=adapter_dim, out_dim=emb_dim, dtype=dtype),
                }) for _ in range(_start, _end)],
                *[None] * (n_layers - _end)
            ])
        else:
            lora_mlp_list = nn.ModuleList([None] * n_layers)

        if use_ssf_attn:
            ssf_attn_list = nn.ModuleList([
                *[None] * (_start),
                *[nn.ModuleDict({
                    "attn_in": SSF(attn_in_dim, dtype=dtype),
                    "attn_out": SSF(attn_out_dim, dtype=dtype),
                }) for _ in range(_start, _end)],
                *[None] * (n_layers - _end)
            ])
        else:
            ssf_attn_list = nn.ModuleList([None] * n_layers)

        if use_ssf_mlp:
            ssf_mlp_list = nn.ModuleList([
                *[None] * (_start),
                *[nn.ModuleDict({
                    "mlp_in": SSF(mlp_in_dim, dtype=dtype),
                    "mlp_out": SSF(mlp_out_dim, dtype=dtype),
                }) for _ in range(_start, _end)],
                *[None] * (n_layers - _end)
            ])
        else:
            ssf_mlp_list = nn.ModuleList([None] * n_layers)
        
        if use_ssf_ln:
            ssf_ln_list = nn.ModuleList([
                *[None] * (_start),
                *[nn.ModuleDict({
                    "ln_1": SSF(emb_dim, dtype=dtype),
                    "ln_2": SSF(emb_dim, dtype=dtype),
                }) for _ in range(_start, _end)],
                *[None] * (n_layers - _end)
            ])
        else:
            ssf_ln_list = nn.ModuleList([None] * n_layers)

        if use_mask:
            generator = torch.Generator().manual_seed(mask_seed)
            masked_linear_list = nn.ModuleList([
                *[None] * (_start),
                *[nn.ModuleDict({
                    "attn_in": MaskedLinear(weight=get_attn_in_weight(i), bias=get_attn_in_bias(i),
                                            ratio=mask_ratio, generator=generator),
                    "attn_out": MaskedLinear(weight=get_attn_out_weight(i), bias=get_attn_out_bias(i),
                                             ratio=mask_ratio, generator=generator),
                    "mlp_in": MaskedLinear(weight=get_mlp_in_weight(i), bias=get_mlp_in_bias(i),
                                           ratio=mask_ratio, generator=generator),
                    "mlp_out": MaskedLinear(weight=get_mlp_out_weight(i), bias=get_mlp_out_bias(i),
                                            ratio=mask_ratio, generator=generator),
                }) for i in range(_start, _end)],
                *[None] * (n_layers - _end)
            ])
        else:
            masked_linear_list = nn.ModuleList([None] * n_layers)

        # To be optimized
        self.block_tuned = block_tuned
        self.bias_tuned = bias_tuned
        self.ln_tuned = ln_tuned
        self.adapter_list = adapter_list
        self.adaptformer_list = adaptformer_list
        self.lora_list = lora_list
        self.lora_mlp_list = lora_mlp_list
        self.ssf_attn_list = ssf_attn_list
        self.ssf_mlp_list = ssf_mlp_list
        self.ssf_ln_list = ssf_ln_list
        self.masked_linear_list = masked_linear_list


class Peft_Text(nn.Module):
    def __init__(self, clip_model:CLIP):
        super().__init__()

        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.blocks = clip_model.transformer.resblocks
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.out_dim = clip_model.text_projection.shape[1]
        self.dtype = clip_model.dtype

    def forward(self, text, tuner:Text_Tuner=None):
        x = self.token_embedding(text).to(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD

        _bsz = x.shape[0]
        _seq_len = x.shape[1]
        _emb_dim = x.shape[2]

        n_layers = len(self.blocks)

        for i in range(n_layers):
            block = self.blocks[i]

            if tuner is not None:
                adapter = tuner.adapter_list[i]
                adaptformer = tuner.adaptformer_list[i]
                lora = tuner.lora_list[i]
                lora_mlp = tuner.lora_mlp_list[i]
                ssf_attn = tuner.ssf_attn_list[i]
                ssf_mlp = tuner.ssf_mlp_list[i]
                ssf_ln = tuner.ssf_ln_list[i]
                masked_linear = tuner.masked_linear_list[i]
            else:
                adapter = adaptformer = lora = lora_mlp = ssf_attn = ssf_mlp = ssf_ln = masked_linear = None

            x = x.permute(1, 0, 2)  # NLD -> LND

            _attn = block.attn
            _ln_1 = block.ln_1
            _mlp = block.mlp
            _ln_2 = block.ln_2

            _attn_in_proj_weight = _attn.in_proj_weight
            _attn_in_proj_bias = _attn.in_proj_bias
            _attn_out_proj_weight = _attn.out_proj.weight
            _attn_out_proj_bias = _attn.out_proj.bias
            _mlp_in_proj_weight = _mlp[0].weight
            _mlp_in_proj_bias = _mlp[0].bias
            _mlp_act = _mlp[1]
            _mlp_out_proj_weight = _mlp[2].weight
            _mlp_out_proj_bias = _mlp[2].bias

            _num_heads = _attn.num_heads
            _head_dim = _emb_dim // _num_heads

            ###############################
            ## Multi-Head Self-Attention ##
            ###############################
            identity = x

            x = _ln_1(x)
            if ssf_ln is not None:
                x = ssf_ln["ln_1"](x)

            if masked_linear is not None:
                qkv = masked_linear["attn_in"](x, _attn_in_proj_weight, _attn_in_proj_bias)
            else:
                qkv = F.linear(x, _attn_in_proj_weight, _attn_in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)

            if lora is not None:
                q = q + lora["q"](x)
                v = v + lora["v"](x)
            
            if ssf_attn is not None:
                qkv = torch.cat([q, k, v], dim=-1)
                qkv = ssf_attn["attn_in"](qkv)
                q, k, v = qkv.chunk(3, dim=-1)

            q = q.contiguous().view(q.shape[0], q.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            k = k.contiguous().view(k.shape[0], k.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            v = v.contiguous().view(v.shape[0], v.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            
            attn_mask = block.attn_mask.to(dtype=x.dtype, device=x.device) if block.attn_mask is not None else None
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            # scaled_dot_product_attention:
            # q = q / math.sqrt(_head_dim)
            # attn = torch.bmm(q, k.transpose(-2, -1))
            # attn = F.softmax(attn, dim=-1)
            # x = torch.bmm(attn, v)

            x = x.transpose(0, 1).contiguous().view(-1, _emb_dim)
            
            if masked_linear is not None:
                x = masked_linear["attn_out"](x, _attn_out_proj_weight, _attn_out_proj_bias)
            else:
                x = F.linear(x, _attn_out_proj_weight, _attn_out_proj_bias)
            if ssf_attn is not None:
                x = ssf_attn["attn_out"](x)

            x = x.view(_seq_len, _bsz, _emb_dim)

            x = x + identity

            ##########################
            ## Feed-Forward Network ##
            ##########################
            identity = x

            x = _ln_2(x)
            if ssf_ln is not None:
                x = ssf_ln["ln_2"](x)

            if masked_linear is not None:
                x_out = masked_linear["mlp_in"](x, _mlp_in_proj_weight, _mlp_in_proj_bias)
            else:
                x_out = F.linear(x, _mlp_in_proj_weight, _mlp_in_proj_bias)
            
            if lora_mlp is not None:
                x_out = x_out + lora_mlp["1"](x)
            
            x = x_out

            if ssf_mlp is not None:
                x = ssf_mlp["mlp_in"](x)
            
            x = _mlp_act(x)

            if masked_linear is not None:
                x_out = masked_linear["mlp_out"](x, _mlp_out_proj_weight, _mlp_out_proj_bias)
            else:
                x_out = F.linear(x, _mlp_out_proj_weight, _mlp_out_proj_bias)
            
            if lora_mlp is not None:
                x_out = x_out + lora_mlp["2"](x)
            
            x = x_out

            if ssf_mlp is not None:
                x = ssf_mlp["mlp_out"](x)
            
            if adapter is not None:
                x = x + 0.1 * adapter(x)
            
            if adaptformer is not None:
                x = x + 0.1 * adaptformer(identity)
            
            x = x + identity
            
            x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_final(x).to(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        
        return x