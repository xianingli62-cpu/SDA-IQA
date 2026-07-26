import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from diffusers import AutoencoderKL
from diffusers.models.unet_2d_condition import UNet2DConditionModel, UNet2DConditionOutput
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from .image_adapter import create_adapter_variant
except ImportError:
    from image_adapter import create_adapter_variant


class UNet(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:

        default_overall_up_factor = 2**self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
                class_labels = class_labels.to(dtype=sample.dtype)
            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)
            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
        elif self.config.addition_embed_type == "text_image":
            image_embs = added_cond_kwargs.get("image_embeds")
            text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
            aug_emb = self.add_embedding(text_embs, image_embs)
        elif self.config.addition_embed_type == "text_time":
            text_embeds = added_cond_kwargs.get("text_embeds")
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(add_embeds)
        elif self.config.addition_embed_type == "image":
            image_embs = added_cond_kwargs.get("image_embeds")
            aug_emb = self.add_embedding(image_embs)

        emb = emb + aug_emb if aug_emb is not None else emb
        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)

        sample = self.conv_in(sample)

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        is_adapter = mid_block_additional_residual is None and down_block_additional_residuals is not None

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                additional_residuals = {}
                if is_adapter and len(down_block_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_block_additional_residuals.pop(0)

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                if is_adapter and len(down_block_additional_residuals) > 0:
                    sample += down_block_additional_residuals.pop(0)
            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)
            down_block_res_samples = new_down_block_res_samples

        if self.mid_block is not None:
            sample = self.mid_block(
                sample, emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )
            if is_adapter and len(down_block_additional_residuals) > 0:
                sample += down_block_additional_residuals.pop(0)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        up_sample_0 = []
        up_sample_1 = []
        up_sample_2 = []
        up_sample_3 = []

        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb,
                    res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

            if i == 0:
                up_sample_0 = sample
            elif i == 1:
                up_sample_1 = sample
            elif i == 2:
                up_sample_2 = sample
            elif i == 3:
                up_sample_3 = sample

        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # Return multi-level features from the four up-blocks instead of the
        # final noise prediction, for hierarchical quality decoding.
        return up_sample_0, up_sample_1, up_sample_2, up_sample_3


class TextAdapter(nn.Module):
    """Lightweight two-layer adapter (GELU in between) that modulates the
    initial quality-text embeddings in a residual manner."""
    def __init__(self, text_dim=768, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = text_dim
        self.fc = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, text_dim)
        )

    def forward(self, latents, texts, gamma):
        bs = latents.shape[0]
        texts_after = self.fc(texts)
        texts = texts + gamma * texts_after
        texts = repeat(texts, 'n c -> b n c', b=bs)
        return texts


class SELayer(nn.Module):
    """Squeeze-and-Excitation channel refinement."""
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class LAAF(nn.Module):
    """Layer Adaptive Attention Fusion.

    Dynamically assigns aggregation weights to the four U-Net stages based on
    global content descriptors: per-stage global average pooling produces
    content descriptors, which are concatenated and mapped by a lightweight
    MLP to a Softmax-normalized weight vector. The stage features are then
    weighted element-wise before channel concatenation.
    """
    def __init__(self, channels=512, num_stages=4, hidden_dim=64):
        super().__init__()
        self.num_stages = num_stages
        self.mlp = nn.Sequential(
            nn.Linear(channels * num_stages, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_stages),
        )

    def forward(self, feats: List[torch.Tensor]):
        """
        Args:
            feats: list of stage features, each [B, C, H, W]
        Returns:
            weighted: element-wise weighted stage features
            weights:  the Softmax-normalized layer weights [B, num_stages]
        """
        descriptors = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feats]
        joint = torch.cat(descriptors, dim=1)            # [B, C * num_stages]
        weights = F.softmax(self.mlp(joint), dim=1)      # [B, num_stages]
        weighted = [f * weights[:, i].view(-1, 1, 1, 1) for i, f in enumerate(feats)]
        return weighted, weights


class HAQD(nn.Module):
    """Hierarchical Adaptive Quality Decoder.

    Unifies the four-stage U-Net features to 64x64 via bicubic interpolation,
    projects them to a 512-dim channel space, refines each stage with an SE
    module, aggregates the stages with LAAF (content-driven adaptive weights),
    and regresses the quality score through a three-layer MLP.
    """
    def __init__(self, use_laaf: bool = True):
        super(HAQD, self).__init__()
        self.use_laaf = use_laaf

        self.conv1 = nn.Conv2d(320, 512, kernel_size=3, padding=1)
        self.se1 = SELayer(512)

        self.conv2 = nn.Conv2d(640, 512, kernel_size=3, padding=1)
        self.se2 = SELayer(512)

        self.conv3 = nn.Conv2d(1280, 512, kernel_size=3, padding=1)
        self.se3 = SELayer(512)

        self.conv4 = nn.Conv2d(1280, 512, kernel_size=3, padding=1)
        self.se4 = SELayer(512)

        # LAAF: content-driven adaptive aggregation over the four stages
        self.laaf = LAAF(channels=512, num_stages=4) if use_laaf else None

        self.fuse_conv = nn.Conv2d(2048, 512, kernel_size=3, padding=1)
        self.se_fuse = SELayer(512)
        self.relu = nn.ReLU()

        self.conv_score1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv_score2 = nn.Conv2d(128, 32, kernel_size=1)
        self.conv_score3 = nn.Conv2d(32, 8, kernel_size=1)
        self.fc1 = nn.Linear(8 * 64 * 64, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x1, x2, x3, x4):
        # Spatial alignment and channel projection
        x1 = F.interpolate(x1, size=(64, 64), mode='bicubic', align_corners=False)
        x1 = self.se1(self.conv1(x1))

        x2 = F.interpolate(x2, size=(64, 64), mode='bicubic', align_corners=False)
        x2 = self.se2(self.conv2(x2))

        x3 = F.interpolate(x3, size=(64, 64), mode='bicubic', align_corners=False)
        x3 = self.se3(self.conv3(x3))

        x4 = F.interpolate(x4, size=(64, 64), mode='bicubic', align_corners=False)
        x4 = self.se4(self.conv4(x4))

        # Layer adaptive attention fusion (or plain concatenation when disabled)
        if self.use_laaf:
            feats, layer_weights = self.laaf([x1, x2, x3, x4])
        else:
            feats, layer_weights = [x1, x2, x3, x4], None

        # Deep fusion and channel refinement
        fused = torch.cat(feats, dim=1)
        fused = self.relu(self.se_fuse(self.fuse_conv(fused)))

        score_map = self.relu(self.conv_score1(fused))
        score_map = self.relu(self.conv_score2(score_map))
        score_map = self.relu(self.conv_score3(score_map))

        feature_map = score_map
        score_map = score_map.view(score_map.size(0), -1)

        score = self.relu(self.fc1(score_map))
        score = self.relu(self.fc2(score))
        score = self.fc3(score)

        return score, feature_map, layer_weights


class SDAIQA(nn.Module):
    def __init__(
        self,
        class_embedding_path='quality_embeddings.pth',
        gamma_init_value=1e-4,
        sd_path='runwayml/stable-diffusion-v1-5',
        adapter_variant: str = 'all',
    ):
        super(SDAIQA, self).__init__()

        self.vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae", use_safetensors=True)
        self.unet = UNet.from_pretrained(sd_path, subfolder="unet", use_safetensors=True)

        # Freeze the VAE
        self.vae.requires_grad_(False)

        # Pre-generated conditional text embeddings (see gene_text_embedding.py)
        self.class_embeddings = torch.load(class_embedding_path)
        text_dim = self.class_embeddings.size(-1)

        # Quality conditional embedding: learnable modulation + TextAdapter
        self.gamma = nn.Parameter(torch.ones(text_dim) * gamma_init_value)
        self.text_adapter = TextAdapter(text_dim=text_dim)

        # MFEB: multi-level feature extraction branch in the pixel space,
        # injecting multi-scale residual features into the U-Net downsampling path
        self.mfeb_kwargs = {
            "in_channels": 3,
            "channels": [320, 640, 1280, 1280],
            "num_res_blocks": 2,
            "downscale_factor": 8,
        }
        self.mfeb = create_adapter_variant(variant_name=adapter_variant, **self.mfeb_kwargs)

        # HAQD: hierarchical adaptive quality decoder.
        # LAAF is enabled for the 'laaf' and 'all' variants.
        use_laaf = adapter_variant in ('laaf', 'all')
        self.haqd = HAQD(use_laaf=use_laaf)
        print(f"[SDA-IQA] variant: {adapter_variant} (LAAF: {use_laaf})")

    def get_priors(self, img):
        with torch.no_grad():
            latents = self.vae.encode(img).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        latents = latents.to(img.device)
        self.class_embeddings = self.class_embeddings.to(img.device)

        c_crossattn = self.text_adapter(latents, self.class_embeddings, self.gamma)

        # Extract multi-scale residual features and inject them into the frozen U-Net
        down_block_additional_residuals = self.mfeb(img)

        t = torch.ones((img.shape[0],), device=img.device).long()
        x1, x2, x3, x4 = self.unet(
            latents,
            t,
            encoder_hidden_states=c_crossattn,
            down_block_additional_residuals=[
                sample for sample in down_block_additional_residuals
            ])
        return x1, x2, x3, x4

    def forward(self, img):
        x1, x2, x3, x4 = self.get_priors(img)
        score, feature_map, _ = self.haqd(x4, x3, x2, x1)
        return score, feature_map

    def forward_with_analysis(self, img):
        """Forward pass that additionally returns the LAAF layer weights,
        for visualization and analysis."""
        x1, x2, x3, x4 = self.get_priors(img)
        score, feature_map, layer_weights = self.haqd(x4, x3, x2, x1)
        return score, feature_map, layer_weights