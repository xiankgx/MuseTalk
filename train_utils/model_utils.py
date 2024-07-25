import math
import os
import time

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from PIL import Image

from .utils import decode_latents

RESIZED_IMG = 256


class PositionalEncoding(nn.Module):
    """
    Transformer 中的位置编码（positional encoding）
    """

    def __init__(self, d_model=384, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        b, seq_len, d_model = x.size()
        pe = self.pe[:, :seq_len, :]
        # print(b, seq_len, d_model)
        x = x + pe.to(x.device)
        return x


def validation(
    vae: torch.nn.Module,
    vae_fp32: torch.nn.Module,
    unet: torch.nn.Module,
    unet_config,
    weight_dtype: torch.dtype,
    epoch: int,
    global_step: int,
    val_data_loader,
    output_dir,
    whisper_model_type,
    UNet2DConditionModel=UNet2DConditionModel,
):
    # Get the validation pipeline
    unet_copy = UNet2DConditionModel(**unet_config)
    unet_copy.load_state_dict(unet.state_dict())
    unet_copy.to(vae.device).to(dtype=weight_dtype)
    unet_copy.eval()

    if whisper_model_type == "tiny":
        # pe = PositionalEncoding(d_model=384)
        # XXX GX
        # Instead of 5 (layers) x 384 for each audio feature, I'm converting it to 1 x 1920 because we are later using another level of multi-features with using neighboring audio features.
        # With this change, using 10 audio frames, the audio feature become of shape 10 x 1920, rather than 50 x 384.
        pe = PositionalEncoding(d_model=384 * 5)
    elif whisper_model_type == "largeV2":
        pe = PositionalEncoding(d_model=1280)
    elif whisper_model_type == "tiny-conv":
        pe = PositionalEncoding(d_model=384)
        print(f" whisper_model_type: {whisper_model_type} Validation does not need PE")
    else:
        print(f"not support whisper_model_type {whisper_model_type}")
    pe.to(vae.device, dtype=weight_dtype)

    start = time.time()
    with torch.no_grad():
        for step, (ref_image, image, _, mask, audio_feature) in enumerate(
            val_data_loader
        ):
            masked_image = image * mask

            mask = mask.to(vae.device)
            ref_image = ref_image.to(vae.device)
            image = image.to(vae.device)
            masked_image = masked_image.to(vae.device)

            # Convert images to latent space
            latents = vae.encode(
                image.to(dtype=weight_dtype)
            ).latent_dist.sample()  # init image
            latents = latents * vae.config.scaling_factor

            # Convert masked images to latent space
            masked_latents = vae.encode(
                masked_image.reshape(image.shape).to(dtype=weight_dtype)  # masked image
            ).latent_dist.sample()
            masked_latents = masked_latents * vae.config.scaling_factor

            # Convert ref images to latent space
            ref_latents = vae.encode(
                ref_image.reshape(image.shape).to(dtype=weight_dtype)  # ref image
            ).latent_dist.sample()
            ref_latents = ref_latents * vae.config.scaling_factor

            # mask = torch.stack(
            #     [
            #         torch.nn.functional.interpolate(mask, size=(mask.shape[-1] // 8, mask.shape[-1] // 8))
            #         for mask in masks
            #     ]
            # )
            # mask = mask.reshape(-1, 1, mask.shape[-1], mask.shape[-1])
            mask = torch.nn.functional.interpolate(mask, size=masked_latents.shape[-2])

            bsz = latents.shape[0]
            timesteps = torch.tensor([0], device=latents.device)

            if unet_config["in_channels"] == 9:
                latent_model_input = torch.cat(
                    [mask.to(dtype=masked_latents.dtype), masked_latents, ref_latents],
                    dim=1,
                )
            else:
                latent_model_input = torch.cat([masked_latents, ref_latents], dim=1)

            audio_feature = audio_feature.to(dtype=weight_dtype)
            # XXX Missing
            audio_feature = pe(audio_feature)

            # print(f"masks.dtype: {masks.dtype}")
            # print(f"masked_latents.dtype: {masked_latents.dtype}")
            # print(f"ref_latents.dtype: {ref_latents.dtype}")
            # print(f"latent_model_input.dtype: {latent_model_input.dtype}")
            # print(f"timesteps.dtype: {timesteps.dtype}")
            # print(f"audio_feature.dtype: {audio_feature.dtype}")

            image_pred = unet_copy(
                latent_model_input, timesteps, encoder_hidden_states=audio_feature
            ).sample

            image = Image.new("RGB", (RESIZED_IMG * 4, RESIZED_IMG))
            image.paste(decode_latents(vae_fp32, masked_latents), (0, 0))
            image.paste(decode_latents(vae_fp32, ref_latents), (RESIZED_IMG, 0))
            image.paste(decode_latents(vae_fp32, latents), (RESIZED_IMG * 2, 0))
            image.paste(decode_latents(vae_fp32, image_pred), (RESIZED_IMG * 3, 0))

            val_img_dir = f"{output_dir}/images/{global_step}"
            if not os.path.exists(val_img_dir):
                os.makedirs(val_img_dir)
            image.save(
                # "{0}/val_epoch_{1}_{2}_image.png".format(val_img_dir, global_step, step)
                os.path.join(
                    val_img_dir,
                    f"validation-global_step={global_step:09d}-{step:04d}.jpg",
                )
            )

            print(
                "validation for step: {0}, time: {1:.1f} s".format(step, time.time() - start)
            )

        print(
            "validation done for epoch: {0}, time: {1:.1f} s".format(
                epoch, time.time() - start
            )
        )
