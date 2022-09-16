import base64
import io
import numpy as np
import torch

from torch import autocast
from pytorch_lightning import seed_everything
from contextlib import nullcontext
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image
from random import randint

from sd.ldm.util import instantiate_from_config
from sd.optimUtils import split_weighted_subprompts

from urllib.request import urlopen


import common


def save_images(seed, prompt, ddim_steps, ddim_eta, sampler,
                scale, width, height, batch_size,  samples_ddim,
                all_samples):
    # sample_path = os.path.join(outdir, "_".join(re.split(":| ", prompt)))[:150]

    # os.makedirs(sample_path, exist_ok=True)

    # base_count = len(os.listdir(sample_path))

    paths = []

    for i in range(batch_size):
        x_samples_ddim = common.modelFS.decode_first_stage(
            samples_ddim[i].unsqueeze(0)
        )

        x_sample = torch.clamp(
            (x_samples_ddim + 1.0) / 2.0,
            min=0.0,
            max=1.0
        )

        all_samples.append(x_sample.to("cpu"))

        x_sample = 255.0 * \
            rearrange(
                x_sample[0].cpu().numpy(), "c h w -> h w c")

        # save_name = os.path.join(
        #     sample_path,
        #     "seed_" + str(seed) + "_" +
        #     f"{base_count:05}.png"
        # )

        img = Image.fromarray(x_sample.astype(np.uint8))

        # img.save(save_name)

        jpg_bytes = io.BytesIO()
        thumb_bytes = io.BytesIO()

        # img.save(png_bytes, format="PNG")
        img.save(jpg_bytes, format="jpeg", quality=97, subsampling=0)
        img = img.resize((64, 64))
        img.save(thumb_bytes, format="jpeg", quality=95)

        paths.append({
            # "path": save_name,
            "seed": str(seed),
            "ddim_steps": ddim_steps,
            "ddim_eta": ddim_eta,
            "sampler": sampler,
            "scale": scale,
            "width": width,
            "height": height,
            "prompt": prompt,
            "image": base64.b64encode(jpg_bytes.getvalue()).decode(),
            "thumbnail": base64.b64encode(thumb_bytes.getvalue()).decode()
        })

        seed += 1
        # base_count += 1

        del img
        del jpg_bytes
        del x_sample
        del x_samples_ddim

    return paths
