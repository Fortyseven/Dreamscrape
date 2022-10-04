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
from common import console


def load_img(image, h0, w0):
    image = image.convert("RGB")
    w, h = image.size
    console.log(f"Loaded input image of size ({w}, {h})")
    if h0 is not None and w0 is not None:
        h, w = h0, w0

    # resize to integer multiple of 32
    w, h = map(lambda x: x - x % 64, (w, h))

    console.log(f"New image size ({w}, {h})")
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def load_mask(mask, h0, w0, invert=False):
    image = mask.convert("RGB")
    w, h = image.size
    console.log(f"Loaded input image of size ({w}, {h})")
    if(h0 is not None and w0 is not None):
        h, w = h0, w0

    # resize to integer multiple of 32
    w, h = map(lambda x: x - x % 64, (w, h))

    console.log(f"New image size ({w}, {h})")
    image = image.resize((64, 64), resample=Image.LANCZOS)
    image = np.array(image)

    if invert:
        console.log("Inverted")
        where_0, where_1 = np.where(image == 0), np.where(image == 255)
        image[where_0], image[where_1] = 255, 0
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


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

        png_bytes = io.BytesIO()
        thumb_bytes = io.BytesIO()

        img.save(png_bytes, format="PNG")
        # img.save(jpg_bytes, format="jpeg", quality=97, subsampling=0)
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
            "image": base64.b64encode(png_bytes.getvalue()).decode(),
            "thumbnail": base64.b64encode(thumb_bytes.getvalue()).decode()
        })

        seed += 1
        # base_count += 1

        del img
        del png_bytes
        del x_sample
        del x_samples_ddim

    return paths
