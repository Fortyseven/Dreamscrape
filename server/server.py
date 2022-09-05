#!/usr/bin/env python3

import mimetypes
import numpy as np
import os
import re
import time
import torch

from torch import autocast
from pytorch_lightning import seed_everything
from contextlib import nullcontext
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from random import randint

from sd.ldm.util import instantiate_from_config
from sd.optimUtils import split_weighted_subprompts

from flask import Flask, request, make_response


# FIXME
TMP_IMAGE_CACHE_PATH = "/tmp/sdfart"


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


def init():
    global model, modelCS, modelFS, outdir

    outdir = TMP_IMAGE_CACHE_PATH
    os.makedirs(outdir, exist_ok=True)

    config = "sd/v1-inference.yaml"
    ckpt = "sd/models/ldm/stable-diffusion-v1/model.ckpt"

    sd = load_model_from_config(f"{ckpt}")

    li, lo = [], []

    for key, v_ in sd.items():
        sp = key.split(".")
        if (sp[0]) == "model":
            if "input_blocks" in sp:
                li.append(key)
            elif "middle_block" in sp:
                li.append(key)
            elif "time_embed" in sp:
                li.append(key)
            else:
                lo.append(key)

    for key in li:
        sd["model1." + key[6:]] = sd.pop(key)

    for key in lo:
        sd["model2." + key[6:]] = sd.pop(key)

    config = OmegaConf.load(f"{config}")

    model = instantiate_from_config(config.modelUNet)
    _, _ = model.load_state_dict(sd, strict=False)
    model.eval()

    modelCS = instantiate_from_config(config.modelCondStage)
    _, _ = modelCS.load_state_dict(sd, strict=False)
    modelCS.eval()

    modelFS = instantiate_from_config(config.modelFirstStage)
    _, _ = modelFS.load_state_dict(sd, strict=False)
    modelFS.eval()

    del sd


def save_images(seed, prompt, ddim_steps, ddim_eta, sampler, scale, width, height, batch_size,  samples_ddim, all_samples):
    sample_path = os.path.join(outdir, "_".join(re.split(":| ", prompt)))[:150]

    os.makedirs(sample_path, exist_ok=True)

    base_count = len(os.listdir(sample_path))

    # seeds = []
    paths = dict()

    for i in range(batch_size):
        x_samples_ddim = modelFS.decode_first_stage(
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

        save_name = os.path.join(
            sample_path,
            "seed_" + str(seed) + "_" +
            f"{base_count:05}.png"
        )

        Image.fromarray(x_sample.astype(np.uint8)).save(save_name)

        paths[str(seed)] = {
            "path": save_name,
            "ddim_steps": ddim_steps,
            "ddim_eta": ddim_eta,
            "sampler": sampler,
            "scale": scale,
            "width": width,
            "height": height,
        }

        seed += 1
        base_count += 1

        del x_sample
        del x_samples_ddim

    return paths


def generate(
    prompt,
    ddim_steps,
    batch_size,
    height,
    width,
    scale,
    ddim_eta,
    unet_bs,
    device,
    seed,
    turbo,
    full_precision,
    sampler,
):
    print("############################################")
    C = 4
    f = 8
    start_code = None
    model.unet_bs = unet_bs
    model.turbo = turbo
    model.cdevice = device
    modelCS.cond_stage_model.device = device

    if seed == "":
        seed = randint(0, 1000000)

    seed = int(seed)
    seed_everything(seed)

    if device != "cpu" and full_precision == False:
        model.half()
        modelFS.half()
        modelCS.half()

    tic = time.time()

    assert prompt is not None

    data = [batch_size * [prompt]]

    if full_precision == False and device != "cpu":
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    all_samples = []

    results = dict()

    with torch.no_grad():
        all_samples = list()
        for prompts in data:
            with precision_scope("cuda"):
                modelCS.to(device)

                uc = None
                if scale != 1.0:
                    uc = modelCS.get_learned_conditioning(
                        batch_size * [""])

                if isinstance(prompts, tuple):
                    prompts = list(prompts)

                subprompts, weights = split_weighted_subprompts(prompts[0])

                if len(subprompts) > 1:
                    c = torch.zeros_like(uc)
                    totalWeight = sum(weights)
                    # normalize each "sub prompt" and add it
                    for i in range(len(subprompts)):
                        weight = weights[i]
                        # if not skip_normalize:
                        weight = weight / totalWeight
                        c = torch.add(c, modelCS.get_learned_conditioning(
                            subprompts[i]), alpha=weight)
                else:
                    c = modelCS.get_learned_conditioning(prompts)

                shape = [batch_size, C, height // f, width // f]

                if device != "cpu":
                    mem = torch.cuda.memory_allocated() / 1e6
                    modelCS.to("cpu")
                    while torch.cuda.memory_allocated() / 1e6 >= mem:
                        time.sleep(1)

                samples_ddim = model.sample(
                    S=ddim_steps,
                    conditioning=c,
                    seed=seed,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc,
                    eta=ddim_eta,
                    x_T=start_code,
                    sampler=sampler,
                )

                modelFS.to(device)

                print("saving images")

                results = save_images(
                    seed,
                    prompt,
                    ddim_steps,
                    ddim_eta,
                    sampler,
                    scale,
                    width,
                    height,
                    batch_size,
                    samples_ddim,
                    all_samples)

                if device != "cpu":
                    mem = torch.cuda.memory_allocated() / 1e6
                    modelFS.to("cpu")
                    while torch.cuda.memory_allocated() / 1e6 >= mem:
                        time.sleep(1)

                del samples_ddim

                print("memory_final = ", torch.cuda.memory_allocated() / 1e6)

    toc = time.time()
    time_taken = (toc - tic) / 60.0

    # grid = torch.cat(all_samples, 0)
    # grid = make_grid(grid, nrow=n_iter)
    # grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()

    mins = str(round(time_taken, 3))
    txt = (f"Finished in {mins} minutes.")

    return results
    # returrestulte.fromarray(grid.astype(np.uint8)), txt


app = Flask(__name__)

SD_ROOT_SCRIPTS = "./sd/scripts/"


@app.get("/get_image")
def api_get_image():
    # FIXME: sanitize this path; this is DANGEROUS as-is
    file_path = request.query_string
    with open(file_path, "rb") as f:
        response = make_response(f.read())
        response.headers.set('Content-Type', 'image/jpeg')
        return response


@app.post("/generate")
def api_generate():
    settings = request.get_json()

    result_paths = generate(
        prompt=settings.get("prompt", ""),
        ddim_steps=settings.get("ddim_steps", 50),
        # n_iter=settings.get("n_iter", 1),
        batch_size=settings.get("batch_size", 1),
        height=settings.get("height", ""),
        width=settings.get("width", ""),
        scale=settings.get("scale", ""),
        ddim_eta=settings.get("ddim_eta", ""),
        unet_bs=settings.get("unet_bs", 1),
        device=settings.get("device", "cuda"),
        seed=settings.get("seed", 0),
        turbo=settings.get("turbo", True),
        full_precision=settings.get("full_precision", False),
        sampler=settings.get("sampler", "plms")
    )

    # dir(request)
    # return f"<h1>Ok...?</h1><plaintext>{status}"
    return result_paths


# logging.set_verbosity_error()
mimetypes.init()
mimetypes.add_type("application/javascript", ".js")

init()
