#!/usr/bin/env python3

import base64
import traceback
import bookmark
import io
import mimetypes
import numpy as np
import os
import time
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

from flask import Flask, request, make_response
from flask_cors import CORS
from flask_colors import init_app
from urllib.request import urlopen

import db

# import profiler


# FIXME
TMP_IMAGE_CACHE_PATH = "/tmp/sdfart"


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = TMP_IMAGE_CACHE_PATH
CORS(app)


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


def load_img(image, h0, w0):
    image = image.convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    if h0 is not None and w0 is not None:
        h, w = h0, w0

    # resize to integer multiple of 32
    w, h = map(lambda x: x - x % 64, (w, h))

    print(f"New image size ({w}, {h})")
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


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
    # sample_path = os.path.join(outdir, "_".join(re.split(":| ", prompt)))[:150]

    # os.makedirs(sample_path, exist_ok=True)

    # base_count = len(os.listdir(sample_path))

    paths = []

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


def generate_img2img(
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
    strength,
    image
):

    print("############################################generate_img2img")
    # args = (prompt,
    #         ddim_steps,
    #         batch_size,
    #         height,
    #         width,
    #         scale,
    #         ddim_eta,
    #         unet_bs,
    #         device,
    #         seed,
    #         turbo,
    #         full_precision,
    #         strength,
    #         image)
    # print(args)

    if seed == "":
        seed = randint(0, 1000000)

    seed = int(seed)
    seed_everything(seed)

    # Logging
    sampler = "ddim"

    init_image = load_img(image, height, width).to(device)
    model.unet_bs = unet_bs
    model.turbo = turbo
    model.cdevice = device
    modelCS.cond_stage_model.device = device

    if device != "cpu" and full_precision == False:
        model.half()
        modelCS.half()
        modelFS.half()
        init_image = init_image.half()

    tic = time.time()
    # n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    assert prompt is not None
    data = [batch_size * [prompt]]

    modelFS.to(device)

    init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)
    init_latent = modelFS.get_first_stage_encoding(
        modelFS.encode_first_stage(init_image))  # move to latent space

    if device != "cpu":
        mem = torch.cuda.memory_allocated() / 1e6
        modelFS.to("cpu")
        while torch.cuda.memory_allocated() / 1e6 >= mem:
            time.sleep(1)

    assert 0.0 <= strength <= 1.0, "can only work with strength in [0.0, 1.0]"
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    if full_precision == False and device != "cpu":
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    all_samples = []

    with torch.no_grad():
        all_samples = list()
        for prompts in data:
            print("prompts", prompts)
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

                if device != "cpu":
                    mem = torch.cuda.memory_allocated() / 1e6
                    modelCS.to("cpu")
                    while torch.cuda.memory_allocated() / 1e6 >= mem:
                        time.sleep(1)

                # encode (scaled latent)
                z_enc = model.stochastic_encode(
                    init_latent, torch.tensor(
                        [t_enc] * batch_size).to(device),
                    seed,
                    ddim_eta,
                    ddim_steps
                )
                # decode it
                samples_ddim = model.sample(
                    t_enc,
                    c,
                    z_enc,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc,
                    sampler=sampler
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

                # print("memory_final = ", torch.cuda.memory_allocated() / 1e6)

    toc = time.time()
    time_taken = (toc - tic) / 60.0

    # grid = torch.cat(all_samples, 0)
    # grid = make_grid(grid, nrow=n_iter)
    # grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()

    mins = str(round(time_taken, 3))
    txt = (f"Finished in {mins} minutes.")

    return results


# @profiler.my_profiler
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
    print("############################################generate")
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
                print("subprompts:", subprompts)
                print("weights:", weights)

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

    toc = time.time()
    time_taken = (toc - tic) / 60.0

    mins = str(round(time_taken, 3))
    txt = (f"Finished in {mins} minutes.")

    return results


# @ app.get("/get_image")
# def api_get_image():
#     # FIXME: sanitize this path; this is DANGEROUS as-is
#     file_path = urllib.parse.unquote(request.query_string)
#     with open(file_path, "rb") as f:
#         response = make_response(f.read())
#         response.headers.set('Content-Type', 'image/png')
#         return response


@ app.post("/generate")
def api_generate():
    image = None
    result_paths = []

    try:
        if 'init_image' in request.form.keys():
            # imgdata is expected to be a data url encoded png regardless
            # of whether it was uploaded or pasted in
            with urlopen(request.form.get('init_image')) as imgdata:
                image_data = imgdata.read()

            image = Image.open(io.BytesIO(image_data))
            result_paths = generate_img2img(
                prompt=request.form.get("prompt", ""),
                ddim_steps=int(request.form.get("ddim_steps", 50)),
                batch_size=int(request.form.get("batch_size", 1)),
                height=int(request.form.get("height", "")),
                width=int(request.form.get("width", "")),
                scale=float(request.form.get("scale", 7.5)),
                ddim_eta=float(request.form.get("ddim_eta", "")),
                unet_bs=int(request.form.get("unet_bs", 1)),
                device=request.form.get("device", "cuda"),
                seed=request.form.get("seed", ''),
                turbo=bool(request.form.get("turbo", True)),
                full_precision=bool(request.form.get("full_precision", False)),
                strength=float(request.form.get("strength", 0.5)),
                image=image
            )
        else:
            print("REQUEST", request.form)
            result_paths = generate(
                prompt=request.form.get("prompt", ""),
                ddim_steps=int(request.form.get("ddim_steps", 50)),
                batch_size=int(request.form.get("batch_size", 1)),
                height=int(request.form.get("height", "")),
                width=int(request.form.get("width", "")),
                scale=float(request.form.get("scale", 7.5)),
                ddim_eta=float(request.form.get("ddim_eta", "")),
                unet_bs=int(request.form.get("unet_bs", 1)),
                device=request.form.get("device", "cuda"),
                seed=request.form.get("seed", ''),
                turbo=bool(request.form.get("turbo", True)),
                full_precision=bool(request.form.get("full_precision", False)),
                sampler=request.form.get("sampler", "plms"),
            )
    except Exception as e:
        print("EXCEPTION!", e)
        print(traceback.print_tb(e.__traceback__))
        torch.cuda.empty_cache()
        return "Fuck", 500

    return result_paths


# logging.set_verbosity_error()
mimetypes.init()
mimetypes.add_type("application/javascript", ".js")

db.init()

# Initialize extension with your app.
init_app(app)


app.add_url_rule('/bookmark', view_func=bookmark.getBookmarks,
                 methods=["GET"])
app.add_url_rule('/bookmark', view_func=bookmark.saveBookmark,
                 methods=["POST"])
app.add_url_rule('/bookmark', view_func=bookmark.deleteBookmark,
                 methods=["DELETE"])

init()
