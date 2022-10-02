#!/usr/bin/env python3

import traceback
import bookmark
import io
import mimetypes
import torch

from torch import autocast
from contextlib import nullcontext
from omegaconf import OmegaConf
from PIL import Image
from random import randint

from sd.ldm.util import instantiate_from_config

from flask import Flask, request, make_response
from flask_cors import CORS
from flask_colors import init_app
from urllib.request import urlopen
from rich import print
import rich
import common
import modes

# import profiler

TMP_UPLOAD_PATH = "/tmp/dsuploads"


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = TMP_UPLOAD_PATH
CORS(app)


def load_model_from_config(ckpt, verbose: bool = False):
    print(f"# Loading model from {ckpt}")

    pl_sd = torch.load(ckpt, map_location="cpu")

    if "global_step" in pl_sd:
        print(f"# Global Step: {pl_sd['global_step']}")

    sd = pl_sd["state_dict"]

    return sd


def init():
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

    common.model = instantiate_from_config(config.modelUNet)
    _, _ = common.model.load_state_dict(sd, strict=False)
    common.model.eval()

    common.modelCS = instantiate_from_config(config.modelCondStage)
    _, _ = common.modelCS.load_state_dict(sd, strict=False)
    common.modelCS.eval()

    common.modelFS = instantiate_from_config(config.modelFirstStage)
    _, _ = common.modelFS.load_state_dict(sd, strict=False)
    common.modelFS.eval()

    del sd


@ app.post("/generate")
def api_generate():
    src_image = None
    result_paths = []
    print(request.form)

    try:
        if 'src_image' in request.form.keys():
            # imgdata is expected to be a DATA URL encoded png regardless
            # of whether it was uploaded or pasted in
            with urlopen(request.form.get('src_image')) as imgdata:
                src_image_data = imgdata.read()

            src_image = Image.open(io.BytesIO(src_image_data))

            mask_image_data = None

            if request.form.get('mask_image'):
                print("HAS MASK!")
                with urlopen(request.form.get('mask_image')) as imgdata:
                    mask_image_data = imgdata.read()
                    mask = Image.open(io.BytesIO(mask_image_data))
            else:
                mask = None
                print("NO MASK")

            # mask = modes.modes.shared.load_mask(mask_img)
            result_paths = modes.inpaint.generate(
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
                # turbo=bool(request.form.get("turbo", True)),
                turbo=True,
                # full_precision=bool(request.form.get("full_precision", True)),
                full_precision=True,
                strength=float(request.form.get("strength", 0.5)),
                image=src_image,
                mask=mask
            )
        else:
            result_paths = modes.txt2txt.generate(
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
                # turbo=bool(request.form.get("turbo", True)),
                turbo=True,
                # full_precision=bool(request.form.get("full_precision", True)),
                full_precision=True,
                sampler=request.form.get("sampler", "plms"),
            )
    except Exception as e:
        print("# EXCEPTION!", e)
        print(traceback.print_tb(e.__traceback__))
        torch.cuda.empty_cache()
        return e.__str__(), 500

    return result_paths


@app.errorhandler(500)
def internal_error(error):
    return "500 error"


# logging.set_verbosity_error()
mimetypes.init()
mimetypes.add_type("application/javascript", ".js")

common.init()

# Initialize extension with your app.
init_app(app)


app.add_url_rule('/bookmark', view_func=bookmark.getBookmarks,
                 methods=["GET"])
app.add_url_rule('/bookmark', view_func=bookmark.saveBookmark,
                 methods=["POST"])
app.add_url_rule('/bookmark', view_func=bookmark.deleteBookmark,
                 methods=["DELETE"])

init()
