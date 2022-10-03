import time
import torch
import rich
from rich import print

from torch import autocast
from pytorch_lightning import seed_everything
from contextlib import nullcontext
from random import randint

from sd.optimUtils import split_weighted_subprompts

from modes.shared import save_images
import common
from common import console


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
    console.log("############################################generate_txt2txt")
    C = 4
    f = 8

    start_code = None

    common.model.unet_bs = unet_bs
    common.model.turbo = turbo
    common.model.cdevice = device
    common.modelCS.cond_stage_model.device = device

    if seed == "":
        seed = randint(0, 1000000)

    seed = int(seed)
    seed_everything(seed)

    if device != "cpu" and full_precision == False:
        console.log("XXX#### ASSMOINKJEYHFJEIF  ##############")
        common.model.half()
        common.modelFS.half()
        common.modelCS.half()

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
                common.modelCS.to(device)

                uc = None

                if scale != 1.0:
                    uc = common.modelCS.get_learned_conditioning(
                        batch_size * [""])

                if isinstance(prompts, tuple):
                    prompts = list(prompts)

                subprompts, weights = split_weighted_subprompts(prompts[0])
                console.log("Subprompts:", (subprompts, weights))
                # print("# weights:", weights)

                if len(subprompts) > 1:
                    c = torch.zeros_like(uc)
                    totalWeight = sum(weights)
                    # normalize each "sub prompt" and add it
                    for i in range(len(subprompts)):
                        weight = weights[i]
                        # if not skip_normalize:
                        weight = weight / totalWeight
                        c = torch.add(c, common.modelCS.get_learned_conditioning(
                            subprompts[i]), alpha=weight)
                else:
                    c = common.modelCS.get_learned_conditioning(prompts)

                shape = [batch_size, C, height // f, width // f]

                if device != "cpu":
                    mem = torch.cuda.memory_allocated() / 1e6
                    common.modelCS.to("cpu")
                    while torch.cuda.memory_allocated() / 1e6 >= mem:
                        time.sleep(1)

                samples_ddim = common.model.sample(
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

                common.modelFS.to(device)

                console.log("Saving images...")

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
                    common.modelFS.to("cpu")
                    while torch.cuda.memory_allocated() / 1e6 >= mem:
                        time.sleep(1)

                del samples_ddim

    toc = time.time()
    time_taken = (toc - tic) / 60.0

    mins = str(round(time_taken, 3))
    txt = (f"# Finished in {mins} minutes.")

    return results
