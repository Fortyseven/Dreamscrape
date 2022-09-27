import time
import torch
from torch import autocast
from pytorch_lightning import seed_everything
from contextlib import nullcontext
from einops import repeat
from random import randint

from sd.optimUtils import split_weighted_subprompts

from modes.shared import save_images, load_img
import common


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
    strength,
    image
):
    print("############################################generate_img2img")

    if seed == "":
        seed = randint(0, 1000000)

    seed = int(seed)
    seed_everything(seed)

    sampler = "ddim"

    src_image = load_img(image, height, width).to(device)

    common.model.unet_bs = unet_bs
    common.model.turbo = turbo
    common.model.cdevice = device
    common.modelCS.cond_stage_model.device = device

    if device != "cpu" and full_precision == False:
        common.model.half()
        common.modelCS.half()
        common.modelFS.half()
        src_image = src_image.half()

    tic = time.time()

    assert prompt is not None
    data = [batch_size * [prompt]]

    common.modelFS.to(device)
# TODO: check  similar spot in  inpaint mode
    src_image = repeat(src_image, "1 ... -> b ...", b=batch_size)
    init_latent = common.modelFS.get_first_stage_encoding(
        common.modelFS.encode_first_stage(src_image))  # move to latent space

    if device != "cpu":
        mem = torch.cuda.memory_allocated() / 1e6
        common.modelFS.to("cpu")
        while torch.cuda.memory_allocated() / 1e6 >= mem:
            time.sleep(1)

    assert 0.0 <= strength <= 1.0, "can only work with strength in [0.0, 1.0]"

    t_enc = int(strength * ddim_steps)

    print(f"# target t_enc is {t_enc} steps")

    if full_precision == False and device != "cpu":
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    all_samples = []

    with torch.no_grad():
        all_samples = list()
        for prompts in data:
            print("# prompts", prompts)

            with precision_scope("cuda"):
                common.modelCS.to(device)

                uc = None

                if scale != 1.0:
                    uc = common.modelCS.get_learned_conditioning(
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
                        c = torch.add(c, common.modelCS.get_learned_conditioning(
                            subprompts[i]), alpha=weight)
                else:
                    c = common.modelCS.get_learned_conditioning(prompts)

                if device != "cpu":
                    mem = torch.cuda.memory_allocated() / 1e6
                    common.modelCS.to("cpu")
                    while torch.cuda.memory_allocated() / 1e6 >= mem:
                        time.sleep(1)

                # encode (scaled latent)
                z_enc = common.model.stochastic_encode(
                    init_latent, torch.tensor(
                        [t_enc] * batch_size).to(device),
                    seed,
                    ddim_eta,
                    ddim_steps
                )
                # decode it
                samples_ddim = common.model.sample(
                    t_enc,
                    c,
                    z_enc,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc,
                    sampler=sampler
                )

                common.modelFS.to(device)
                print("# saving images")

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

    # grid = torch.cat(all_samples, 0)
    # grid = make_grid(grid, nrow=n_iter)
    # grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()

    mins = str(round(time_taken, 3))
    txt = (f"Finished in {mins} minutes.")

    return results
