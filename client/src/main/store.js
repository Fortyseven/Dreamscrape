import { writable } from "svelte/store";

export const defaults = {
  prompt: "",
  //   ddim_steps: 40,
  //   batch_size: 4,
  ddim_steps: 35,
  batch_size: 1,
  width: 512,
  height: 512,
  scale: 7.5,
  ddim_eta: 0.1,
  seed: "",
  turbo: true,
  full_precision: true,
  sampler: "plms",
  do_upscale: true,
  strength: 0.5,
  result_selected: 0,
};


// TODO:
// export const frobs = writable({
//     prompt : defaults.prompt,
//     ddim_steps : defaults.ddim_steps,
//     batch_size : defaults.batch_size,
//     width : defaults.width,
//     height : defaults.height,
//     scale : defaults.scale,
//     ddim_eta : defaults.ddim_eta,
//     seed : defaults.seed,
//     turbo : defaults.turbo,
//     full_precision : defaults.full_precision,
//     sampler : defaults.sampler,
//     do_upscale : defaults.do_upscale,
//     strength : defaults.strength,
// });

export const prompt = writable(defaults.prompt);
export const ddim_steps = writable(defaults.ddim_steps);
export const batch_size = writable(defaults.batch_size);
export const width = writable(defaults.width);
export const height = writable(defaults.height);
export const scale = writable(defaults.scale);
export const ddim_eta = writable(defaults.ddim_eta);
export const seed = writable(defaults.seed);
export const turbo = writable(defaults.turbo);
export const full_precision = writable(defaults.full_precision);
export const sampler = writable(defaults.sampler);
export const do_upscale = writable(defaults.do_upscale);
export const strength = writable(defaults.strength);

export const result_selected = writable(0);
export const is_loading = writable(false);
export const src_image = writable(null);
export const mask_image = writable(null);
// export const mask_image = writable("data:image/png;base64,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");

export const gen_results = writable([]);
export const prompt_log = writable([]);
export const bookmarks = writable([]);
export const api = writable(null);

/* ----------------------------------------------*/
export function resetStore() {
  prompt.set(defaults.prompt);
  ddim_steps.set(defaults.ddim_steps);
  batch_size.set(defaults.batch_size);
  width.set(defaults.width);
  height.set(defaults.height);
  scale.set(defaults.scale);
  ddim_eta.set(defaults.ddim_eta);
  seed.set(defaults.seed);
  turbo.set(defaults.turbo);
  full_precision.set(defaults.full_precision);
  sampler.set(defaults.sampler);
  do_upscale.set(defaults.do_upscale);
  strength.set(defaults.strength);
  result_selected.set(0);

  is_loading.set(false);

  gen_results.set([]);
  window.sessionStorage.clear();
}

export function resetImages() {
  src_image.set(null);
//   mask_image.set(undefined);
}

let snd_error_audio = undefined;
let snd_finished_audio = undefined;

/* ----------------------------------------------*/
export function snd_error() {
  if (!snd_error_audio) {
    snd_error_audio = new Audio("/media/error.wav");
    return;
  }
  snd_error_audio.play();
}

/* ----------------------------------------------*/
export function snd_finished() {
  if (!snd_finished_audio) {
    snd_finished_audio = new Audio("/media/finished.wav");
    return;
  }
  snd_finished_audio.play();
}

