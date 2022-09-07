import { writable } from "svelte/store";

export const defaults = {
  prompt: "",
  ddim_steps: 50,
  batch_size: 4,
  //  ddim_steps : 8,
  //  batch_size : 4,
  width: 512,
  height: 512,
  scale: 7.5,
  ddim_eta: 0.1,
  seed: "",
  turbo: true,
  full_precision: false,
  sampler: "plms",
  do_upscale: true,
  strength: 0.5,
  result_selected: 0,
};

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
export const init_image = writable(undefined);

export const gen_results = writable([]);

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
  init_image.set(undefined);

  gen_results.set([]);
}

let snd_error_audio = undefined;
let snd_finished_audio = undefined;

export function snd_error() {
  if (!snd_error_audio) {
    snd_error_audio = new Audio("/media/error.wav");
    return;
  }
  snd_error_audio.play();
}
export function snd_finished() {
  if (!snd_finished_audio) {
    snd_finished_audio = new Audio("/media/finished.wav");
    return;
  }
  snd_finished_audio.play();
}
