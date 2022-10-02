import {
  prompt,
  ddim_steps,
  batch_size,
  height,
  width,
  scale,
  ddim_eta,
  seed,
  turbo,
  gen_results,
  is_loading,
  resetStore,
  src_image,
  strength,
  result_selected,
  snd_error,
  snd_finished,
  prompt_log,
} from "./store.js";

export function promoteBookmarkToSession(entry) {
  if (confirm("Copy bookmark values to main session?")) {
    prompt.set(entry.prompt);
    ddim_steps.set(entry.ddim_steps);
    width.set(entry.width);
    height.set(entry.height);
    scale.set(entry.scale);
    ddim_eta.set(entry.ddim_eta);
    seed.set(entry.seed);
    ddim_eta.set(entry.ddim_eta);
    ddim_eta.set(entry.ddim_eta);

    if (!isNaN(entry.strength)) {
      strength.set(entry.strength);
    }
    batch_size.set(1);
  }
}
