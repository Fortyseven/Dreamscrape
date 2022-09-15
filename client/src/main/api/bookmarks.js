import axios from "axios";

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
  init_image,
  strength,
  result_selected,
  snd_error,
  snd_finished,
  prompt_log,
  reloadBookmarks,
} from "../store.js";

const SERVER_URL = "http://localhost:5000";

const api = axios.create({
  baseURL: SERVER_URL,
});

/* ----------------------------------------------*/
export async function getBookmarks(bookmarks) {
  const { data } = await api.get(`/bookmark`);
  return data.result;
}

/* ----------------------------------------------*/
export async function createBookmark(entry) {
  // strip out the big image, leave the `thumbnail`
  const saving = Object.entries(entry).filter((i) =>
    i[0] !== "image" ? i : null
  );

  const response = await api({
    url: "bookmark",
    method: "post",
    data: entry,
  });

  await reloadBookmarks();
}

/* ----------------------------------------------*/
export async function deleteBookmark(id) {
  if (confirm("Delete this bookmark?")) {
    const response = await api({
      url: "/bookmark",
      method: "delete",
      data: { id },
    });
  }
}
