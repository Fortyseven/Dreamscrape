import { api, bookmarks } from "../store.js";

/* ----------------------------------------------*/
export async function get(bookmarks) {
    if (api) {
        console.log("API", window.api)
        const { data } = await window.api.get(`/bookmark`);
        return data.result;
    }
    else {
        console.log("API NOT READY")
        return [];
    }
}

/* ----------------------------------------------*/
export async function create(entry) {
  // strip out the big image, leave the `thumbnail`
  const saving = Object.entries(entry).filter((i) =>
    i[0] !== "image" ? i : null
  );

  const response = await window.api({
    url: "bookmark",
    method: "post",
    data: entry,
  });

  await reload();
}

/* ----------------------------------------------*/
export async function rm(id) {
  if (confirm("Delete this bookmark?")) {
    const response = await window.api({
      url: "/bookmark",
      method: "delete",
      data: { id },
    });
  }
}

export async function reload() {
  bookmarks.set(await get());
}
