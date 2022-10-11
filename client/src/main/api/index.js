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
    gen_results,
    is_loading,
    resetStore,
    src_image,
    strength,
    result_selected,
    snd_error,
    snd_finished,
    prompt_log,
    api
} from "../store.js";


const SERVER_URL = "http://localhost:5000";

window.api = axios.create({
  baseURL: SERVER_URL,
});

import * as bookmarks from "./bookmarks";

export {
    bookmarks,
}