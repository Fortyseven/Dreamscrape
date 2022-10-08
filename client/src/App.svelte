<script>
    import { onMount } from "svelte";
    import Main from "./main/index.svelte";
    import {
        gen_results,
        snd_finished,
        snd_error,
        src_image,
    } from "./main/store.js";

    /* ----------------------------------------------*/
    onMount(() => {
        // first play caches
        snd_error();
        snd_finished();

        // restore sessionstorage if available
        const results = window.sessionStorage.getItem("results") || [];

        if (results.length === 0) {
            $gen_results = [];
        } else {
            $gen_results = JSON.parse(results);
        }

        const src_image_results =
            window.sessionStorage.getItem("src_image") || [];

        if (src_image_results.length === 0) {
            $src_image = null;
        } else {
            $src_image = JSON.parse(src_image_results);
        }
    });
</script>

<main>
    <Main />
</main>
