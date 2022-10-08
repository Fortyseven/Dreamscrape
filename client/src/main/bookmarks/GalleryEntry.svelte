<script>
    import { onMount } from "svelte";
    import { Button, ButtonGroup, Icon, Table, Tooltip } from "sveltestrap";
    import * as api from "../api";
    import { bookmarks } from "../store";
    import { promoteBookmarkToSession } from "../utils";

    export let entry;

    const GALLERY_ENTRY_WIDTH = 128;
</script>

<div class="entry">
    <a href="http://localhost:5000/media?hash={entry.img_hash}" target="_blank">
        <div
            class="image"
            id="galleryid-{entry.img_hash}"
            style="background-image:url(http://localhost:5000/media?hash={entry.img_hash}&thumb=1&width=128); width: 128px; height:{(GALLERY_ENTRY_WIDTH *
                entry.height) /
                entry.width}px"
        />
    </a>
    <Tooltip placement="bottom" target="galleryid-{entry.img_hash}">
        {entry.prompt} [{entry.seed}]
    </Tooltip>
</div>

<style>
    .entry {
        flex: 0 1 128px;
        margin: 1em;
        width: 128px;
        box-shadow: 0 0.3rem 10px #000;
        cursor: pointer;
        background-color: black;
    }

    .entry:hover {
        box-shadow: 0 0.3rem 10px #888;
    }

    .image {
    }
    .prompt {
        font-size: 8pt;
        width: 100%;
    }
</style>
