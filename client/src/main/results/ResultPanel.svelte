<script>
    import { gen_results, result_selected } from "../store";
    import ControlBar from "./ControlBar.svelte";
    import PreviewThumbs from "./PreviewThumbs.svelte";

    let result_image = null;

    /* ----------------------------------------------*/
    $: if (result_image && $gen_results.length > 0) {
        result_image.src =
            "data:image/jpeg;base64," + $gen_results[$result_selected].image;
    }
</script>

<div>
    {#if $gen_results.length > 0}
        <PreviewThumbs />
        <div style="width: {$gen_results[$result_selected].width}px">
            <ControlBar />
        </div>
        <!-- svelte-ignore a11y-missing-attribute -->
        <img bind:this={result_image} />
    {/if}
</div>
