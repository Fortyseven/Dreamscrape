<script>
    import { result_selected, gen_results } from "../store";

    /* ----------------------------------------------*/
    function onSelectThumb(image_index) {
        $result_selected = image_index;
    }
</script>

<div id="preview-thumbs">
    {#each $gen_results as image, i}
        <div on:click={() => onSelectThumb(i)} class="thumb">
            <!-- svelte-ignore a11y-missing-attribute -->
            <img src="data:image/jpeg;base64,{image.thumbnail}" />
        </div>
    {/each}
</div>

<style scoped>
    #preview-thumbs {
        display: flex;
    }

    #preview-thumbs .thumb {
        margin: 1rem 0;
        margin-right: 0.5rem;
        box-shadow: 3px 3px 2px rgba(0, 0, 0, 0.25);
    }

    #preview-thumbs .thumb:hover {
        box-shadow: none;
    }
</style>
