<script>
    import { Button, ButtonGroup, Icon } from "sveltestrap";
    import { createBookmark } from "../api/bookmarks";
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
        result_selected,
        strength,
    } from "../store";

    /* ----------------------------------------------*/
    function promoteBookmarkToSession(entry) {
        if (confirm("Copy bookmark values to main session?")) {
            $prompt = entry.prompt;
            $ddim_steps = entry.ddim_steps;
            $width = entry.width;
            $height = entry.height;
            $scale = entry.scale;
            $ddim_eta = entry.ddim_eta;
            $seed = entry.seed;

            if (!isNaN(entry.strength)) {
                $strength = entry.strength;
            }
            $batch_size = 1;
        }
    }

    /* ----------------------------------------------*/
    function onCreateBookmark() {
        createBookmark($gen_results[$result_selected]);
    }

    /* ----------------------------------------------*/
    function pinSettings() {
        $prompt = $gen_results[$result_selected].prompt;
        $seed = $gen_results[$result_selected].seed;
        $ddim_steps = $gen_results[$result_selected].ddim_steps;
        $width = $gen_results[$result_selected].width;
        $height = $gen_results[$result_selected].height;
        $scale = $gen_results[$result_selected].scale;
        $ddim_eta = $gen_results[$result_selected].ddim_eta;
        $strength = $gen_results[$result_selected].strength;

        $batch_size = 1;
    }
</script>

<div class="container">
    {#if $gen_results[$result_selected]}
        <div>
            <span title="Seed" class="seed"
                >{$gen_results[$result_selected].seed}</span
            >
        </div>
        <div class="right">
            <ButtonGroup>
                <Button title="Pin Image" on:click={pinSettings}>
                    <Icon name="pin" />
                </Button>
                <Button
                    title="Create Bookmark"
                    color="primary"
                    on:click={onCreateBookmark}
                >
                    <Icon name="bookmark" />
                </Button>
            </ButtonGroup>
        </div>
    {/if}
</div>

<style scoped>
    .container {
        background: #ddd;
        display: flex;
        padding: 0;
        padding-left: 1em;
        border-radius: 10px;
        border-bottom-left-radius: 0;
        border-bottom-right-radius: 0;
    }
    .container > div {
        margin: auto;
        flex: 1 1 auto;
    }
    .container > div.right {
        text-align: right; /* Come on, man. */
    }
</style>
