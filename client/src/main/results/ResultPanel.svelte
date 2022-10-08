<script>
    import {
        Dropdown,
        DropdownItem,
        DropdownMenu,
        DropdownToggle,
    } from "sveltestrap";
    import {
        batch_size,
        ddim_eta,
        ddim_steps,
        gen_results,
        height,
        prompt,
        scale,
        seed,
        src_image,
        strength,
        width,
    } from "../store";
    import * as api from "../api";

    let isOpen = {};

    function createBookmark(entry) {
        const data = {
            ...entry,
            strength: $strength,
        };

        if ($src_image) {
            data["src_image"] = $src_image;
        }

        delete data["thumbnail"];

        api.bookmarks.create(data);
    }

    function pinEntry(entry) {
        $prompt = entry.prompt;
        $seed = entry.seed;
        $ddim_steps = entry.ddim_steps;
        $width = entry.width;
        $height = entry.height;
        $scale = entry.scale;
        $ddim_eta = entry.ddim_eta;
        // if (!isNaN(entry)) {
        //     $strength = entry.strength;
        // } else {
        //     $strength = 0.5;
        // }

        $batch_size = 1;
    }

    function useAsSourceImage(entry) {
        //
    }
</script>

<div>
    {#if $gen_results.length > 0}
        <div class="result-container">
            <div class="result-container-inner">
                {#each $gen_results as result}
                    <div class="result">
                        <Dropdown
                            isOpen={isOpen[result.seed]}
                            toggle={() =>
                                (isOpen[result.seed] = !isOpen[result.seed])}
                        >
                            <DropdownToggle tag="div">
                                <!-- svelte-ignore a11y-missing-attribute -->
                                <img
                                    src={`data:image/jpeg;base64,${result.image}`}
                                />
                            </DropdownToggle>
                            <DropdownMenu right>
                                <DropdownItem header>
                                    {result.seed}
                                </DropdownItem>
                                <DropdownItem header />
                                <DropdownItem
                                    on:click={() => createBookmark(result)}
                                >
                                    🔖 Bookmark This!
                                </DropdownItem>
                                <DropdownItem on:click={() => pinEntry(result)}>
                                    📌 Pin Seed/Settings
                                </DropdownItem>
                                <DropdownItem
                                    on:click={() => useAsSourceImage(result)}
                                >
                                    🖼 Use as Source Image
                                </DropdownItem>
                            </DropdownMenu>
                        </Dropdown>
                    </div>
                {/each}
            </div>
        </div>
    {/if}
</div>

<style>
    .result-container {
        overflow: scroll;
        height: 90vh;
        margin-top: 1rem;
    }
    .result-container-inner {
        display: flex;
        flex-wrap: wrap;
    }
    .result {
        flex: 0 1 auto;
        box-shadow: 0 10px 20px #0008;
    }
</style>
