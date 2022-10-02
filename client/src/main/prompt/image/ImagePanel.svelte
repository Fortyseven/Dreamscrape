<script>
    // @ts-nocheck

    import { src_image, mask_image, strength, resetImages } from "../../store";
    import { Button, ButtonGroup, Icon, Input, Label, Row } from "sveltestrap";
    // import MaskPaintLayer from "./MaskPaintLayer.svelte";
    import I2ICanvas from "./I2ICanvas/I2ICanvas.svelte";

    var files;
    var input_field;

    /**
     * Processes an image uploaded with the input widget (a opposed to pasted in)
     * @param e
     */
    function onInitFileChange(e) {
        input_field = e;

        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.addEventListener("load", function () {
                $src_image = reader.result;
            });
            reader.readAsDataURL(file);
        }
    }

    function removeImage() {
        if (confirm("Remove source image?")) {
            console.info("Removing source image");
            resetImages();

            $strength = 0.5;

            window.sessionStorage.clear("src_image");

            files = undefined;

            if (input_field && input_field.srcElement)
                input_field.srcElement.value = "";
        }
    }
</script>

<!-- ################################################################# -->
<div>
    <Label for="file">Source Image</Label>
    <Row>
        <ButtonGroup>
            <Input
                type="file"
                name="file"
                id="exampleFile"
                bind:files
                on:change={onInitFileChange}
                bind:this={input_field}
            />
            <Button on:click={removeImage}><Icon name="x" /></Button>
        </ButtonGroup>
    </Row>
    {#if $src_image}
        <br />
        <Row>
            <!-- content here -->
            <Label for="strength">
                Strength: <span class="value">{$strength}</span>
            </Label>
            <Input
                type="range"
                name="strength"
                id="strength"
                min={0}
                max={1}
                step={0.01}
                bind:value={$strength}
            />
        </Row>
    {/if}
</div>
<I2ICanvas />

<!-- <img
    class:invisible_preview={$src_image === undefined}
    id="PreviewImage"
    src={$src_image}
/>

<img
    class:invisible_preview={$mask_image === undefined}
    id="MaskImage"
    src={$mask_image}
/> -->
<style>
    /* img {
        max-width: 100%;
        margin-top: 1em;
        background: #f0f;
        box-shadow: 0.25rem 0.25rem 1rem black;
    } */
    .invisible_preview {
        visibility: hidden;
    }

    /* .value {
        font-family: monospace;
        color: blue;
        float: right;
    }

    :global(.form-label) {
        display: block !important;
        margin-top: 1em;
    } */
</style>
