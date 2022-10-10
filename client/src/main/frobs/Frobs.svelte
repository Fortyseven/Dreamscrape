<script>
    import { Button, Col, Input, Label, Row } from "sveltestrap";

    import {
        ddim_steps,
        batch_size,
        height,
        width,
        scale,
        ddim_eta,
        do_upscale,
        defaults,
    } from "../store";

    import AspectRatioSelector from "./AspectRatioSelector.svelte";

    let showAdvanced = false;

    function swapDimensions(ev) {
        if (ev && ev.shiftKey) {
            $width = 512;
            $height = 512;
        } else {
            $width ^= $height;
            $height ^= $width;
            $width ^= $height;
        }
    }
</script>

<div>
    <Col>
        <Row>
            <Label for="ddim_steps">
                <span on:dblclick={() => ($ddim_steps = defaults.ddim_steps)}>
                    Steps: <span class="value">{$ddim_steps}</span>
                </span>
            </Label>
            <Input
                type="range"
                name="ddim_steps"
                id="ddim_steps"
                min={4}
                max={200}
                bind:value={$ddim_steps}
            />
        </Row>
        <Row>
            <Label for="batch_size">
                <span on:dblclick={() => ($batch_size = defaults.batch_size)}>
                    Batch Size <span class="value">{$batch_size}</span>
                </span>
            </Label>
            <Input
                type="range"
                name="batch_size"
                id="batch_size"
                min={1}
                max={25}
                step={1}
                bind:value={$batch_size}
            />
        </Row>
        <Row>
            <Col style="padding-left: 0">
                <Label for="width" class="block">
                    Width<span class="value">{$width}</span>
                </Label>
                <Input
                    type="range"
                    name="width"
                    id="width"
                    min={384}
                    max={1024}
                    step={64}
                    bind:value={$width}
                />
            </Col>
            <Col xs="auto" style="padding-right: 0">
                <Button on:click={swapDimensions}>↔</Button>
            </Col>
            <Col style="padding-right: 0">
                <Label for="height">
                    Height<span class="value">{$height}</span>
                </Label>
                <Input
                    type="range"
                    name="height"
                    id="height"
                    min={384}
                    max={1024}
                    step={64}
                    bind:value={$height}
                />
            </Col>
            <Col xs="auto" style="padding-right: 0">
                <AspectRatioSelector />
            </Col>
        </Row>
        <Row>
            <Label for="scale">Scale<span class="value">{$scale}</span></Label>
            <Input
                type="range"
                name="scale"
                id="scale"
                min={0}
                max={50}
                step={0.1}
                bind:value={$scale}
            />
        </Row>
        {#if showAdvanced}
            <!-- content here -->
            <Row>
                <Label for="ddim_eta">
                    ddim_eta <span class="value">{$ddim_eta}</span>
                </Label>
                <Input
                    type="range"
                    name="ddim_eta"
                    id="ddim_eta"
                    min={0}
                    max={1}
                    step={0.01}
                    bind:value={$ddim_eta}
                />
            </Row>
        {/if}
        <!-- <Row>
      <Label for="sampler">sampler</Label>
      <Input type="select" name="sampler" id="sampler" bind:value={sampler}>
        <option>ddim</option>
        <option>plms</option>
      </Input>
    </Row> -->
        <!-- <Row> -->
        <!-- <Input
                label="Do Upscale"
                type="switch"
                name="do_upscale"
                id="do_upscale"
                bind:checked={$do_upscale}
            /> -->
        <!-- <Input
        label="use ddim sampler"
        type="switch"
        name="sampler_useddim"
        id="sampler_useddim"
        bind:checked={sampler_useddim}
      /> -->
        <!-- </Row> -->
    </Col>
</div>

<style>
    .value {
        font-family: monospace;
        color: #fb0;
        float: right;
    }

    :global(.form-label) {
        display: block !important;
        margin: 0 !important;
        padding: 0 !important;
    }
</style>
