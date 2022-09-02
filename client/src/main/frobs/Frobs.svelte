<script>
  import { Col, Input, Label, Row } from "sveltestrap";

  let strength = 0.7;
  let ddim_steps = 80;
  let n_iter = 0;
  let batch_size = 1;
  let height = 512;
  let width = 512;
  let scale = 10;
  let ddim_eta = 0.15;
  let unet_bs = 0;
  let sampler_useddim = true; // "ddim";
  let turbo = true;
</script>

<div>
  <Col>
    <Row>
      <Label for="strength"
        >strength: <span class="value">{strength}</span></Label
      >
      <Input
        type="range"
        name="strength"
        id="strength"
        min={0}
        max={1}
        step={0.01}
        bind:value={strength}
      />
    </Row>
    <Row>
      <Label for="ddim_steps"
        >ddim_steps: <span class="value">{ddim_steps}</span></Label
      >
      <Input
        type="range"
        name="ddim_steps"
        id="ddim_steps"
        min={0}
        max={1000}
        bind:value={ddim_steps}
      />
    </Row>
    <Row>
      <Label for="n_iter">n_iter: <span class="value">{n_iter}</span></Label>
      <Input
        type="range"
        name="n_iter"
        id="n_iter"
        min={1}
        max={100}
        step={1}
        bind:value={n_iter}
      />
    </Row>
    <Row>
      <Label for="batch_size"
        >batch_size: <span class="value">{batch_size}</span></Label
      >
      <Input
        type="range"
        name="batch_size"
        id="batch_size"
        min={0}
        max={100}
        step={1}
        bind:value={batch_size}
      />
    </Row>
    <Row>
      <Label for="width">width: <span class="value">{width}</span></Label>
      <Input
        type="range"
        name="width"
        id="width"
        min={64}
        max={4096}
        step={64}
        bind:value={width}
      />
    </Row>
    <Row>
      <Label for="height">height: <span class="value">{height}</span></Label>
      <Input
        type="range"
        name="height"
        id="height"
        min={64}
        max={4096}
        step={64}
        bind:value={height}
      />
    </Row>
    <Row>
      <Label for="scale">scale: <span class="value">{scale}</span></Label>
      <Input
        type="range"
        name="scale"
        id="scale"
        min={0}
        max={50}
        step={0.1}
        bind:value={scale}
      />
    </Row>
    <Row>
      <Label for="ddim_eta"
        >ddim_eta: <span class="value">{ddim_eta}</span></Label
      >
      <Input
        type="range"
        name="ddim_eta"
        id="ddim_eta"
        min={0}
        max={1}
        step={0.01}
        bind:value={ddim_eta}
      />
    </Row>
    <Row>
      <Label for="unet_bs">unet_bs: <span class="value">{unet_bs}</span></Label>
      <Input
        type="range"
        name="unet_bs"
        id="unet_bs"
        min={1}
        max={2}
        step={1}
        bind:value={unet_bs}
      />
    </Row>
    <!-- <Row>
      <Label for="sampler">sampler</Label>
      <Input type="select" name="sampler" id="sampler" bind:value={sampler}>
        <option>ddim</option>
        <option>plms</option>
      </Input>
    </Row> -->
    <br />
    <Row>
      <Input
        label="turbo"
        type="switch"
        name="turbo"
        id="turbo"
        bind:checked={turbo}
      />
      <Input
        label="use ddim sampler"
        type="switch"
        name="sampler_useddim"
        id="sampler_useddim"
        bind:checked={sampler_useddim}
      />
    </Row>
  </Col>
</div>

<style>
  .value {
    font-family: monospace;
    color: blue;
    float: right;
  }
</style>
