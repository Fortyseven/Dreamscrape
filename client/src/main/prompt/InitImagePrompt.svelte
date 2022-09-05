<script>
  import { Form, Input, Label, Row } from "sveltestrap";
  import { init_image, strength } from "../store";

  let files;
  let previewImage;

  function onInitFileChange(e) {
    console.log("got a thing", e.target.files[0]);
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.addEventListener("load", function () {
        $init_image = file;
        previewImage.setAttribute("src", reader.result);
      });
      reader.readAsDataURL(file);
    }
  }
</script>

<Form>
  <Label for="file">Init Image</Label>
  <Input
    type="file"
    name="file"
    id="exampleFile"
    bind:files
    on:change={onInitFileChange}
  />
  {#if $init_image}
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
</Form>
<img
  class:invisible_preview={!files}
  bind:this={previewImage}
  id="PreviewImage"
/>

<style>
  #PreviewImage {
    max-width: 100%;
    margin-top: 1em;
  }
  .invisible_preview {
    visibility: hidden;
  }

  .value {
    font-family: monospace;
    color: blue;
    float: right;
  }

  :global(.form-label) {
    display: block !important;
    margin-top: 1em;
  }
</style>
