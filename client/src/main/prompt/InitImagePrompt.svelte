<script>
  import {
    Button,
    ButtonGroup,
    Form,
    Icon,
    Input,
    Label,
    Row,
  } from "sveltestrap";
  import { init_image, strength } from "../store";

  let files;
  //   let init_image_el;
  let input_field;

  function onInitFileChange(e) {
    input_field = e;

    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.addEventListener("load", function () {
        // $init_image = file;
        // previewImage.setAttribute("src", reader.result);
        $init_image = reader.result;
        console.log("file upload", $init_image);
      });
      reader.readAsDataURL(file);
    }
  }

  function removeImage() {
    $init_image = undefined;
    // init_image_el = undefined;
    $strength = 0.5;
    files = undefined;
    if (input_field && input_field.srcElement)
      input_field.srcElement.value = "";
  }
</script>

<div>
  <Label for="file">Init Image</Label>
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
</div>
<!-- {#if files} -->
<!-- svelte-ignore a11y-missing-attribute -->
<img
  class:invisible_preview={$init_image === undefined}
  id="PreviewImage"
  src={$init_image}
/>

<!-- {/if} -->
<style>
  img {
    max-width: 100%;
    margin-top: 1em;
    background: #f0f;
    border: 1px solid red;
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
