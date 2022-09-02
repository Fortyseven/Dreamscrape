<script>
  import {
    Badge,
    Card,
    Form,
    FormGroup,
    FormText,
    Input,
    Label,
  } from "sveltestrap";

  let prompt = "";
</script>

<FormGroup floating label="Prompt">
  <Input value={prompt} />
  <br />
  <Input type="file" name="file" id="exampleFile" />
  <FormText color="muted">Optionally provide an image to start from.</FormText>
</FormGroup>
