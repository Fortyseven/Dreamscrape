<script>
  import {
    prompt,
    ddim_steps,
    batch_size,
    height,
    width,
    scale,
    ddim_eta,
    seed,
    turbo,
    gen_results,
    is_loading,
    resetStore,
    init_image,
    strength,
    result_selected,
  } from "./store";

  import {
    Container,
    Row,
    Col,
    Card,
    Button,
    ButtonGroup,
    Icon,
  } from "sveltestrap";

  import Frobs from "./frobs/Frobs.svelte";
  import Prompt from "./prompt/Prompt.svelte";
  import ResultPanel from "./results/ResultPanel.svelte";
  import axios from "axios";
  import InitImagePrompt from "./prompt/InitImagePrompt.svelte";
  import FormData from "form-data";
  import WaitingBar from "./ui/WaitingBar.svelte";

  function btnGenerate() {
    const values = {
      prompt: $prompt,
      ddim_steps: $ddim_steps,
      batch_size: $batch_size,
      height: $height,
      width: $width,
      scale: $scale,
      ddim_eta: $ddim_eta,
      seed: $seed,
      turbo: !!$turbo,
      full_precision: false,
      strength: $strength,
      sampler: "plms",
    };

    console.info("Generating samples", values);

    $is_loading = true;

    const fd = new FormData();

    fd.append("file", $init_image);
    for (const k in values) {
      fd.append(k, values[k]);
    }

    axios({
      url: "http://localhost:5000/generate",
      method: "post",
      headers: {
        Accept: "application/json",
        "Content-Type": "multipart/form-data",
      },
      data: fd,
      //   data: { ...values, ...fd },
    })
      .then(function (response) {
        $gen_results = response.data;
        $is_loading = false;
        $result_selected = 0;
        console.info("Response", response.data);
      })
      .catch(function (error) {
        console.log("ERR", error);
        $is_loading = false;
      });
  }

  function btnReset() {
    resetStore();
  }
</script>

<Container fluid>
  <Row width="100%">
    <Col xs={12} lg={5}>
      <Prompt on:generate={btnGenerate} />
      <br />
      <Col>
        <Row>
          <ButtonGroup>
            <Button title="Reset" color="secondary" on:click={btnReset}>
              <Icon name="arrow-repeat" />
            </Button>
            <Button
              style="width:75%"
              title="Generate"
              color="primary"
              on:click={btnGenerate}
              disabled={$is_loading || !$prompt.length}
            >
              <Icon name="play-fill" />
            </Button>
          </ButtonGroup>
        </Row>
        {#if $is_loading}
          <Row>
            <WaitingBar style="padding: 1em" />
          </Row>
        {/if}
      </Col>
      <br />
      <Card>
        <Frobs />
      </Card>
      <br />
      <Card>
        <InitImagePrompt />
      </Card>
    </Col>
    <Col>
      <ResultPanel />
    </Col>
  </Row>
</Container>
