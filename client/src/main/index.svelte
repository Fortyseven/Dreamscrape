<script>
  // @ts-nocheck

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
    snd_error,
    snd_finished,
    prompt_log,
  } from "./store";

  import {
    Container,
    Row,
    Col,
    Card,
    Button,
    ButtonGroup,
    Icon,
    Toast,
    Modal,
    TabPane,
    TabContent,
  } from "sveltestrap";

  import Frobs from "./frobs/Frobs.svelte";
  import Prompt from "./prompt/Prompt.svelte";
  import ResultPanel from "./results/ResultPanel.svelte";
  import axios from "axios";
  import FormData from "form-data";
  import WaitingBar from "./ui/WaitingBar.svelte";
  import Log from "./log/Log.svelte";
  import InitImagePrompt from "./prompt/InitImagePrompt.svelte";
  import BookmarkManager from "./bookmarks/BookmarkManager.svelte";

  let hasError = false;
  let errMessage = "";

  /* -------------------- */
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

    console.debug("Generating samples", values);

    const fd = new FormData();

    if ($init_image !== undefined) {
      //   console.debug("init_image [size]", $init_image?.length);
      fd.append("init_image", $init_image);
    }

    for (const k in values) {
      fd.append(k, values[k]);
    }

    $is_loading = true;

    axios({
      url: "http://localhost:5000/generate",
      method: "post",
      headers: {
        Accept: "application/json",
        "Content-Type": "multipart/form-data",
      },
      data: fd,
    })
      .then(function (response) {
        console.debug("GENERATED RESPONSE", response.data);
        snd_finished();
        $result_selected = 0;
        $gen_results = response.data;
        $is_loading = false;
        prependToSessionLog(response.data);
        window.sessionStorage.setItem("results", JSON.stringify(response.data));
      })
      .catch(function (error) {
        console.error("GENERATE ERR", error);
        snd_error();
        hasError = true;
        errMessage = `${error.message}`;
        $is_loading = false;
      });
  }

  /* -------------------- */
  function prependToSessionLog(entries) {
    for (let i in entries) {
      let temp = $prompt_log;

      temp.unshift(entries[i]);

      $prompt_log = temp;
    }
  }

  /* -------------------- */
  function btnReset() {
    resetStore();
  }

  let open = false;
  const toggle = () => (open = !open);

  /* -------------------- */
  document.onpaste = function (event) {
    var items = (event.clipboardData || event.originalEvent.clipboardData)
      .items;

    // console.log(JSON.stringify(items)); // might give you mime types

    for (var index in items) {
      var item = items[index];
      if (item.kind === "file") {
        var blob = item.getAsFile();
        var reader = new FileReader();
        reader.onload = function (event) {
          $init_image = event.target.result;
          //   console.log(event.target.result); // data url!
        };
        reader.readAsDataURL(blob);
      }
    }
  };
</script>

<div id="AppWrapper">
  <div id="Left">
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
        <Modal
          body
          header="Error"
          isOpen={hasError}
          toggle={() => (toggle(), (hasError = false))}
          on:close={() => (hasError = false)}
        >
          {errMessage}
        </Modal>
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
    <InitImagePrompt />
  </div>
  <div id="Right">
    <TabContent>
      <TabPane tabId="generated" tab="🖼 Generated" active>
        <ResultPanel />
      </TabPane>
      <TabPane tabId="log" tab="👨‍💻 Session Log">
        <Log />
      </TabPane>
      <TabPane tabId="results" tab="🔖 Bookmarks">
        <BookmarkManager />
      </TabPane>
    </TabContent>
  </div>
</div>

<style>
  #AppWrapper {
    display: flex;
  }
  #Left {
    width: 512px;
    margin: 0 1em;
  }
  #Right {
    width: 100%;
    margin: 0 1em;
  }
</style>
