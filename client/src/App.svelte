<script>
  import { onMount } from "svelte";

  import { Styles, Button, Container } from "sveltestrap";
  import Main from "./main/index.svelte";
  import { snd_finished, snd_error } from "./main/store.js";

  onMount(() => {
    // first play caches
    snd_error();
    snd_finished();
  });
</script>

<main>
  <Styles />
  <Main />
</main>
