<script>
  import { onMount } from "svelte";

  import { Styles } from "sveltestrap";
  import Main from "./main/index.svelte";
  import { gen_results, snd_finished, snd_error } from "./main/store.js";

  /* ----------------------------------------------*/
  onMount(() => {
    // first play caches
    snd_error();
    snd_finished();

    // restore sessionstorage if available
    let results = window.sessionStorage.getItem("results") || [];

    if (results.length === 0) {
      $gen_results = [];
    } else {
      $gen_results = JSON.parse(results);
    }
  });
</script>

<main>
  <Styles />
  <Main />
</main>
