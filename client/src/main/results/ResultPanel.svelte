<script>
  import { Button, Col, Figure, Image, Row, Spinner } from "sveltestrap";
  import {
    prompt,
    gen_results,
    is_loading,
    seed,
    result_selected,
  } from "../store";

  let selected = 0;

  function onSelected(index, e) {
    $result_selected = index;
    if (e.ctrlKey) {
      $seed = $gen_results[index].seed;
    }
  }

  function onBigSelected(e) {
    if (e.ctrlKey) {
      $seed = $gen_results[$result_selected].seed;
    }
  }
</script>

{#if !$gen_results.length}
  Ready...
{:else}
  <Row>
    <Col>
      {#each Object.entries($gen_results) as [_, sample], i}
        <span on:click={(e) => onSelected(i, e)}>
          <Figure class="caption" caption={sample.seed}>
            <Image
              width={128}
              thumbnail
              label="Ass"
              src="http://localhost:5000/get_image?{sample.path}#{i}"
            />
          </Figure>
        </span>
      {/each}
    </Col>
  </Row>
  <Row>
    <span on:click={(e) => onBigSelected(e)}>
      <Figure class="caption" caption={$gen_results[$result_selected].seed}>
        <Image
          src="http://localhost:5000/get_image?{$gen_results[$result_selected]
            .path}"
        />
      </Figure>
    </span>
  </Row>
{/if}
