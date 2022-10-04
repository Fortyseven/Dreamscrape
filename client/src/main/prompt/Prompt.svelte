<script>
    import { onMount, createEventDispatcher } from "svelte";

    import { Col, FormGroup, Input, Row } from "sveltestrap";
    import { seed, prompt, prompt_extra, is_loading } from "../store";

    const dispatch = createEventDispatcher();
    let inner;

    onMount(async () => {
        inner.focus();
    });

    function onPromptKeydown(e) {
        if (e.keyCode === 13) {
            dispatch("generate");
        }
    }
</script>

<Row>
    <Col>
        <FormGroup floating label="Prompt">
            <Input
                bind:value={$prompt}
                disabled={$is_loading}
                on:keydown={onPromptKeydown}
                bind:inner
            />
        </FormGroup>
    </Col>
</Row>
<Row>
    <Col>
        <FormGroup floating label="Persistent Style">
            <Input
                bind:value={$prompt_extra}
                disabled={$is_loading}
                on:keydown={onPromptKeydown}
                bind:inner
            />
        </FormGroup>
    </Col>
    <Col xs="4">
        <FormGroup floating label="Seed">
            <Input
                bind:value={$seed}
                disabled={$is_loading}
                on:keydown={onPromptKeydown}
            />
        </FormGroup>
    </Col>
</Row>
