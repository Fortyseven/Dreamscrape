<script>
    import { onMount, createEventDispatcher } from "svelte";

    import { FormGroup, Input } from "sveltestrap";
    import { seed, prompt, is_loading } from "../store";
    import WaitingBar from "../ui/WaitingBar.svelte";

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

<FormGroup floating label="Prompt">
    <Input
        bind:value={$prompt}
        disabled={$is_loading}
        on:keydown={onPromptKeydown}
        bind:inner
    />
</FormGroup>

<FormGroup floating label="Seed">
    <Input
        bind:value={$seed}
        disabled={$is_loading}
        on:keydown={onPromptKeydown}
    />
</FormGroup>
