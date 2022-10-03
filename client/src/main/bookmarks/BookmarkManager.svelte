<script>
    import { onMount } from "svelte";
    import { Button, ButtonGroup, Icon, Table } from "sveltestrap";
    import { promoteBookmarkToSession } from "../utils";
    import { bookmarks } from "../store";
    import * as api from "../api";

    const LIST_MODE = 0;
    const GRID_MODE = 1;

    let view_mode = LIST_MODE;

    /* -------------------------------------- */
    onMount(async () => {
        await api.bookmarks.reload();
    });
</script>

<div style="float:right">
    <Button on:click={api.bookmarks.reload}>Reload</Button>
</div>
{#if bookmarks}
    {#if view_mode === LIST_MODE}
        <div id="BookmarkTable">
            <Table striped hover>
                <thead>
                    <tr>
                        <th>Seed</th>
                        <th>Prompt</th>
                        <th>Steps</th>
                        <th>Scale</th>
                        <th>Strength</th>
                    </tr>
                </thead>
                <tbody>
                    {#each $bookmarks as entry, i}
                        <tr>
                            <th scope="row">{entry.seed}</th>
                            <td>{entry.prompt}</td>
                            <td>{entry.ddim_steps}</td>
                            <td>{entry.scale}</td>
                            <td>{entry.strength || "--"}</td>
                            <td>
                                <img
                                    src="data:image/jpeg;base64,{entry.thumbnail}"
                                />
                            </td>
                            <td>
                                <ButtonGroup>
                                    <Button
                                        on:click={() =>
                                            promoteBookmarkToSession(entry)}
                                    >
                                        <Icon
                                            title="Load Generation Data"
                                            name="box-arrow-in-up-left"
                                        />
                                    </Button>
                                    <Button
                                        on:click={async () => {
                                            await api.bookmarks.rm(entry.id);
                                            await api.bookmarks.reload();
                                        }}
                                        color="danger"
                                    >
                                        <Icon
                                            title="Delete Bookmark"
                                            name="trash"
                                        />
                                    </Button>
                                </ButtonGroup>
                            </td>
                        </tr>
                    {/each}
                </tbody>
            </Table>
        </div>
    {/if}
{/if}

<style>
    #BookmarkTable {
        height: 50vh;
        overflow: scroll;
    }
</style>
