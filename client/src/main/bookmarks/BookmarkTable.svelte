<script>
    import { onMount } from "svelte";
    import { Button, ButtonGroup, Icon, Table } from "sveltestrap";
    import * as api from "../api";
    import { bookmarks } from "../store";
    import { promoteBookmarkToSession } from "../utils";

    /* -------------------------------------- */
    onMount(async () => {
        await api.bookmarks.reload();
    });
</script>

{#if bookmarks}
    <div id="BookmarkTable">
        <Table striped hover>
            <thead>
                <tr>
                    <th>Seed</th>
                    <th>Prompt</th>
                    <th>Steps</th>
                    <th>Scale</th>
                    <th>Strength</th>
                    <th>Src</th>
                    <th>Image</th>
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
                            {#if entry.src_hash}
                                <a
                                    href="{api.SERVER_URL}/media?hash={entry.src_hash}"
                                    target="_blank"
                                >
                                    <!-- svelte-ignore a11y-missing-attribute -->
                                    <img
                                        src="{api.SERVER_URL}/media?hash={entry.src_hash}&thumb=1"
                                    />
                                </a>
                            {/if}
                        </td>
                        <td>
                            <a
                                href="{api.SERVER_URL}/media?hash={entry.img_hash}"
                                target="_blank"
                            >
                                <!-- svelte-ignore a11y-missing-attribute -->
                                <img
                                    src="{api.SERVER_URL}/media?hash={entry.img_hash}&thumb=1"
                                />
                            </a>
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

<style>
    #BookmarkTable {
        height: 50vh;
        overflow: scroll;
    }
</style>
