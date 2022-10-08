<script>
    import { onMount } from "svelte";
    import {
        Button,
        ButtonGroup,
        Col,
        Container,
        Icon,
        Row,
        Table,
        Tooltip,
    } from "sveltestrap";
    import * as api from "../api";
    import { bookmarks } from "../store";
    import { promoteBookmarkToSession } from "../utils";
    import BookmarkTable from "./BookmarkTable.svelte";
    import GalleryEntry from "./GalleryEntry.svelte";

    const LIST_MODE = 0;
    const GRID_MODE = 1;

    let view_mode = GRID_MODE;

    /* -------------------------------------- */
    onMount(async () => {
        await api.bookmarks.reload();
    });

    const toggleGrid = () => {
        view_mode = view_mode == GRID_MODE ? LIST_MODE : GRID_MODE;
    };
</script>

<Container>
    <Row>
        <Col>
            <div style="float:right;margin-top:1rem">
                <Button on:click={toggleGrid}>Toggle View</Button>
                <Button on:click={api.bookmarks.reload}>Reload</Button>
            </div>
        </Col>
    </Row>
    {#if bookmarks}
        <Row>
            {#if view_mode === LIST_MODE}
                <BookmarkTable />
            {:else}
                <div id="BookmarkGallery">
                    {#each $bookmarks as entry, i}
                        <GalleryEntry {entry} />
                    {/each}
                </div>
            {/if}
        </Row>
    {/if}
</Container>

<style>
    #BookmarkGallery {
        display: flex;
        flex-wrap: wrap;
    }
</style>
