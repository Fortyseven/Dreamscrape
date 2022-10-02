<script>
    import { Button, Icon, Table } from "sveltestrap";
    import { prompt_log } from "../store";
    import { bookmarks } from "../api";
    /* -------------------------------------- */
</script>

<Table striped>
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
        {#each $prompt_log as entry, i}
            <!-- content here -->
            <tr>
                <th scope="row">{entry.seed}</th>
                <td>{entry.prompt}</td>
                <td>{entry.ddim_steps}</td>
                <td>{entry.scale}</td>
                <td>{entry.strength || "--"}</td>
                <td>
                    <Button on:click={() => bookmarks.create(entry)}>
                        <Icon title="Bookmark This" name="bookmark" />
                    </Button>
                </td>
            </tr>
        {/each}
    </tbody>
</Table>
