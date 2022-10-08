<script>
    import {
        Dropdown,
        DropdownItem,
        DropdownMenu,
        DropdownToggle,
        Row,
    } from "sveltestrap";

    import { prompt_extra } from "../store";

    const sections = {
        "Quick Styles": [
            "35mm, photography, detailed, dramatic, cinematic",
            "art by greg rutkowski, professional lighting, deviantart, artstation, cinematic, dramatic",
            "photograph, sharp focus, very detailed, nikon d850",
            "highly detailed painting by gaston bussiere, craig mullins, j. c. leyendecker",
        ],
        "Junk Drawer": [
            "chromatic aberration",
            "daguerreotype",
            "instamatic",
            "polaroid",
            "35mm",
            "8mm",
            "vintage color photo",
        ],
        "By...": [
            "by greg rutkowski",
            ">Artists",
            "by Andy Warhol",
            "by Drew Struzan",
            "by Kate Moross",
            "by Milton Glaser",
            "by Saul Bass",
            ">Comics",
            "by Frank Miller",
            "by Jack Kirby",
            "by Jim Lee",
            "by Neal Adams",
            "by Steve Ditko",
            ">Photography",
            "by Ansel Adams",
            "by Elliott Erwitt",
            "by Garry Winogrand",
            "by Vivian Maier",
            ">Graffiti",
            "by Banksy",
            "by Keith Haring",
        ],
    };

    function appendPromptExtra(str, prompt_replace = false, ev) {
        if (prompt_replace || ev.shiftKey || $prompt_extra.length == 0) {
            $prompt_extra = str;
        } else if (!$prompt_extra.includes(str)) {
            $prompt_extra = [$prompt_extra, str].join(", ");
        }
    }
</script>

<div style="display:flex;">
    {#each Object.entries(sections) as [button_title, entries], index}
        <Dropdown style="align-self: center;" size="sm">
            <DropdownToggle caret>{button_title}</DropdownToggle>
            <DropdownMenu dark>
                {#each entries as entry}
                    {#if entry.startsWith(">")}
                        <DropdownItem divider />
                        <DropdownItem header>
                            {entry.substr(1)}
                        </DropdownItem>
                    {:else}
                        <DropdownItem
                            on:click={(ev) =>
                                appendPromptExtra(entry, index === 0, ev)}
                        >
                            {entry}
                        </DropdownItem>
                    {/if}
                {/each}
            </DropdownMenu>
        </Dropdown>
        &nbsp;
    {/each}
</div>
