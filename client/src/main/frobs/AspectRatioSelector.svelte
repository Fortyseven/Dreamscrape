<script>
    import {
        Dropdown,
        DropdownItem,
        DropdownMenu,
        DropdownToggle,
    } from "sveltestrap";

    import { height, width } from "../store";

    function setAspect(w, h) {
        $width = w;
        $height = h;
    }
</script>

<Dropdown direction="right" style="align-self: center;">
    <DropdownToggle caret>Aspect</DropdownToggle>
    <DropdownMenu dark>
        <DropdownItem header>SD (512)</DropdownItem>

        <DropdownItem on:click={() => setAspect(512, 512)}>1:1</DropdownItem>
        <DropdownItem on:click={() => setAspect(512, 384)}>4:3</DropdownItem>
        <DropdownItem on:click={() => setAspect(384, 512)}>3:4</DropdownItem>
        <DropdownItem on:click={() => setAspect(512, 287)}>16:9</DropdownItem>
        <DropdownItem on:click={() => setAspect(287, 512)}>9:16</DropdownItem>

        <DropdownItem header>SD+ (768)</DropdownItem>

        <DropdownItem on:click={() => setAspect(768, 768)}>1:1</DropdownItem>
        <DropdownItem on:click={() => setAspect(768, 576)}>4:3</DropdownItem>
        <DropdownItem on:click={() => setAspect(576, 768)}>3:4</DropdownItem>
        <DropdownItem on:click={() => setAspect(768, 431)}>16:9</DropdownItem>
        <DropdownItem on:click={() => setAspect(431, 768)}>9:16</DropdownItem>

        <DropdownItem header>HD (1024)</DropdownItem>

        <DropdownItem on:click={() => setAspect(1024, 1024)}>1:1</DropdownItem>
        <DropdownItem on:click={() => setAspect(1024, 768)}>4:3</DropdownItem>
        <DropdownItem on:click={() => setAspect(768, 1024)}>3:4</DropdownItem>
        <DropdownItem on:click={() => setAspect(1024, 576)}>16:9</DropdownItem>
        <DropdownItem on:click={() => setAspect(576, 1024)}>9:16</DropdownItem>
    </DropdownMenu>
</Dropdown>
