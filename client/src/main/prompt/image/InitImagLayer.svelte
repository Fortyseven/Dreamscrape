<script>
    /*

    */
    // import { onMount } from "svelte";

    import { Layer } from "svelte-canvas";
    import { src_image } from "../../store";

    let src_img = new Image();

    $: if ($src_image) {
        console.log("new init iamge");
        src_img.src = $src_image;
    }

    $: render = ({ context: ctx, width, height }) => {
        ctx.restore();
        console.log("initiagebackgroundlayer");

        ctx.globalCompositeOperation = "source-out";
        ctx.drawImage(src_img, 0, 0, width, height);
        ctx.save();
    };
</script>

<Layer {render} />
