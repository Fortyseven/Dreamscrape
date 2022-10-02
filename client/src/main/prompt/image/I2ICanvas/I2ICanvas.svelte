<script>
    // @ts-nocheck

    import { onMount } from "svelte";

    import { Canvas, Layer, t } from "svelte-canvas";
    import { src_image, mask_image, strength } from "../../../store";
    import InitImageLayer from "../InitImagLayer.svelte";

    let canvas = null;
    let ctx = null;

    let mask_framebuffer = null;

    let is_pen_down = false;
    let penPos = [null, null];
    let prev = [null, null];

    onMount(() => {
        ctx = canvas.getCanvas().getContext("2d");
        console.log("onmount", ctx);
        mask_framebuffer = ctx.createImageData(512, 512);
        // for (let i = 0; i < mask_framebuffer.data.length; i += 4) {
        //     // Modify pixel data
        //     mask_framebuffer.data[i + 0] = 190; // R value
        //     mask_framebuffer.data[i + 1] = 0; // G value
        //     mask_framebuffer.data[i + 2] = 210; // B value
        //     mask_framebuffer.data[i + 3] = 255; // A value
        // }
        console.log(
            "created mask fb",
            mask_framebuffer,
            mask_framebuffer.length
        );
        //
    });

    const onMouseDown = function () {
        is_pen_down = true;
    };

    const onMouseUp = function () {
        is_pen_down = false;
        prev = [null, null];
    };

    const onMouseMove = function (ev) {
        const rect = canvas.getCanvas().getBoundingClientRect();
        // console.log("MOVEEV", ev);
        const x = ev.clientX - Math.floor(rect.left);
        const y = ev.clientY - Math.floor(rect.top);

        if (is_pen_down) {
            penPos[0] = x;
            penPos[1] = y;
            const offset = (penPos[1] * mask_framebuffer.width + penPos[0]) * 4;

            // mask_framebuffer.data[offset + 0] = 0;
            // mask_framebuffer.data[offset + 1] = 0;
            // mask_framebuffer.data[offset + 2] = 0;
            // mask_framebuffer.data[offset + 3] = 0;
            // // mask_framebuffer.data[offset + 1] = 0xff;
            // // mask_framebuffer.data[offset + 2] = 0xff;
            // // mask_framebuffer.data[offset + 3] = 0xff;
            // console.log("pendownmove", offset);
            // console.log("mask_framebuffer", mask_framebuffer.data);
            canvas.redraw();
        }
    };

    const onKeyDown = () => {
        //
    };

    const maskRender = ({ context, width, height }) => {
        if (mask_framebuffer) {
            console.log("redrawing");
            context.putImageData(mask_framebuffer, 0, 0);
        }
    };
</script>

<!-- ################################################################# -->
{penPos}{is_pen_down}
<Canvas
    bind:this={canvas}
    width={512}
    height={512}
    autoclear={true}
    on:mousedown={onMouseDown}
    on:mouseup={onMouseUp}
    on:mouseout={onMouseUp}
    on:mousemove={onMouseMove}
    on:keypress={onKeyDown}
>
    <!-- <InitImageLayer /> -->
    <Layer render={maskRender} />
    <!-- <Layer {render} /> -->
</Canvas>
