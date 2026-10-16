import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { shallot } from "@dylanebert/shallot/bun";
import { plugin } from "bun";

plugin(shallot({ root: import.meta.dir }));
const { createApp } = await import("@dylanebert/shallot");
const { attachTexture, captureTexture, Camera, Anchor } = await import(
    "@dylanebert/shallot/rendering"
);
const { Transform } = await import("@dylanebert/shallot/transform");
const { CameraBackground, StandardRenderer } = await import(
    "@dylanebert/shallot/standard/rendering"
);
const { Sprite, SpriteAlphaMode, SpriteFill, SpritePlugin, Sky, SkyPlugin, skyBackgroundHandle } =
    await import("@dylanebert/shallot/extras");
const { setupGlobals } = await import("@dylanebert/shallot/webgpu");
const { Gallery, GalleryPlugin } = await import("../examples/materials/src/gallery");
const { Clock, CyclePlugin, CycleScene } = await import("../examples/day-night-cycle/src/cycle");
await setupGlobals();
const directory = process.argv[2];
if (!directory) throw new Error("Usage: bun scripts/frame.oracle.ts <directory> [prior-directory]");
mkdirSync(directory, { recursive: true });
for (const subject of ["day-night", "sky", "materials", "sprite"]) {
    const plugins =
        subject === "day-night"
            ? [CyclePlugin]
            : subject === "sky"
              ? [SkyPlugin]
              : subject === "sprite"
                ? [SpritePlugin]
                : [GalleryPlugin];
    const app = await createApp({ plugins });
    const { world } = app;
    console.log(world.gpu.adapter);
    try {
        let camera: number;
        if (subject === "sprite") {
            camera = world.create();
            world.add(camera, Transform, { translation: [0, 0, 5] });
            world.add(camera, Camera);
            world.add(camera, StandardRenderer);
            const atlas = world.gpu.device.createTexture({
                size: [2, 2, 1],
                format: "rgba8unorm-srgb",
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            });
            world.own(atlas);
            world.gpu.device.queue.writeTexture(
                { texture: atlas },
                new Uint8Array([
                    255, 255, 255, 255, 255, 255, 255, 80, 255, 255, 255, 160, 255, 255, 255, 0,
                ]),
                { bytesPerRow: 8 },
                [2, 2, 1],
            );
            world.gpu.textures.set("images", atlas);
            for (const blend of [0, 1]) {
                const alphaMode = blend === 0 ? SpriteAlphaMode.Mask() : SpriteAlphaMode.Blend;
                for (const fillMethod of Object.values(SpriteFill)) {
                    const sprite = world.create();
                    world.add(sprite, Transform, {
                        translation: [(fillMethod - 1.5) * 1.1, blend - 0.5, 0],
                    });
                    world.add(sprite, Anchor, { offset: [-0.25, 0.25] });
                    world.add(sprite, Sprite, {
                        size: [0.8, 0.8],
                        alphaMode,
                        fillMethod,
                        fillAmount: 0.65,
                        color: 0xffaa55,
                        opacity: 0.8,
                    });
                }
            }
        } else if (subject === "sky") {
            camera = world.create();
            world.add(camera, Transform);
            world.add(camera, Camera);
            world.add(camera, StandardRenderer);
            world.add(camera, CameraBackground, { background: skyBackgroundHandle(world) });
            world.add(world.create(), Sky);
        } else {
            camera =
                subject === "day-night"
                    ? world.resource(CycleScene).camera
                    : world.resource(Gallery).camera;
        }
        attachTexture(world, camera, { width: 320, height: 240 });
        const hours = subject === "day-night" ? [0, 6, 10, 12, 18, 22] : [null];
        for (const hour of hours) {
            const name = hour === null ? subject : `${subject}-${hour}`;
            if (hour !== null) world.resource(Clock).hour = hour;
            world.step(0);
            world.step(0);
            const { rgba } = await captureTexture(world, camera);
            if (process.argv[3]) {
                const before = readFileSync(`${process.argv[3]}/${name}.rgba`);
                if (before.length !== rgba.length) throw new Error(`${name}: frame sizes differ`);
                const max = [0, 0, 0, 0];
                for (let i = 0; i < rgba.length; i++)
                    max[i % 4] = Math.max(max[i % 4]!, Math.abs(rgba[i]! - before[i]!));
                console.log({ subject: name, max });
            }
            writeFileSync(`${directory}/${name}.rgba`, rgba);
        }
    } finally {
        app.dispose();
    }
}
process.exit(0);
