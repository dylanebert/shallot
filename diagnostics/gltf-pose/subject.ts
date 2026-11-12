import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { MeshInstance, MeshPlugin } from "../../src/core/mesh";
import { TransformPlugin } from "../../src/core/transform";
import { createApp } from "../../src/engine/app";
import { loadGltfScene, poseGltfScene, spawnGltfScene } from "../../src/extras/gltf/scene";
import { GltfPlugin, MorphedMesh, SkinPalette, SkinnedMesh } from "../../src/extras/gltf/skinning";
import { setupGlobals } from "@dylanebert/shallot/webgpu";

const [asset, mode, output] = process.argv.slice(2);
if (!asset || !["check", "time"].includes(mode)) throw new Error("subject requires asset and check|time");
// Node's fetch does not read file URLs; the pinned GLB embeds its buffers and images.
const fetch = globalThis.fetch;
globalThis.fetch = ((input, init) => String(input).startsWith("file:")
    ? Promise.resolve(new Response(readFileSync(new URL(String(input)))))
    : fetch(input, init)) as typeof globalThis.fetch;
await setupGlobals();
// GltfPlugin's palette allocation creates textures when instances attach SkinnedMesh.
// Only setup needs the device: no world.step, queue upload, draw or readback is timed.
const app = await createApp({ defaults: false, plugins: [MeshPlugin, TransformPlugin, GltfPlugin] });
try {
    const world = app.world;
    const gltf = await loadGltfScene(world, pathToFileURL(asset));
    const animation = gltf.animations[0];
    const duration = Math.max(...animation.samplers.map(s => s.input[s.input.length - 1]));
    const population = 2560;
    // Same deterministic phases and placements as the frame oracle.
    const hash = (i: number) => {
        let x = (i + 1) * 0x9e3779b1;
        x ^= x >>> 16;
        x = Math.imul(x, 0x85ebca6b);
        x ^= x >>> 13;
        return (x >>> 0) / 0x100000000;
    };
    const side = Math.ceil(Math.sqrt(population));
    const instances = Array.from({ length: population }, (_, i) => {
        const yaw = hash(i * 2) * Math.PI * 2;
        return spawnGltfScene(world, gltf, {
            translation: [((i % side) - (side - 1) / 2) * 1.25, 0, (Math.floor(i / side) - (side - 1) / 2) * 1.25],
            rotation: [0, Math.sin(yaw / 2), 0, Math.cos(yaw / 2)],
        });
    });
    const phases = instances.map((_, i) => hash(i * 2 + 1) * duration);
    const pose = (frame: number) => {
        const time = frame / 60;
        for (let i = 0; i < population; i++)
            poseGltfScene(world, instances[i], animation, (time + phases[i]) % duration);
    };
    if (mode === "check") {
        const palette = world.resource(SkinPalette);
        const skin = world.storage(SkinnedMesh), morph = world.storage(MorphedMesh);
        const mesh = world.storage(MeshInstance);
        // Check every instance, at both ends and throughout one animation cycle.
        for (let frame = 0; frame <= Math.ceil(duration * 60); frame += 15) {
            pose(frame);
            const values: number[] = [];
            for (const instance of instances) {
                for (const global of instance.globals) values.push(...global);
                for (const p of instance.poses)
                    values.push(...p.translation, ...p.rotation, ...p.scale, ...p.weights);
                for (const eid of instance.entities) {
                    if (world.has(eid, SkinnedMesh)) {
                        const first = skin.firstJoint.get(eid) * 16;
                        values.push(...palette.matrices.subarray(first, first + skin.jointCount.get(eid) * 16));
                    }
                    if (world.has(eid, MorphedMesh)) {
                        const first = morph.firstWeight.get(eid);
                        values.push(...palette.morph.weights.subarray(first, first + morph.weightCount.get(eid)));
                    }
                    const b = mesh.cullBounds;
                    values.push(b.x.get(eid), b.y.get(eid), b.z.get(eid), b.w.get(eid));
                }
            }
            writeFileSync(join(output, `${frame}.bin`), new Uint8Array(new Float64Array(values).buffer));
            console.log(`C ${frame} ${values.length}`);
        }
    } else {
        for (let frame = 0; frame < 120; frame++) pose(frame);
        const samples: number[] = [];
        for (let frame = 120; frame < 360; frame++) {
            const start = performance.now();
            pose(frame);
            samples.push((performance.now() - start) * 1000 / population);
        }
        samples.sort((a, b) => a - b);
        console.log(`T ${JSON.stringify({ median: samples[samples.length / 2], calls: 240 * population })}`);
    }
} finally {
    app.dispose();
}
