// THREE=/path/to/three.js bun diagnostics/gltf-records/generate-lights.ts
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import manifest from "../../examples/gltf/assets.json";
import { syncFiles } from "../../examples/gltf/sync";

const revision = "1e7e3cff304a05416a54343e7ce273c709bccefe";
const checkout = process.env.THREE;
if (!checkout || Bun.spawnSync(["git", "-C", checkout, "rev-parse", "HEAD"]).stdout.toString().trim() !== revision)
    throw new Error(`THREE must name a checkout at ${revision}`);
const { GLTFLoader } = await import(pathToFileURL(resolve(checkout, "examples/jsm/loaders/GLTFLoader.js")).href);
const { LoadingManager, AnimationMixer, LoopOnce } = await import(pathToFileURL(resolve(checkout, "build/three.module.js")).href);
globalThis.self = globalThis as typeof self;
globalThis.ProgressEvent = class extends Event {} as unknown as typeof ProgressEvent;
globalThis.createImageBitmap = (async () => ({ width: 1, height: 1, close() {} })) as typeof createImageBitmap;
const names = ["LightsPunctualLamp", "PointLightIntensityTest", "DirectionalLight"];
await syncFiles(manifest.models.filter(model => names.includes(model.name)).flatMap(model => model.files),
    new URL(`${manifest.commit}/`, manifest.source), resolve(import.meta.dir, "../../examples/gltf/public/assets"));
const records = [];
for (const name of [...names, "PunctualLights"]) {
    const path = resolve(import.meta.dir, name === "PunctualLights" ? "../../examples/gltf/public/models/PunctualLights.gltf" : `../../examples/gltf/public/assets/Models/${name}/glTF/${name}.gltf`);
    const manager = new LoadingManager();
    manager.setURLModifier((url: string) => url.startsWith("data:") || url.startsWith("blob:") ? url
        : `data:application/octet-stream;base64,${readFileSync(resolve(dirname(path), url)).toString("base64")}`);
    const reference = await new GLTFLoader(manager).parseAsync(readFileSync(path, "utf8"), "");
    const json = reference.parser.json;
    reference.scene.updateMatrixWorld(true);
    const nodes: { node: number; matrix: number[] }[] = [];
    reference.scene.traverse((object: any) => {
        const node = reference.parser.associations.get(object)?.nodes;
        if (node !== undefined) nodes.push({ node, matrix: object.matrixWorld.elements.slice() });
    });
    const mixer = new AnimationMixer(reference.scene);
    const animations = reference.animations.map((clip: any, index: number) => {
        const samples = [-1, 0, 0.5, 1, 2].map(time => {
            mixer.stopAllAction();
            const action = mixer.clipAction(clip).setLoop(LoopOnce, 1);
            action.clampWhenFinished = true;
            action.reset().play();
            // glTF evaluates without playback: clamp before asking the reference mixer to pose.
            mixer.setTime(Math.max(0, Math.min(time, clip.duration)));
            reference.scene.updateMatrixWorld(true);
            const worlds: { node: number; matrix: number[] }[] = [];
            reference.scene.traverse((object: any) => {
                const node = reference.parser.associations.get(object)?.nodes;
                if (node !== undefined) worlds.push({ node, matrix: object.matrixWorld.elements.slice() });
            });
            return { time, worlds };
        });
        mixer.stopAllAction();
        return { index, samples };
    });
    records.push({ name, nodes, animations });
}
writeFileSync(resolve(import.meta.dir, "../../examples/gltf/tests/lights.gold.json"), `${JSON.stringify({ revision, records }, null, 2)}\n`);
