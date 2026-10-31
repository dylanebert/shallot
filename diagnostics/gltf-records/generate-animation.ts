// THREE=/path/to/three.js bun diagnostics/gltf-records/generate-animation.ts
// Audit the named clauses before accepting regenerated values.
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { accessorFixture, animationFixtures, writtenAnimationFixtures } from "./animation-fixtures";
import manifest from "../../examples/gltf/assets.json";
import { syncFiles } from "../../examples/gltf/sync";

const revision = "1e7e3cff304a05416a54343e7ce273c709bccefe";
const checkout = process.env.THREE;
if (!checkout || Bun.spawnSync(["git", "-C", checkout, "rev-parse", "HEAD"]).stdout.toString().trim() !== revision)
    throw new Error(`THREE must name a checkout at ${revision}`);
const { GLTFLoader } = await import(pathToFileURL(resolve(checkout, "examples/jsm/loaders/GLTFLoader.js")).href);
const { LoadingManager } = await import(pathToFileURL(resolve(checkout, "build/three.module.js")).href);
globalThis.self = globalThis as typeof self;
globalThis.ProgressEvent = class extends Event {} as unknown as typeof ProgressEvent;
globalThis.createImageBitmap = (async () => ({ width: 1, height: 1, close() {} })) as typeof createImageBitmap;
await syncFiles(manifest.models.filter(model => animationFixtures.some(name => name === model.name)).flatMap(model => model.files),
    new URL(`${manifest.commit}/`, manifest.source), resolve(import.meta.dir, "../../examples/gltf/public/assets"));
const records = [];
for (const fixture of [...animationFixtures.map(name => `../../examples/gltf/public/assets/Models/${name}/glTF/${name}.gltf`), ...writtenAnimationFixtures, accessorFixture]) {
    const path = resolve(import.meta.dir, fixture);
    const manager = new LoadingManager();
    manager.setURLModifier((url: string) => url.startsWith("data:") || url.startsWith("blob:") ? url
        : `data:application/octet-stream;base64,${readFileSync(resolve(dirname(path), url)).toString("base64")}`);
    const bytes = readFileSync(path);
    const reference = await new GLTFLoader(manager).parseAsync(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), "");
    const parser = reference.parser;
    const json = parser.json;
    const nodes: any[] = [];
    reference.scene.traverse((object: any) => {
        const index = parser.associations.get(object)?.nodes;
        if (index !== undefined) nodes.push({ index, object,
            translation: object.position.toArray(), rotation: object.quaternion.toArray(), scale: object.scale.toArray(),
            weights: json.nodes[index].weights ?? json.meshes?.[json.nodes[index].mesh]?.weights
                ?? Array(json.meshes?.[json.nodes[index].mesh]?.primitives.find((primitive: any) => primitive.targets?.length)?.targets?.length ?? 0).fill(0) });
    });
    const animations = [];
    for (const [animationIndex, animation] of json.animations.entries()) {
        const divergences: any[] = [];
        const channels: { node: number; path: "translation" | "rotation" | "scale" | "weights"; interpolation: string; input: number[]; interpolant: { evaluate(time: number): Float32Array } }[] = [];
        for (const channel of animation.channels) {
            if (channel.target.node === undefined) continue;
            const sampler = animation.samplers[channel.sampler];
            const node = await parser.getDependency("node", channel.target.node);
            const input = await parser.getDependency("accessor", sampler.input);
            const output = await parser.getDependency("accessor", sampler.output);
            const track = parser._createAnimationTracks(node, input, output, sampler, channel.target)[0];
            // accessor-data-types: snorm8/16 use max(value / positiveMax, -1).
            // GLTFLoader scales signed minima without this clamp.
            if (output.normalized && (output.array instanceof Int8Array || output.array instanceof Int16Array)) {
                const specified = Float32Array.from(track.values, (value: number) => Math.max(-1, value));
                if (specified.some((value: number, index: number) => value !== track.values[index])) {
                    divergences.push({ field: `channels[${channels.length}].output`, clause: "accessor-data-types (signed normalized conversion)",
                        reason: "GLTFLoader _getArrayFromAccessor scales signed minima below -1 without clamping.", loader: Array.from(track.values), specified: Array.from(specified) });
                    track.values = specified;
                }
            }
            channels.push({ node: channel.target.node, path: channel.target.path, interpolation: sampler.interpolation ?? "LINEAR", input: Array.from(input.array) as number[], interpolant: track.createInterpolant() });
        }
        const times = new Set<number>();
        for (const channel of channels) {
            const input = channel.input;
            times.add(input[0] - 1); times.add(input[input.length - 1] + 1);
            input.forEach((t, i) => { times.add(t); if (i) { times.add(input[i - 1] * .75 + t * .25); times.add((input[i - 1] + t) / 2); } });
        }
        const bounded = json.skins?.length > 0;
        const start = Math.min(...channels.map(channel => channel.input[0]));
        const end = Math.max(...channels.map(channel => channel.input[channel.input.length - 1]));
        const sampleTimes = bounded ? [start - 1, start, (start + end) / 2, end, end + 1] : [...times].sort((a,b) => a-b);
        const samples = sampleTimes.map(time => {
            for (const node of nodes) {
                node.object.position.fromArray(node.translation); node.object.quaternion.fromArray(node.rotation); node.object.scale.fromArray(node.scale);
            }
            const values = channels.map(channel => {
                const value = Array.from(channel.interpolant.evaluate(time)) as number[];
                const node = nodes.find(node => node.index === channel.node);
                if (node && channel.path !== "weights") node.object[{ translation: "position", rotation: "quaternion", scale: "scale" }[channel.path as "translation" | "rotation" | "scale"]].fromArray(value);
                return value;
            });
            reference.scene.updateMatrixWorld(true);
            return { time, values, worlds: nodes.map(node => ({ node: node.index, matrix: node.object.matrixWorld.elements.slice() })) };
        });
        animations.push({ index: animationIndex, name: animation.name ?? `Animation${animationIndex}`, channels: channels.map(({ node, path, interpolation }) => ({ node, path, interpolation })), divergences, samples });
    }
    records.push({ fixture, clauses: json.skins?.length ? "transformations; joint-hierarchy; skinned-mesh-primitives: joint globals use parent global * local T * R * S; mesh-node transform is not a skin placement. Sampling bounded to before/start/midpoint/end/after each animation, rather than every key, to bound record size." : "animations; transformations", nodes: nodes.map(({ object, ...node }) => node), animations });
}
writeFileSync(resolve(import.meta.dir, "../../examples/gltf/tests/animations.gold.json"), `${JSON.stringify({
    specification: "a55d8458c1e82720413acd3b420fd09e73b8e40b", generator: revision,
    clauses: { values: "animations (independent sampler clamping; unchanged non-animated properties); appendix-c-interpolation (exact keys, STEP, lerp, shortest-path slerp, Hermite with duration-scaled tangents, normalized interpolated cubic quaternions); accessor-data-types; sparse-accessors",
        nodes: "nodes-and-hierarchy; transformations (default scene, authored local TRS and morph defaults)",
        worlds: "transformations (column-major parent global times local T * R * S); three.js updateMatrixWorld after applying sampled local values; no playback or blending" },
    tolerance: "Samples are stored by three.js interpolants as float32; world matrices are composed as doubles. Consumers admit float32 arithmetic error, not changed formulas. Exact keys and clamped endpoints retain authored values.", records,
}, null, 2).replace(/\[\n\s+[-\d.eE+,\s]+\n\s*\]/g, value => value.replace(/\s+/g, " "))}\n`);
