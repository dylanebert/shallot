import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { composeMat4, multiplyMat4 } from "../../src/engine/utils";
import { evaluateGltfAnimation, type GltfJson, parse } from "../../src/extras/gltf";
import gold from "./animations.gold.json";
import { syncAssets } from "../../examples/gltf/tests/setup";

await syncAssets();

const subjects = gold.records.map(record => {
    const path = resolve(import.meta.dir, record.fixture);
    const json = JSON.parse(readFileSync(path, "utf8")) as GltfJson;
    const buffers = (json.buffers ?? []).map(buffer => {
        const bytes = buffer.uri!.startsWith("data:") ? Buffer.from(buffer.uri!.split(",")[1], "base64")
            : readFileSync(resolve(dirname(path), buffer.uri!));
        return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    });
    return { record, scene: parse(json, buffers) };
});

// Float32 accessor/interpolant storage and Float32 world composition, relative to magnitude.
function close(actual: ArrayLike<number>, expected: ArrayLike<number>): void {
    expect(actual.length).toBe(expected.length);
    for (let i = 0; i < actual.length; i++)
        expect(Math.abs(actual[i] - expected[i])).toBeLessThanOrEqual(3e-5 * Math.max(1, Math.abs(expected[i])));
}

for (const mode of ["STEP", "LINEAR", "CUBICSPLINE", "clamping"]) {
    test(`recorded animation ${mode} follows Animations / Appendix C`, () => {
        let assertions = 0;
        for (const { record, scene } of subjects) {
            for (const animation of record.animations) {
                const decoded = scene.animations[animation.index];
                for (const sample of animation.samples) {
                    const poses = evaluateGltfAnimation(scene.nodes, decoded, sample.time);
                    animation.channels.forEach((channel, i) => {
                        const sampler = decoded.samplers[decoded.channels[i].sampler];
                        const outside = sample.time < sampler.input[0] || sample.time > sampler.input[sampler.input.length - 1];
                        if (mode === "clamping" ? !outside : outside || channel.interpolation !== mode) return;
                        const index = scene.nodes.findIndex(node => node.index === channel.node);
                        close(poses[index][decoded.channels[i].path], sample.values[i]);
                        assertions++;
                    });
                }
            }
        }
        expect(assertions).toBeGreaterThan(0);
    });
}

test("recorded default-scene nodes retain authored local values and parent-first order", () => {
    for (const { record, scene } of subjects) {
        expect(scene.nodes.map(node => node.index)).toEqual(record.nodes.map(node => node.index));
        scene.nodes.forEach((node, i) => {
            expect(node.parent).toBeLessThan(i);
            const expected = record.nodes[i];
            for (const path of ["translation", "rotation", "scale", "weights"] as const) close(node[path], expected[path]);
        });
        expect(scene.unsupported).not.toContain("animation");
    }
});

test("sampled local poses compose to recorded node world matrices", () => {
    for (const { record, scene } of subjects) {
        for (const animation of record.animations) {
            for (const sample of animation.samples) {
                const poses = evaluateGltfAnimation(scene.nodes, scene.animations[animation.index], sample.time);
                const worlds: Float32Array[] = [];
                poses.forEach((pose, i) => {
                    const { translation: t, rotation: r, scale: s } = pose;
                    const local = composeMat4(t[0], t[1], t[2], r[0], r[1], r[2], r[3], s[0], s[1], s[2]);
                    const parent = scene.nodes[i].parent;
                    worlds.push(parent === -1 ? local : multiplyMat4(worlds[parent], local));
                    close(worlds[i], sample.worlds[i].matrix);
                    for (const path of ["translation", "rotation", "scale", "weights"] as const)
                        if (!animation.channels.some(channel => channel.node === scene.nodes[i].index && channel.path === path))
                            expect(pose[path]).toEqual(scene.nodes[i][path]);
                });
            }
        }
    }
});
