import { test } from "bun:test";
import { createApp, Meshes, MeshPlugin } from "@dylanebert/shallot";
import { GltfPlugin, loadGltf, loadGltfScene } from "@dylanebert/shallot/gltf";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { assets, fileFetch, syncAssets } from "../../examples/gltf/tests/setup";

test("quantized fixture base positions through core unorm16 storage", async () => {
await syncAssets();
await setupGlobals();
const originalFetch = globalThis.fetch;
globalThis.fetch = fileFetch;
try {
    for (const name of ["Avocado", "Duck", "Lantern", "AnimatedMorphCube"]) {
        const url = new URL(`Models/${name}/glTF-Quantized/${name}.gltf`, assets);
        const source = await loadGltf(url, fileFetch);
        let model: Awaited<ReturnType<typeof loadGltfScene>>;
        const app = await createApp({ defaults: false, plugins: [MeshPlugin, {
            name: "QuantizationMeasurement",
            dependencies: [GltfPlugin],
            async initialize(world) { model = await loadGltfScene(world, url); },
        }] });
        try {
            let maxError = 0, changed = 0, components = 0, recovered = 0;
            const maxAxisError = [0, 0, 0];
            for (const [p, primitive] of model!.primitives.entries()) {
                const mesh = app.world.resource(Meshes).get(primitive.mesh)!;
                const vertices = await mesh.vertices.read();
                const indices = await mesh.indices.read();
                const id = vertices[indices[mesh.indexBase]].y >>> 16;
                const q = (await mesh.quant!.read())[id];
                const local = vertices.slice(0, q.streams || vertices.length).filter((v) => v.y >>> 16 === id);
                const offset = [q.posOffset.x, q.posOffset.y, q.posOffset.z];
                const extent = [q.posScale.x, q.posScale.y, q.posScale.z];
                for (const [v, words] of local.entries()) {
                    const codes = [words.x & 65535, words.x >>> 16, words.y & 65535];
                    for (let a = 0; a < 3; a++) {
                        const expected = source.meshes[source.instances[p].mesh].vertices[v * 8 + a];
                        const decoded = Math.fround(offset[a] + Math.fround(Math.fround(codes[a] / 65535) * extent[a]));
                        const error = Math.abs(decoded - expected);
                        maxError = Math.max(maxError, error);
                        maxAxisError[a] = Math.max(maxAxisError[a], error);
                        changed += Number(decoded !== expected);
                        recovered += Number(Math.round(decoded) === expected);
                        components++;
                    }
                }
            }
            console.log(JSON.stringify({ name, adapter: app.world.gpu.adapter, components, changed, recovered, maxError, maxAxisError }));
            if (recovered !== components) throw new Error(`${name}: unorm16 loses an integer base-position code`);
        } finally { app.dispose(); }
    }
} finally { globalThis.fetch = originalFetch; }
});
