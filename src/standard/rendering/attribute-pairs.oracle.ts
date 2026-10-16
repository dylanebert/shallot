import { test } from "bun:test";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { cube, Meshes, MeshInstance } from "../../core/mesh";
import { Transform } from "../../core/transform";
import { createApp } from "../../engine/app";
import {
    MaterialPlugin,
    Materials,
    MaterialVertexInput,
    MeshMaterial,
    MeshRenderPlugin,
    materialFragmentContext,
    materialLayout,
    materialType,
    materialVertexOutput,
    StandardMaterial,
    StandardRenderingPlugin,
} from "./index";

await setupGlobals();
const Parameters = d.struct({ unused: d.u32 });
const layout = materialLayout(Parameters, { phase: { type: "attribute", element: d.f32 } });
const Vertex = materialVertexOutput();
const Fragment = materialFragmentContext();
const AttributeMaterial = materialType({
    name: "AttributePairOracle",
    layout,
    defaults: { unused: 0 },
    vertex: tgpu.fn(
        [MaterialVertexInput],
        Vertex,
    )((input) => {
        "use gpu";
        return { world: input.world, worldNormal: input.worldNormal };
    }),
    fragment: tgpu.fn(
        [Fragment],
        d.vec4f,
    )(() => {
        "use gpu";
        return d.vec4f(1);
    }),
});

test("measure CPU steps with an attribute material and an unrelated mesh", async () => {
    const warn = console.warn;
    console.warn = (...args: unknown[]) => console.info(...args);
    try {
        for (const count of [7, 1000, 10000]) {
            const app = await createApp({
                defaults: false,
                plugins: [
                    {
                        name: "AttributePairScene",
                        dependencies: [
                            StandardRenderingPlugin,
                            MeshRenderPlugin,
                            MaterialPlugin(AttributeMaterial),
                        ],
                        initialize(world) {
                            const geometry = cube();
                            const tinted = world.resource(Meshes).add({
                                name: "attribute",
                                ...geometry,
                                attributes: {
                                    phase: {
                                        element: d.f32,
                                        data: new Float32Array(geometry.vertices.length / 8),
                                    },
                                },
                            });
                            const plain = world
                                .resource(Meshes)
                                .add({ name: "plain", ...geometry });
                            const attribute = world.resource(AttributeMaterial).add({ unused: 0 });
                            const standard = world
                                .resource(Materials)
                                .add(StandardMaterial({ unlit: true }));
                            for (let i = 0; i < count; i++) {
                                const eid = world.create();
                                world.add(eid, Transform);
                                world.add(eid, MeshInstance, { mesh: i % 2 ? tinted : plain });
                                world.add(eid, MeshMaterial, i % 2 ? attribute : standard);
                            }
                        },
                    },
                ],
            });
            try {
                const { world } = app;
                console.info(
                    `attribute-pairs adapter=${world.gpu.adapter.identity} instances=${count}`,
                );
                const samples: number[] = [];
                for (let round = 0; round < 11; round++) {
                    let elapsed = 0;
                    for (let i = 0; i < 100; i++) {
                        const start = performance.now();
                        world.step(0);
                        elapsed += performance.now() - start;
                        await world.gpu.device.queue.onSubmittedWorkDone();
                    }
                    if (round >= 4) samples.push(elapsed / 100);
                }
                console.info(
                    `attribute-pairs instances=${count} median CPU step ms=${samples.toSorted((a, b) => a - b)[3]}`,
                );
            } finally {
                app.dispose();
            }
        }
    } finally {
        console.warn = warn;
    }
});
