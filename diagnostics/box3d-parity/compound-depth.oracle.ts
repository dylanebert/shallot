import { expect, setDefaultTimeout, test } from "bun:test";
import { BodyType, defaultShapeDef, defaultSurfaceMaterial } from "../../src/standard/physics/common/types";
import { createCompound } from "../../src/standard/physics/shapes/compound";
import { createCompoundShape } from "../../src/standard/physics/shapes/shape";
import { createTestBody, createTestWorld, destroyTestWorld } from "../../src/standard/physics/testing";
import { bodyGetTransform } from "../../src/standard/physics/world/body";
import { BodyField, bodyField } from "../../src/standard/physics/kernel/bodyrecords";
import { setQueryCallback } from "../../src/standard/physics/kernel/kernel";
import { queryColumns } from "../../src/standard/physics/kernel/querycolumns";
import { nativeSseOutput } from "./native-evidence";
import { assertPublicOracleKernel } from "./oracle-kernel";

setDefaultTimeout(180_000);
await assertPublicOracleKernel();
const native = nativeSseOutput("compound-depth.c", "", ["8 leaves", "1023 leaves", "1024 leaves"])
    .trim()
    .split("\n")
    .map((line) => line.split(" ").map(Number));

function measure(leaves: number) {
    const depth = leaves - 1;
    const compound = createCompound({
        spheres: Array.from({ length: leaves }, (_, i) => ({
            sphere: { center: { x: i === depth ? 0 : 100, y: 0, z: 0 }, radius: 0.5 },
            material: defaultSurfaceMaterial(),
        })),
    })!;
    // Author the retained image before attaching it; queries never mutate its tree.
    const tree = compound.tree;
    tree.ni.fill(0);
    tree.root = 0;
    tree.nodeCount = 2 * depth + 1;
    tree.proxyCount = leaves;
    tree.freeList = -1;
    function node(
        index: number,
        lower: number[],
        upper: number[],
        parent: number,
        height: number,
        flags: number,
    ) {
        const o = index * 12;
        tree.nf.set([...lower, ...upper], o);
        tree.ni[o + 6] = tree.ni[o + 7] = -1;
        tree.ni[o + 10] = parent;
        tree.ni[o + 11] = flags | (height << 16);
    }
    for (let i = 0; i < depth; ++i) {
        node(i, [-0.5, -0.5, -0.5], [100.5, 0.5, 0.5], i - 1, depth - i, 1);
        tree.ni[i * 12 + 8] = depth + i;
        tree.ni[i * 12 + 9] = i + 1 < depth ? i + 1 : 2 * depth;
        node(depth + i, [99.5, -0.5, -0.5], [100.5, 0.5, 0.5], i, 0, 5);
        tree.ni[(depth + i) * 12 + 8] = i;
    }
    node(2 * depth, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], depth - 1, 0, 5);
    tree.ni[2 * depth * 12 + 8] = depth;
    const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        const body = createTestBody(world, { type: BodyType.Static });
        createCompoundShape(world, body, defaultShapeDef(), compound);
        const origin = { x: 0, y: 0, z: 0 };
        const proxy = { points: [origin], count: 1, radius: 0.25 };
        const q = queryColumns(world);
        const k = q.prepare(origin);
        q.proxy(proxy);
        let count = 0;
        const previous = setQueryCallback(world.ecsState, () => {
            ++count;
            return 1;
        });
        try {
            k.worldQuery(world.worldId, 1, 1);
        } finally {
            setQueryCallback(world.ecsState, previous);
        }
        const bodyKernel = q.prepare(origin);
        q.placement(bodyGetTransform(world, body, {
            p: { x: 0, y: 0, z: 0 },
            q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
        }), origin);
        q.proxy(proxy);
        bodyKernel.bodyQuery(
            world.worldId,
            2,
            bodyField(world, body, BodyField.headShapeId),
            0,
        );
        return [leaves, count, Number(q.resultU[0] !== 0xffffffff)];
    } finally {
        destroyTestWorld(world);
    }
}
const actual = native.map((row) => measure(row[0]));
test("compound retained images below the traversal depth bound retain the deepest child", () => {
    expect(actual.slice(0, 2)).toEqual(native.slice(0, 2));
});
test("compound_query.rs:267-301 vs dynamic_tree.c:1140: traversal permits B3_TREE_STACK_SIZE - 1 node pairs, not 1022", () => {
    expect(actual[2]).toEqual(native[2]);
});
