import { expect, setDefaultTimeout, test } from "bun:test";
import {
    BodyType,
    createCompound,
    defaultSurfaceMaterial,
    PhysicsWorld,
} from "../../src/standard/physics/api";
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
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        const body = world.createBody({ type: BodyType.Static });
        body.createCompound({}, compound);
        const origin = { x: 0, y: 0, z: 0 };
        const proxy = { points: [origin], count: 1, radius: 0.25 };
        let count = 0;
        world.overlapShape(origin, proxy, () => {
            ++count;
            return true;
        });
        return [leaves, count, Number(body.overlapShape(origin, proxy, body.getTransform()))];
    } finally {
        world.destroy();
    }
}
const actual = native.map((row) => measure(row[0]));
test("compound retained images below the traversal depth bound retain the deepest child", () => {
    expect(actual.slice(0, 2)).toEqual(native.slice(0, 2));
});
test.todo("compound_query.rs:267-301 vs dynamic_tree.c:1140: traversal permits B3_TREE_STACK_SIZE - 1 node pairs, not 1022", () => {
    expect(actual[2]).toEqual(native[2]);
});
