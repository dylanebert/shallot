import { expect, setDefaultTimeout, test } from "bun:test";
import { readFileSync } from "node:fs";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { World } from "../../engine";
import { treeBox, treeSweep } from "./fixtures/tree";
import {
    createPhysicsTree,
    createPhysicsTreeProxy,
    destroyPhysicsTree,
    getPhysicsTreeNode,
    getPhysicsTreeStats,
    loadPhysicsTree,
    type PhysicsTreeRayHit,
    queryPhysicsTree,
    rayCastPhysicsTree,
    StandardPhysicsPlugin,
    savePhysicsTree,
    validatePhysicsTree,
} from "./index";
import { kernel } from "./kernel/kernel";

setDefaultTimeout(TIMEOUT.node);
async function subject(run: (world: World) => void): Promise<void> {
    const world = new World();
    await StandardPhysicsPlugin.initialize!(world);
    try {
        run(world);
    } finally {
        await StandardPhysicsPlugin.dispose!(world);
        world.dispose();
    }
}
test("independent tree construction, rebuilds, nodes, statistics, filtered callbacks and scaled save/load equal native Box3D", async () => {
    await subject((world) => {
        expect(treeSweep(world)).toBe(
            readFileSync(new URL("./tree.gold.txt", import.meta.url), "utf8"),
        );
    });
});
test("tree ownership and borrowed records are explicit", async () => {
    await subject((world) => {
        const a = createPhysicsTree(world),
            b = createPhysicsTree(world);
        const id = createPhysicsTreeProxy(world, a, treeBox(0), 1n, 9n);
        expect(getPhysicsTreeStats(world, b).proxyCount).toBe(0);
        const stats = getPhysicsTreeStats(world, a);
        expect(getPhysicsTreeStats(world, a)).toBe(stats);
        const node = getPhysicsTreeNode(world, a, id);
        expect(getPhysicsTreeNode(world, a, id)).toBe(node);
        const callback = () => true;
        const result = queryPhysicsTree(world, a, treeBox(0), callback);
        expect(queryPhysicsTree(world, a, treeBox(0), callback)).toBe(result);
        destroyPhysicsTree(world, a);
        expect(() => destroyPhysicsTree(world, a)).toThrow("not live");
        const foreign = new World();
        try {
            expect(() => getPhysicsTreeStats(foreign, b)).toThrow("not live");
        } finally {
            foreign.dispose();
        }
        // World disposal owns b.
    });
});
test("tree callbacks propagate errors, restore callbacks and permit queries of other trees", async () => {
    await subject((world) => {
        const a = createPhysicsTree(world),
            b = createPhysicsTree(world);
        createPhysicsTreeProxy(world, a, treeBox(0));
        createPhysicsTreeProxy(world, b, treeBox(0));
        const error = new Error("callback failed");
        expect(() =>
            queryPhysicsTree(world, a, treeBox(0), () => {
                throw error;
            }),
        ).toThrow(error);
        expect(() =>
            queryPhysicsTree(world, a, treeBox(0), () => {
                destroyPhysicsTree(world, a);
                return true;
            }),
        ).toThrow("query callback");
        let count = 0;
        queryPhysicsTree(world, a, treeBox(0), () => {
            queryPhysicsTree(world, b, treeBox(0), () => {
                count++;
                return true;
            });
            return true;
        });
        expect(count).toBe(1);
        validatePhysicsTree(world, a);
        destroyPhysicsTree(world, a);
        destroyPhysicsTree(world, b);
    });
});
test("tree callbacks retain borrowed ray records across memory growth", async () => {
    await subject((world) => {
        const tree = createPhysicsTree(world);
        for (let i = 0; i < 3; i++) createPhysicsTreeProxy(world, tree, treeBox(0));
        let count = 0;
        let borrowed: PhysicsTreeRayHit | undefined;
        const result = rayCastPhysicsTree(
            world,
            tree,
            {
                origin: { x: -2, y: 0, z: 0 },
                translation: { x: 4, y: 0, z: 0 },
                maxFraction: 1,
            },
            (hit) => {
                expect(hit.maxFraction).toBe(1);
                if (count++ === 0) {
                    borrowed = hit;
                    kernel(world).memory.grow(1);
                } else expect(hit).toBe(borrowed!);
                return 1;
            },
        );
        expect(result.leafVisits).toBe(3);
        destroyPhysicsTree(world, tree);
    });
});
test("tree images own their bytes and reject malformed topology and versions", async () => {
    await subject((world) => {
        const a = createPhysicsTree(world);
        createPhysicsTreeProxy(world, a, treeBox(0));
        const image = savePhysicsTree(world, a);
        destroyPhysicsTree(world, a);
        const b = loadPhysicsTree(world, image);
        expect(getPhysicsTreeStats(world, b).proxyCount).toBe(1);
        destroyPhysicsTree(world, b);
        const corrupt = image.slice();
        new DataView(corrupt.buffer).setInt32(4, 9999, true);
        expect(() => loadPhysicsTree(world, corrupt)).toThrow("invalid");
        corrupt[0] = 0;
        expect(() => loadPhysicsTree(world, corrupt)).toThrow("invalid");
        expect(() => loadPhysicsTree(world, image.subarray(1))).toThrow("invalid");
        expect(() => loadPhysicsTree(world, image, -1)).toThrow("invalid");
        const empty = createPhysicsTree(world),
            restored = loadPhysicsTree(world, savePhysicsTree(world, empty));
        validatePhysicsTree(world, restored);
        destroyPhysicsTree(world, empty);
        destroyPhysicsTree(world, restored);
    });
});
