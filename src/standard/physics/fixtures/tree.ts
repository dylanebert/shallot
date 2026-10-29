import type { World } from "../../../engine";
import {
    createPhysicsTree,
    createPhysicsTreeProxy,
    destroyPhysicsTree,
    getPhysicsTreeNode,
    getPhysicsTreeStats,
    loadPhysicsTree,
    type PhysicsTree,
    queryClosestPhysicsTree,
    queryPhysicsTree,
    rayCastPhysicsTree,
    rebuildPhysicsTree,
    savePhysicsTree,
    validatePhysicsTree,
} from "../index";

export function treeBox(i: number) {
    const x = (i * 7) % 19,
        y = i % 5,
        z = (i * 3) % 11,
        s = (i % 3) + 1;
    return { lowerBound: { x, y, z }, upperBound: { x: x + s, y: y + s, z: z + s } };
}
export function treeSweep(world: World): string {
    const rows: string[] = [];
    const bitBuffer = new ArrayBuffer(4),
        f = new Float32Array(bitBuffer),
        u = new Uint32Array(bitBuffer);
    const bits = (value: number) => {
        f[0] = value;
        return u[0];
    };
    const report = (tree: PhysicsTree) => {
        validatePhysicsTree(world, tree);
        const s = getPhysicsTreeStats(world, tree);
        rows.push(
            `tree ${s.root} ${s.nodeCount} ${s.nodeCapacity} ${s.proxyCount} ${s.height} ${bits(s.areaRatio)}`,
        );
        for (let i = 0; i < s.nodeCapacity; i++) {
            const n = getPhysicsTreeNode(world, tree, i);
            if (!n) continue;
            const lo = n.bounds.lowerBound,
                hi = n.bounds.upperBound;
            rows.push(
                `node ${i} ${n.parent} ${n.child1} ${n.child2} ${n.height} ${Number(n.leaf)} ${n.userData} ${bits(lo.x)} ${bits(lo.y)} ${bits(lo.z)} ${bits(hi.x)} ${bits(hi.y)} ${bits(hi.z)}`,
            );
        }
    };
    const queries = (tree: PhysicsTree) => {
        const tiny = queryClosestPhysicsTree(
            world,
            tree,
            { x: 0, y: 0, z: 0 },
            2 ** -149,
            (hit) => {
                rows.push(`tiny ${hit.proxyId} ${hit.userData} ${bits(hit.minDistanceSquared)}`);
                return hit.minDistanceSquared;
            },
        );
        rows.push(`tiny-stats ${tiny.nodeVisits} ${tiny.leafVisits} ${bits(tiny.distanceSquared)}`);
        // Closest callbacks use caller-owned geometry, not borrowed tree inspection during traversal.
        const boxes = new Map<number, ReturnType<typeof treeBox>>();
        const capacity = getPhysicsTreeStats(world, tree).nodeCapacity;
        for (let i = 0; i < capacity; i++) {
            const n = getPhysicsTreeNode(world, tree, i);
            if (n?.leaf) boxes.set(i, structuredClone(n.bounds));
        }
        for (let mode = 0; mode < 4; mode++) {
            const filter = {
                maskBits:
                    mode === 2 ? 0x100000000n : mode === 3 ? 0x100000001n : 0xffffffffffffffffn,
                requireAllBits: mode === 3,
            };
            let s = queryPhysicsTree(
                world,
                tree,
                { lowerBound: { x: -2, y: -2, z: -2 }, upperBound: { x: 12, y: 12, z: 12 } },
                (id, user) => {
                    rows.push(`overlap ${id} ${user}`);
                    return mode !== 1;
                },
                filter,
            );
            rows.push(`stats ${s.nodeVisits} ${s.leafVisits}`);
            s = rayCastPhysicsTree(
                world,
                tree,
                {
                    origin: { x: -5, y: 2, z: 4 },
                    translation: { x: 40, y: 0, z: 0 },
                    maxFraction: 1,
                },
                (hit) => {
                    rows.push(`ray ${hit.proxyId} ${hit.userData} ${bits(hit.maxFraction)}`);
                    return mode === 1 ? 0 : mode === 2 ? -1 : mode === 3 ? 0.5 : 1;
                },
                filter,
            );
            rows.push(`stats ${s.nodeVisits} ${s.leafVisits}`);
            const closest = queryClosestPhysicsTree(
                world,
                tree,
                { x: 24, y: 10, z: 3 },
                1000000,
                (hit) => {
                    rows.push(
                        `closest ${hit.proxyId} ${hit.userData} ${bits(hit.minDistanceSquared)}`,
                    );
                    const b = boxes.get(hit.proxyId)!;
                    const x = 24 - Math.max(b.lowerBound.x, Math.min(24, b.upperBound.x));
                    const y = 10 - Math.max(b.lowerBound.y, Math.min(10, b.upperBound.y));
                    const z = 3 - Math.max(b.lowerBound.z, Math.min(3, b.upperBound.z));
                    return Math.fround(
                        Math.fround(Math.fround(x * x) + Math.fround(y * y)) + Math.fround(z * z),
                    );
                },
                filter,
            );
            rows.push(
                `stats ${closest.nodeVisits} ${closest.leafVisits} ${bits(closest.distanceSquared)}`,
            );
        }
    };
    const tree = createPhysicsTree(world, 16);
    try {
        report(tree);
        queries(tree);
        for (let i = 0; i < 48; i++)
            createPhysicsTreeProxy(
                world,
                tree,
                treeBox(i),
                (1n << BigInt(i % 3)) | (i % 2 ? 0x100000000n : 0n),
                0x1234567800000000n + BigInt(i),
            );
        report(tree);
        queries(tree);
        rebuildPhysicsTree(world, tree, true);
        report(tree);
        queries(tree);
        rebuildPhysicsTree(world, tree, false);
        report(tree);
        queries(tree);
        const loaded = loadPhysicsTree(world, savePhysicsTree(world, tree), 2);
        try {
            report(loaded);
            queries(loaded);
        } finally {
            destroyPhysicsTree(world, loaded);
        }
    } finally {
        destroyPhysicsTree(world, tree);
    }
    return `${rows.join("\n")}\n`;
}
