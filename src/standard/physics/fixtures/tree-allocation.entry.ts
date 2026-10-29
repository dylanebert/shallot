import { createApp } from "@dylanebert/shallot";
import {
    createPhysicsTree,
    createPhysicsTreeProxy,
    destroyPhysicsTree,
    getPhysicsTreeNode,
    getPhysicsTreeStats,
    queryClosestPhysicsTree,
    queryPhysicsTree,
    rayCastPhysicsTree,
    StandardPhysicsPlugin,
} from "@dylanebert/shallot/standard/physics";

export let controlSink: object | undefined;
export const control = () => {
    controlSink = { tree: 0 };
};
export let leaked: object | undefined;
export default async function create(input: string) {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const world = app.world,
        tree = createPhysicsTree(world),
        empty = createPhysicsTree(world);
    const box = { lowerBound: { x: -1, y: -1, z: -1 }, upperBound: { x: 1, y: 1, z: 1 } };
    for (let i = 0; i < 32; i++)
        createPhysicsTreeProxy(world, tree, box, 0x100000001n, 0x1234567800000000n + BigInt(i));
    const ray = {
        origin: { x: -4, y: 0, z: 0 },
        translation: { x: 8, y: 0, z: 0 },
        maxFraction: 1,
    };
    const point = { x: 4, y: 0, z: 0 };
    const filter = { maskBits: 0x100000001n, requireAllBits: true };
    const miss = { maskBits: 2n };
    const overlap = () => true,
        stopOverlap = () => false;
    const cast = () => 1,
        stopCast = () => 0,
        ignoreCast = () => -1;
    const closest = () => 9;
    return {
        step() {
            const result = queryPhysicsTree(world, tree, box, overlap, filter);
            if (result.leafVisits !== 32) throw new Error("tree overlap lost leaves");
            queryPhysicsTree(world, tree, box, stopOverlap);
            queryPhysicsTree(world, tree, box, overlap, miss);
            queryPhysicsTree(world, empty, box, overlap);
            if (rayCastPhysicsTree(world, tree, ray, cast, filter).leafVisits !== 32)
                throw new Error("tree ray lost leaves");
            rayCastPhysicsTree(world, tree, ray, stopCast);
            rayCastPhysicsTree(world, tree, ray, ignoreCast);
            rayCastPhysicsTree(world, tree, ray, cast, miss);
            rayCastPhysicsTree(world, empty, ray, cast);
            if (
                queryClosestPhysicsTree(world, tree, point, 100, closest, filter)
                    .distanceSquared !== 9
            )
                throw new Error("tree closest lost limit");
            queryClosestPhysicsTree(world, tree, point, 100, closest, miss);
            queryClosestPhysicsTree(world, empty, point, 100, closest);
            getPhysicsTreeStats(world, tree);
            getPhysicsTreeNode(world, tree, 0);
            if (input === "leak") leaked = { ...result };
        },
        dispose() {
            destroyPhysicsTree(world, tree);
            destroyPhysicsTree(world, empty);
            app.dispose();
            leaked = undefined;
        },
    };
}
