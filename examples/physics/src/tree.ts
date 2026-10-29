import type { Plugin, World } from "@dylanebert/shallot";
import {
    createPhysicsTree,
    createPhysicsTreeProxy,
    destroyPhysicsTree,
    getPhysicsTreeNode,
    getPhysicsTreeStats,
    loadPhysicsTree,
    type PhysicsTree,
    type PhysicsTreeClosestHit,
    type PhysicsTreeRayHit,
    queryClosestPhysicsTree,
    queryPhysicsTree,
    rayCastPhysicsTree,
    rebuildPhysicsTree,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
    savePhysicsTree,
    validatePhysicsTree,
} from "@dylanebert/shallot/standard/physics";
import bounds01 from "./bounds01.json" with { type: "json" };
import bounds02 from "./bounds02.json" with { type: "json" };
import bounds03 from "./bounds03.json" with { type: "json" };
import type { Controls } from "./scenes";

const f = Math.fround;
type Point = { x: number; y: number; z: number };
type Bounds = { lowerBound: Point; upperBound: Point };
type Proxy = { bounds: Bounds; id: number; queryStamp: number; rayStamp: number };
type Node = {
    bounds: Bounds;
    parent: number;
    child1: number;
    child2: number;
    leaf: boolean;
    height: number;
    depth: number;
    user: number;
};
type Ray = { origin: Point; translation: Point; maxFraction: number };
const initial = {
    File: 0,
    "Ray Cast": 0,
    Overlap: 0,
    "Closet Point": 0,
    Test: 0,
    Level: -1,
    Kilometers: 1,
    "Load Scale": 1,
};
const point = (): Point => ({ x: 0, y: 0, z: 0 });
const SavedTree = { create: () => ({ image: null as Uint8Array | null }) };
export const TreeBenchmarkState = {
    create() {
        const state = {
            active: false,
            tree: null as PhysicsTree | null,
            values: { ...initial },
            nodes: [] as (Node | null)[],
            proxies: [] as Proxy[],
            rays: [] as Ray[],
            overlaps: [] as Bounds[],
            closestQueries: [] as { center: Point; radius: number }[],
            closestPoint: point(),
            haveClosest: false,
            stamp: 1,
            seed: 12345,
            leaves: 0,
            root: -1,
            height: 0,
            area: 0,
            buildTime: 0,
            rayTime: 0,
            overlapTime: 0,
            closestTime: 0,
        };
        return Object.assign(state, {
            overlap(_id: number, user: bigint) {
                state.proxies[Number(user)]!.queryStamp = state.stamp;
                return true;
            },
            ray(hit: PhysicsTreeRayHit) {
                state.proxies[Number(hit.userData)]!.rayStamp = state.stamp;
                return 1;
            },
            closest(hit: PhysicsTreeClosestHit) {
                const proxy = state.proxies[Number(hit.userData)]!;
                proxy.queryStamp = state.stamp;
                // The sample's callback uses Test, including inside Profile's loop over every query.
                const p = state.closestQueries[state.values.Test]!.center,
                    b = proxy.bounds;
                const q = {
                    x: Math.max(b.lowerBound.x, Math.min(p.x, b.upperBound.x)),
                    y: Math.max(b.lowerBound.y, Math.min(p.y, b.upperBound.y)),
                    z: Math.max(b.lowerBound.z, Math.min(p.z, b.upperBound.z)),
                };
                const x = f(p.x - q.x),
                    y = f(p.y - q.y),
                    z = f(p.z - q.z);
                const distance = f(f(f(x * x) + f(y * y)) + f(z * z));
                if (distance < hit.minDistanceSquared) {
                    Object.assign(state.closestPoint, q);
                    state.haveClosest = true;
                }
                return distance;
            },
        });
    },
};
type State = ReturnType<typeof TreeBenchmarkState.create>;
function release(world: World, s: State): void {
    if (s.tree) destroyPhysicsTree(world, s.tree);
    s.tree = null;
}
function reports(world: World, s: State): void {
    validatePhysicsTree(world, s.tree!);
    const stats = getPhysicsTreeStats(world, s.tree!);
    s.leaves = stats.proxyCount;
    s.root = stats.root;
    s.area = stats.areaRatio;
    s.nodes = Array.from({ length: stats.nodeCapacity }, (_, id) => {
        const n = getPhysicsTreeNode(world, s.tree!, id);
        return n
            ? {
                  bounds: structuredClone(n.bounds),
                  parent: n.parent,
                  child1: n.child1,
                  child2: n.child2,
                  leaf: n.leaf,
                  height: n.height,
                  depth: 0,
                  user: n.leaf ? Number(n.userData) : -1,
              }
            : null;
    });
    s.height = 0;
    if (!s.leaves) return;
    const queue = [stats.root];
    for (let i = 0; i < queue.length; i++) {
        const n = s.nodes[queue[i]!]!;
        n.depth = n.parent < 0 ? 0 : s.nodes[n.parent]!.depth + 1;
        s.height = Math.max(s.height, n.depth);
        if (!n.leaf) queue.push(n.child1, n.child2);
    }
}
function random(s: State, lo: number, hi: number): number {
    let x = s.seed;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    s.seed = x >>> 0;
    return f(f(f(hi - lo) * f((s.seed % 32768) / 32767)) + lo);
}
function generate(s: State): void {
    // RandomVec3 uses g_randomSeed, independent of the sample's srand(42).
    const root = s.nodes[s.root]!;
    const b = root.bounds,
        lo = b.lowerBound,
        hi = b.upperBound;
    const ext = {
        x: f(f(hi.x - lo.x) * 0.5),
        y: f(f(hi.y - lo.y) * 0.5),
        z: f(f(hi.z - lo.z) * 0.5),
    };
    const radius = f(f(f(ext.x + ext.y) + ext.z) / 3);
    const vector = () => ({
        x: random(s, lo.x, hi.x),
        y: random(s, lo.y, hi.y),
        z: random(s, lo.z, hi.z),
    });
    s.rays = [];
    s.overlaps = [];
    s.closestQueries = [];
    for (let i = 0; i < 1024; i++) {
        const origin = vector(),
            end = vector();
        s.rays.push({
            origin,
            translation: { x: f(end.x - origin.x), y: f(end.y - origin.y), z: f(end.z - origin.z) },
            maxFraction: 1,
        });
        const scale = random(s, f(0.01), f(0.2)),
            center = vector();
        const e = { x: f(scale * ext.x), y: f(scale * ext.y), z: f(scale * ext.z) };
        s.overlaps.push({
            lowerBound: { x: f(center.x - e.x), y: f(center.y - e.y), z: f(center.z - e.z) },
            upperBound: { x: f(center.x + e.x), y: f(center.y + e.y), z: f(center.z + e.z) },
        });
        s.closestQueries.push({ center, radius: f(scale * radius) });
    }
}
function createTree(world: World): void {
    const s = world.resource(TreeBenchmarkState);
    release(world, s);
    s.proxies = [];
    s.tree = createPhysicsTree(world, 512);
    const data = [bounds01, bounds02, bounds03][s.values.File]!,
        scale = s.values.File === 2 ? f(0.01) : 1;
    for (let i = 0; i < data.length; i += 6) {
        const p = (j: number): Point => ({
            x: f(scale * f(data[i + j]!)),
            y: f(scale * f(data[i + j + 1]!)),
            z: f(scale * f(data[i + j + 2]!)),
        });
        s.proxies.push({
            bounds: { lowerBound: p(0), upperBound: p(3) },
            id: -1,
            queryStamp: 0,
            rayStamp: 0,
        });
    }
    const start = performance.now();
    for (const [i, proxy] of s.proxies.entries())
        proxy.id = createPhysicsTreeProxy(world, s.tree, proxy.bounds, 1n, BigInt(i));
    s.buildTime = performance.now() - start;
    reports(world, s);
    s.values.Level = -1;
    generate(s);
}
function load(world: World): void {
    const s = world.resource(TreeBenchmarkState),
        saved = world.resource(SavedTree).image;
    release(world, s);
    s.proxies = [];
    s.tree = saved
        ? loadPhysicsTree(world, saved, s.values["Load Scale"])
        : createPhysicsTree(world, 512);
    reports(world, s);
    s.values.Level = -1;
    s.buildTime = 0;
    if (!s.leaves) return;
    for (const [id, n] of s.nodes.entries())
        if (n?.leaf) s.proxies[n.user] = { bounds: n.bounds, id, queryStamp: 0, rayStamp: 0 };
    generate(s);
}
function closestQuery(world: World, s: State, i: number): void {
    const q = s.closestQueries[i]!;
    Object.assign(s.closestPoint, q.center);
    s.haveClosest = false;
    queryClosestPhysicsTree(world, s.tree!, q.center, f(q.radius * q.radius), s.closest);
}
function profile(world: World): void {
    const s = world.resource(TreeBenchmarkState);
    let start = performance.now();
    for (let i = 0; i < 1024; i++) rayCastPhysicsTree(world, s.tree!, s.rays[i]!, s.ray);
    s.rayTime = performance.now() - start;
    start = performance.now();
    for (let i = 0; i < 1024; i++) queryPhysicsTree(world, s.tree!, s.overlaps[i]!, s.overlap);
    s.overlapTime = performance.now() - start;
    for (let i = 0; i < 1024; i++) closestQuery(world, s, i);
    // Profile measures closest from the overlap timer's start, as the sample does.
    s.closestTime = performance.now() - start;
}
export const TreeBenchmarkPlugin: Plugin = {
    name: "TreeBenchmark",
    dependencies: [StandardPhysicsPlugin],
    recovery(world) {
        const s = world.resource(TreeBenchmarkState);
        const snapshot = () => {
            const { tree, overlap: _overlap, ray: _ray, closest: _closest, ...data } = s;
            return {
                data: structuredClone(data),
                image: tree ? savePhysicsTree(world, tree) : null,
            };
        };
        return {
            snapshot,
            restore(value) {
                const image = value as ReturnType<typeof snapshot>;
                release(world, s);
                Object.assign(s, structuredClone(image.data));
                s.tree = image.image ? loadPhysicsTree(world, image.image) : null;
            },
        };
    },
    systems: [
        {
            name: "tree-benchmark-step",
            group: "fixed",
            after: [StepPhysicsSystem],
            update(world) {
                const s = world.resource(TreeBenchmarkState);
                if (!s.active || !s.leaves) return;
                ++s.stamp;
                if (s.values["Ray Cast"])
                    rayCastPhysicsTree(world, s.tree!, s.rays[s.values.Test]!, s.ray);
                if (s.values.Overlap)
                    queryPhysicsTree(world, s.tree!, s.overlaps[s.values.Test]!, s.overlap);
                if (s.values["Closet Point"]) closestQuery(world, s, s.values.Test);
            },
        },
    ],
};
function field(
    name: keyof typeof initial,
    options: Partial<Controls[string]> = {},
): Controls[string] {
    return {
        label: name,
        value: initial[name],
        enabled: (world) => world.resource(TreeBenchmarkState).leaves > 0,
        ...options,
        set(world, _bodies, value, values) {
            const s = world.resource(TreeBenchmarkState);
            s.values[name] = f(value);
            if (name === "File") {
                createTree(world);
                values.Level = -1;
            }
        },
    };
}
function button(name: string, action: (world: World) => void): Controls[string] {
    return {
        label: name,
        value: 0,
        button: true,
        enabled: (world) => world.resource(TreeBenchmarkState).leaves > 0,
        set: (world) => action(world),
    };
}
const controls: Controls = {
    File: field("File", {
        select: true,
        options: ["bounds01", "bounds02", "bounds03"].map((label, value) => ({ label, value })),
    }),
    "Top Down": button("Top Down", (world) => {
        const s = world.resource(TreeBenchmarkState),
            start = performance.now();
        rebuildPhysicsTree(world, s.tree!, true);
        s.buildTime = performance.now() - start;
        reports(world, s);
    }),
    "Ray Cast": field("Ray Cast"),
    Overlap: field("Overlap"),
    "Closet Point": field("Closet Point"),
    Profile: button("Profile", profile),
    Test: field("Test", { min: 0, max: 1023, step: 1, decimals: 0 }),
    Level: field("Level", {
        min: -1,
        max: 0,
        step: 1,
        decimals: 0,
        bounds: (_values, world) => [-1, world?.resource(TreeBenchmarkState).height ?? 0],
    }),
    Kilometers: field("Kilometers", { min: 0.5, max: 20, step: 0.1, decimals: 1 }),
    Save: button("Save", (world) => {
        world.resource(SavedTree).image = savePhysicsTree(
            world,
            world.resource(TreeBenchmarkState).tree!,
        );
    }),
    Load: button("Load", load),
    "Load Scale": field("Load Scale", { min: 0.01, max: 1, step: 0.001, decimals: 3 }),
};
for (const [name, label, read] of [
    ["leaves", "Leaves", (s: State) => s.leaves],
    ["height", "Height", (s: State) => s.height],
    ["area ratio", "Area ratio", (s: State) => s.area],
    ["build time", "Build time, ms", (s: State) => s.buildTime],
    ["ray time", "Ray total, ms", (s: State) => s.rayTime],
    ["overlap time", "Overlap total, ms", (s: State) => s.overlapTime],
    ["closest time", "Closest total, ms", (s: State) => s.closestTime],
    ["ray average", "Ray average, µs", (s: State) => (s.rayTime * 1000) / 1024],
    ["overlap average", "Overlap avg., µs", (s: State) => (s.overlapTime * 1000) / 1024],
    ["closest average", "Closest avg., µs", (s: State) => (s.closestTime * 1000) / 1024],
] as const)
    controls[name] = {
        label,
        value: 0,
        ...(name === "leaves" || name === "height" ? { decimals: 0 } : { precision: 6 }),
        read: (world) => read(world.resource(TreeBenchmarkState)),
        set() {},
    };
export function resetTreeBenchmark(world: World): void {
    const s = world.resource(TreeBenchmarkState);
    release(world, s);
    s.active = false;
    s.leaves = 0;
    s.root = -1;
    s.nodes = [];
    s.proxies = [];
    s.rays = [];
    s.overlaps = [];
    s.closestQueries = [];
}
export const treeScenes = {
    treeBenchmark: {
        construct(world: World): number[] {
            const s = world.resource(TreeBenchmarkState);
            const {
                overlap: _overlap,
                ray: _ray,
                closest: _closest,
                ...fresh
            } = TreeBenchmarkState.create();
            release(world, s);
            Object.assign(s, fresh, { active: true });
            createTree(world);
            return [];
        },
        controls,
        gridExtent: 200,
        gridSpacing: 20,
        gridHeight: f(-4.79),
        camera: {
            yaw: 35,
            pitch: 35,
            distance: 330,
            phoneDistance: 560,
            target: [0, 0, 0],
            phoneTarget: [0, 220, 0],
        },
    },
};

/** Host-independent benchmark state; excludes measured durations and camera-dependent culling. */
export function treeRecords(world: World): number[][] {
    const s = world.resource(TreeBenchmarkState),
        v = s.values;
    const rows = [
        [
            s.leaves,
            s.height,
            s.area,
            v.File,
            v.Test,
            v.Level,
            v.Kilometers,
            v["Load Scale"],
            s.stamp,
            Number(s.haveClosest),
            s.closestPoint.x,
            s.closestPoint.y,
            s.closestPoint.z,
            v["Ray Cast"],
            v.Overlap,
            v["Closet Point"],
        ],
    ];
    for (const [id, n] of s.nodes.entries())
        if (n)
            rows.push([
                id,
                n.parent,
                n.child1,
                n.child2,
                n.depth,
                n.height,
                Number(n.leaf),
                n.user,
                n.bounds.lowerBound.x,
                n.bounds.lowerBound.y,
                n.bounds.lowerBound.z,
                n.bounds.upperBound.x,
                n.bounds.upperBound.y,
                n.bounds.upperBound.z,
            ]);
    for (const [id, p] of s.proxies.entries()) rows.push([id, p.id, p.queryStamp, p.rayStamp]);
    if (s.leaves) {
        const r = s.rays[v.Test]!,
            b = s.overlaps[v.Test]!,
            q = s.closestQueries[v.Test]!;
        rows.push([
            r.origin.x,
            r.origin.y,
            r.origin.z,
            r.translation.x,
            r.translation.y,
            r.translation.z,
            b.lowerBound.x,
            b.lowerBound.y,
            b.lowerBound.z,
            b.upperBound.x,
            b.upperBound.y,
            b.upperBound.z,
            q.center.x,
            q.center.y,
            q.center.z,
            q.radius,
        ]);
    }
    return rows;
}
