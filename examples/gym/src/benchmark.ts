import {
    Transform,
    Camera,
    Part,
    Shape,
    Mesh,
    mesh,
    surface,
    RenderPlugin,
    hull,
} from "@dylanebert/shallot";
import type { Plugin, System, State } from "@dylanebert/shallot";
import {
    Lorenz,
    Benchmark,
    BenchConfig,
    BenchmarkState,
    CameraMode,
    Layout,
    createCone,
    coneMeshId,
    setConeMeshId,
} from "./config";
import { ConfigSystem } from "./effects";

const Sigma = 10;
const Rho = 28;
const Beta = 8 / 3;
const Scale = 2;
const CenterZ = Rho - 1;

const LorenzSystem: System = {
    group: "fixed",
    update(state) {
        const dt = state.time.fixedDeltaTime;
        for (const eid of state.query([Lorenz, Transform])) {
            const x = Lorenz.x[eid];
            const y = Lorenz.y[eid];
            const z = Lorenz.z[eid];

            const dx = Sigma * (y - x);
            const dy = x * (Rho - z) - y;
            const dz = x * y - Beta * z;

            Lorenz.x[eid] = x + dx * dt * 0.25;
            Lorenz.y[eid] = y + dy * dt * 0.25;
            Lorenz.z[eid] = z + dz * dt * 0.25;

            Transform.posX[eid] = Lorenz.x[eid] * Scale;
            Transform.posY[eid] = (Lorenz.z[eid] - CenterZ) * Scale;
            Transform.posZ[eid] = Lorenz.y[eid] * Scale;
        }
    },
};

const PART_COLORS = [0xd49560, 0x8b6040, 0x4078a0, 0xa49070];

function spawnPart(state: State, i: number): number {
    const eid = state.addEntity();
    state.addComponent(eid, Part);
    state.addComponent(eid, Transform);
    const variant = i % 4;
    if (variant === 3) {
        Part.shape[eid] = Shape.Mesh;
        state.addComponent(eid, Mesh);
        Mesh.geometry[eid] = coneMeshId();
    } else if (variant === 2) {
        Part.shape[eid] = Shape.Capsule;
    } else {
        Part.shape[eid] = variant === 0 ? Shape.Box : Shape.Sphere;
    }
    Part.color[eid] = PART_COLORS[variant];
    Part.roughness[eid] = Math.random();
    Part.reflectivity[eid] = Math.random() < 0.3 ? Math.random() : 0;
    return eid;
}

function spawnLorenz(state: State, i: number): number {
    const x = (Math.random() - 0.5) * 30;
    const y = (Math.random() - 0.5) * 30;
    const z = (Math.random() - 0.5) * 30 + CenterZ;

    const eid = spawnPart(state, i);
    state.addComponent(eid, Lorenz);
    Lorenz.x[eid] = x;
    Lorenz.y[eid] = y;
    Lorenz.z[eid] = z;
    Transform.posX[eid] = x * Scale;
    Transform.posY[eid] = (z - CenterZ) * Scale;
    Transform.posZ[eid] = y * Scale;
    return eid;
}

function spawnGrid(state: State, i: number, dim: number, spacing: number): number {
    const eid = spawnPart(state, i);
    Transform.posX[eid] = ((i % dim) - dim / 2) * spacing;
    Transform.posY[eid] = ((Math.floor(i / dim) % dim) - dim / 2) * spacing;
    Transform.posZ[eid] = (Math.floor(i / (dim * dim)) - dim / 2) * spacing;
    const size = spacing * 0.9;
    Part.sizeX[eid] = size;
    Part.sizeY[eid] = size;
    Part.sizeZ[eid] = size;
    return eid;
}

const SpawnSystem: System = {
    group: "simulation",
    update(state) {
        const res = state.getResource(BenchmarkState);
        if (!res || res.externalSpawner) return;

        const benchEid = state.only([Benchmark]);
        const count = benchEid >= 0 ? Benchmark.count[benchEid] : 0;
        const cam = state.only([Camera, BenchConfig]);
        const layout = cam >= 0 ? BenchConfig.layout[cam] : Layout.Lorenz;
        const layoutChanged = layout !== res.prev.layout;

        if (layoutChanged) {
            for (const s of res.spawned) state.removeEntity(s.eid);
            res.spawned = [];
            res.prev.layout = layout;
        }

        const current = res.spawned.length;
        if (count === current && !layoutChanged) return;

        if (count > current) {
            const isGrid = layout === Layout.Grid;
            const dim = isGrid ? Math.ceil(Math.cbrt(count)) : 0;
            const spacing = isGrid ? 120 / dim : 0;
            for (let i = current; i < count; i++) {
                const eid = isGrid ? spawnGrid(state, i, dim, spacing) : spawnLorenz(state, i);
                res.spawned.push({ eid, baseX: Transform.posX[eid] });
            }
        } else if (count < current) {
            const removed = res.spawned.splice(count);
            for (const s of removed) state.removeEntity(s.eid);
        }
    },
};

const PanSystem: System = {
    group: "simulation",
    update(state) {
        const cam = state.only([Camera, BenchConfig]);
        if (cam < 0 || BenchConfig.camera[cam] !== CameraMode.Pan) return;
        const res = state.getResource(BenchmarkState);
        if (!res) return;
        const offset = 120 * Math.sin(state.time.elapsed * 0.5);
        if (BenchConfig.layout[cam] === Layout.Grid) {
            for (const s of res.spawned) Transform.posX[s.eid] = s.baseX + offset;
        } else {
            for (const eid of state.query([Lorenz, Transform])) {
                Transform.posX[eid] = Lorenz.x[eid] * Scale + offset;
            }
        }
        for (const sl of res.spawnedLights) {
            Transform.posX[sl.eid] = sl.basePos[0] + offset;
        }
    },
};

export const BenchmarkPlugin: Plugin = {
    name: "Benchmark",
    dependencies: [RenderPlugin],
    systems: [LorenzSystem, SpawnSystem, ConfigSystem, PanSystem],
    components: { Lorenz, Benchmark, BenchConfig },
    initialize(state) {
        const coneId = mesh(createCone(), "cone");
        hull(coneId);
        setConeMeshId(coneId);

        const vertexSurface = surface(
            {
                vertex: `
    let t = scene.time * 2.0;
    let n = pos + normal * 0.5;
    let wave = sin(n.x * 4.0 + t) * cos(n.z * 3.0 - t * 0.7) * 0.15;
    let ripple = sin(length(n.xz) * 8.0 - t * 1.5) * 0.08;
    pos += normal * (wave + ripple);`,
            },
            "bench-vertex",
        );

        const fragmentSurface = surface(
            {
                fragment: `
    let wp = surface.worldPos * 0.3;
    let pattern = sin(wp.x * 5.0) * sin(wp.y * 5.0) * sin(wp.z * 5.0);
    let stripe = step(0.0, sin(wp.x * 10.0 + wp.z * 10.0));
    let r = mix(0.2, 0.9, stripe);
    let g = mix(0.1, 0.6, clamp(pattern + 0.5, 0.0, 1.0));
    let b = mix(0.3, 0.8, 1.0 - stripe);
    (*surface).baseColor = vec3(r, g, b);`,
            },
            "bench-fragment",
        );

        const trivialSurface = surface(
            { fragment: "(*surface).baseColor = vec3(0.5);" },
            "bench-trivial",
        );

        state.setResource(BenchmarkState, {
            spawned: [],
            spawnedLights: [],
            prev: {
                pipeline: -1,
                effects: -1,
                flags: -1,
                layout: Layout.Lorenz,
            },
            vertexSurface,
            fragmentSurface,
            trivialSurface,
            externalSpawner: false,
        });
    },
};
