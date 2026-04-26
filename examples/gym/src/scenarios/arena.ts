import {
    Transform,
    Body,
    Part,
    Shape,
    Camera,
    Shadows,
    Orbit,
    Tonemap,
    FXAA,
    AmbientLight,
    DirectionalLight,
    Mesh,
} from "@dylanebert/shallot";
import type { System, State } from "@dylanebert/shallot";
import { Benchmark, BenchConfig, coneMeshId } from "../config";

export let stepCount = 0;

export const StepCounterSystem: System = {
    group: "fixed",
    update() {
        stepCount++;
    },
};

export const PILE_SIZE = 0.5;

export function pileSpread(count: number) {
    return Math.max(6, Math.cbrt(count) * PILE_SIZE * 2.5);
}

export function pileDrop(count: number) {
    return pileSpread(count) + Math.cbrt(count) * PILE_SIZE * 2;
}

export function pileRand(seed: number): [number, number] {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return [seed / 0x7fffffff, seed];
}

export type PileShape = 0 | 1 | 2 | 3;

export const PILE_SHAPES: { id: PileShape; label: string }[] = [
    { id: 0, label: "box" },
    { id: 1, label: "sphere" },
    { id: 2, label: "capsule" },
    { id: 3, label: "cone" },
];

const SHAPE_MAP: { shape: number; color: number }[] = [
    { shape: Shape.Box, color: 0xd49560 },
    { shape: Shape.Sphere, color: 0x8b6040 },
    { shape: Shape.Capsule, color: 0x4078a0 },
    { shape: Shape.Mesh, color: 0xa49070 },
];

export function spawnPileBody(
    state: State,
    i: number,
    spread: number,
    drop: number,
    seed: number,
    shapes: PileShape[] = [0],
): [number, number] {
    let v: number;
    [v, seed] = pileRand(seed);
    const x = (v - 0.5) * spread;
    [v, seed] = pileRand(seed);
    const y = PILE_SIZE + v * drop;
    [v, seed] = pileRand(seed);
    const z = (v - 0.5) * spread;

    const eid = state.addEntity();
    state.addComponent(eid, Transform);
    state.addComponent(eid, Part);
    state.addComponent(eid, Body);
    Transform.posX[eid] = x;
    Transform.posY[eid] = y;
    Transform.posZ[eid] = z;
    [v, seed] = pileRand(seed);
    Transform.rotX[eid] = v * 360;
    [v, seed] = pileRand(seed);
    Transform.rotY[eid] = v * 360;
    [v, seed] = pileRand(seed);
    Transform.rotZ[eid] = v * 360;
    const variant = SHAPE_MAP[shapes[i % shapes.length]];
    Part.shape[eid] = variant.shape;
    Part.color[eid] = variant.color;
    if (variant.shape === Shape.Mesh) {
        state.addComponent(eid, Mesh);
        Mesh.geometry[eid] = coneMeshId();
    }
    Part.sizeX[eid] = PILE_SIZE;
    Part.sizeY[eid] = PILE_SIZE;
    Part.sizeZ[eid] = PILE_SIZE;
    Body.mass[eid] = 1;
    Body.friction[eid] = 0.5;
    return [eid, seed];
}

export interface ArenaEntities {
    ground: number;
    walls: number[];
}

export function initArena(state: State, spread: number, height: number): ArenaEntities {
    const cam = state.addEntity();
    state.addComponent(cam, Transform);
    state.addComponent(cam, Camera);
    state.addComponent(cam, Tonemap);
    state.addComponent(cam, FXAA);
    state.addComponent(cam, Shadows);
    state.addComponent(cam, Orbit);
    Orbit.distance[cam] = spread * 2;
    Orbit.maxDistance[cam] = 100;
    Orbit.pitch[cam] = Math.PI / 8;
    state.addComponent(cam, BenchConfig);

    const ambient = state.addEntity();
    state.addComponent(ambient, AmbientLight);
    const dir = state.addEntity();
    state.addComponent(dir, DirectionalLight);

    const ground = makeArenaGround(state, spread);
    const walls = makeArenaWalls(state, spread, height);

    return { ground, walls };
}

function makeArenaGround(state: State, spread: number): number {
    const groundSize = spread + 4;
    const ground = state.addEntity();
    state.addComponent(ground, Transform);
    state.addComponent(ground, Part);
    state.addComponent(ground, Body);
    Transform.posY[ground] = -0.5;
    Part.shape[ground] = Shape.Box;
    Part.sizeX[ground] = groundSize;
    Part.sizeY[ground] = 1;
    Part.sizeZ[ground] = groundSize;
    Part.color[ground] = 0x252220;
    Body.mass[ground] = 0;
    Body.friction[ground] = 0.5;
    return ground;
}

function makeArenaWalls(state: State, spread: number, height: number): number[] {
    const wallThickness = Math.max(0.5, spread * 0.05);
    const wallHeight = height + 5;
    const half = spread / 2 + wallThickness;
    const specs = [
        {
            x: half,
            y: wallHeight / 2 - 0.5,
            z: 0,
            sx: wallThickness,
            sy: wallHeight,
            sz: spread + wallThickness * 2,
        },
        {
            x: -half,
            y: wallHeight / 2 - 0.5,
            z: 0,
            sx: wallThickness,
            sy: wallHeight,
            sz: spread + wallThickness * 2,
        },
        {
            x: 0,
            y: wallHeight / 2 - 0.5,
            z: half,
            sx: spread + wallThickness * 2,
            sy: wallHeight,
            sz: wallThickness,
        },
        {
            x: 0,
            y: wallHeight / 2 - 0.5,
            z: -half,
            sx: spread + wallThickness * 2,
            sy: wallHeight,
            sz: wallThickness,
        },
    ];
    const ids: number[] = [];
    for (const w of specs) {
        const eid = state.addEntity();
        state.addComponent(eid, Transform);
        state.addComponent(eid, Part);
        state.addComponent(eid, Body);
        Transform.posX[eid] = w.x;
        Transform.posY[eid] = w.y;
        Transform.posZ[eid] = w.z;
        Part.shape[eid] = Shape.Box;
        Part.color[eid] = 0x555555;
        Part.opacity[eid] = 0;
        Part.sizeX[eid] = w.sx;
        Part.sizeY[eid] = w.sy;
        Part.sizeZ[eid] = w.sz;
        Body.mass[eid] = 0;
        Body.friction[eid] = 0.3;
        ids.push(eid);
    }
    return ids;
}

export function resizeArena(arena: ArenaEntities, spread: number, height: number) {
    const groundSize = spread + 4;
    Part.sizeX[arena.ground] = groundSize;
    Part.sizeZ[arena.ground] = groundSize;

    const wallThickness = Math.max(0.5, spread * 0.05);
    const wallHeight = height + 5;
    const half = spread / 2 + wallThickness;
    const specs = [
        { x: half, z: 0, sx: wallThickness, sz: spread + wallThickness * 2 },
        { x: -half, z: 0, sx: wallThickness, sz: spread + wallThickness * 2 },
        { x: 0, z: half, sx: spread + wallThickness * 2, sz: wallThickness },
        { x: 0, z: -half, sx: spread + wallThickness * 2, sz: wallThickness },
    ];
    for (let i = 0; i < 4; i++) {
        const w = specs[i];
        const eid = arena.walls[i];
        Transform.posX[eid] = w.x;
        Transform.posY[eid] = wallHeight / 2 - 0.5;
        Transform.posZ[eid] = w.z;
        Part.sizeX[eid] = w.sx;
        Part.sizeY[eid] = wallHeight;
        Part.sizeZ[eid] = w.sz;
    }
}

export interface PileState {
    spawned: number[];
    seed: number;
    arena: ArenaEntities;
    currentSpread: number;
    shapes: PileShape[];
    count: number;
    ecs: State | null;
}

export let pileState: PileState | null = null;

export function initPileState(s: PileState) {
    pileState = s;
}

export function setPileShapes(shapes: PileShape[]) {
    const ps = pileState;
    if (!ps || !ps.ecs) return;
    ps.shapes = shapes;
    for (const eid of ps.spawned) ps.ecs.removeEntity(eid);
    ps.spawned = [];
    ps.seed = 54321;
    const spread = pileSpread(ps.count);
    const drop = pileDrop(ps.count);
    for (let i = 0; i < ps.count; i++) {
        const [eid, newSeed] = spawnPileBody(ps.ecs, i, spread, drop, ps.seed, shapes);
        ps.seed = newSeed;
        ps.spawned.push(eid);
    }
}

export const PileRampSystem: System = {
    group: "simulation",
    update(state) {
        const ps = pileState;
        if (!ps) return;

        const benchEid = state.only([Benchmark]);
        if (benchEid < 0) return;
        const target = Benchmark.count[benchEid];
        const current = ps.spawned.length;
        if (target === current) return;

        ps.count = target;
        if (target > current) {
            const spread = pileSpread(target);
            const drop = pileDrop(target);
            if (spread !== ps.currentSpread) {
                resizeArena(ps.arena, spread, drop);
                ps.currentSpread = spread;
            }
            for (let i = current; i < target; i++) {
                const [eid, newSeed] = spawnPileBody(state, i, spread, drop, ps.seed, ps.shapes);
                ps.seed = newSeed;
                ps.spawned.push(eid);
            }
        } else {
            const removed = ps.spawned.splice(target);
            for (const eid of removed) state.removeEntity(eid);
        }
    },
};
