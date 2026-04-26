import {
    pair,
    ChildOf,
    resource,
    traits,
    Shape,
    Transform,
    WorldTransform,
    Part,
    Mesh,
    Volume,
    mesh,
    surface,
    Camera,
    RenderPlugin,
    type Plugin,
    type State,
    type System,
    type MeshData,
} from "@dylanebert/shallot";

const SEED1 = "vec2(127.1, 311.7)";
const SEED2 = "vec2(269.5, 183.3)";
const CHUNK_SIZE = 40;
const VIEW_DISTANCE = 200;

let waterSurface = -1;

function registerWaterSurface(): void {
    waterSurface = surface({
        fragment: `
    let p = (*surface).worldPos.xz * 3.0;
    let t = scene.time;
    let t1 = vec2(t * 0.6, t * 0.4);
    let t2 = vec2(t * 0.5, t * 0.7);
    let eps = 0.1;

    let h  = (value2d(p + t1, ${SEED1}) + value2d(p - t2, ${SEED2})) * 0.5;
    let hx = (value2d(p + vec2(eps, 0.0) + t1, ${SEED1}) + value2d(p + vec2(eps, 0.0) - t2, ${SEED2})) * 0.5;
    let hz = (value2d(p + vec2(0.0, eps) + t1, ${SEED1}) + value2d(p + vec2(0.0, eps) - t2, ${SEED2})) * 0.5;

    let dx = (hx - h) / eps;
    let dz = (hz - h) / eps;

    let camDist = length((*surface).worldPos.xz - scene.cameraWorld[3].xz);
    let waveFade = 1.0 - smoothstep(${VIEW_DISTANCE * 0.2}, ${VIEW_DISTANCE * 0.9}, camDist);

    (*surface).worldNormal = normalize(vec3(-dx * 0.015 * waveFade, 1.0, -dz * 0.015 * waveFade));`,
    });
}

function createSubdividedPlane(subdivisions: number = 32): MeshData {
    const segments = subdivisions;
    const vertexCount = (segments + 1) * (segments + 1);
    const indexCount = segments * segments * 6;

    const vertices = new Float32Array(vertexCount * 8);
    const indices = new Uint16Array(indexCount);

    let vi = 0;
    for (let z = 0; z <= segments; z++) {
        for (let x = 0; x <= segments; x++) {
            const u = x / segments;
            const v = z / segments;
            vertices[vi++] = u - 0.5;
            vertices[vi++] = 0;
            vertices[vi++] = v - 0.5;
            vertices[vi++] = 0;
            vertices[vi++] = 1;
            vertices[vi++] = 0;
            vertices[vi++] = u;
            vertices[vi++] = v;
        }
    }

    let ii = 0;
    for (let z = 0; z < segments; z++) {
        for (let x = 0; x < segments; x++) {
            const topLeft = z * (segments + 1) + x;
            const topRight = topLeft + 1;
            const bottomLeft = (z + 1) * (segments + 1) + x;
            const bottomRight = bottomLeft + 1;

            indices[ii++] = topLeft;
            indices[ii++] = bottomLeft;
            indices[ii++] = topRight;
            indices[ii++] = topRight;
            indices[ii++] = bottomLeft;
            indices[ii++] = bottomRight;
        }
    }

    return { vertices, indices, vertexCount, indexCount };
}

let waterMesh = -1;

function registerWaterMesh(): void {
    waterMesh = mesh(createSubdividedPlane(64));
}

export const Water = {
    color: [] as number[],
    opacity: [] as number[],
    roughness: [] as number[],
    reflectivity: [] as number[],
    level: [] as number[],
    minX: [] as number[],
    maxX: [] as number[],
    minZ: [] as number[],
    maxZ: [] as number[],
};
traits(Water, {
    defaults: () => ({
        color: 0x3090a8,
        opacity: 0.3,
        roughness: 0.0,
        reflectivity: 0.5,
        level: 0,
        minX: 0,
        maxX: 0,
        minZ: 0,
        maxZ: 0,
    }),
});

function worldToChunk(pos: number, size: number): number {
    return Math.floor(pos / size);
}

function chunkKey(cx: number, cz: number): string {
    return `${cx},${cz}`;
}

const WaterChunk = { cx: [] as number[], cz: [] as number[] };

interface WaterState {
    surface: number;
    mesh: number;
}

const WaterResource = resource<WaterState>("water");

const _desired = new Set<string>();

function findActiveCamera(state: State): number {
    for (const eid of state.query([Camera, WorldTransform])) {
        if (Camera.active[eid]) return eid;
    }
    return -1;
}

function syncChunk(
    waterEid: number,
    chunkEid: number,
    res: WaterState,
    sizeX: number,
    sizeZ: number,
): void {
    Part.shape[chunkEid] = Shape.Mesh;
    Mesh.geometry[chunkEid] = res.mesh;
    Part.surface[chunkEid] = res.surface;
    Part.color[chunkEid] = Water.color[waterEid];
    Part.opacity[chunkEid] = Water.opacity[waterEid];
    Part.roughness[chunkEid] = Water.roughness[waterEid];
    Part.reflectivity[chunkEid] = Water.reflectivity[waterEid];
    Part.sizeX[chunkEid] = sizeX;
    Part.sizeY[chunkEid] = 1;
    Part.sizeZ[chunkEid] = sizeZ;
    Part.volume[chunkEid] = Volume.HalfSpace;
}

function chunkInBounds(
    gx: number,
    gz: number,
    minX: number,
    maxX: number,
    minZ: number,
    maxZ: number,
): boolean {
    if (minX !== maxX) {
        const chunkMaxX = (gx + 1) * CHUNK_SIZE;
        const chunkMinX = gx * CHUNK_SIZE;
        if (chunkMaxX <= minX || chunkMinX >= maxX) return false;
    }
    if (minZ !== maxZ) {
        const chunkMaxZ = (gz + 1) * CHUNK_SIZE;
        const chunkMinZ = gz * CHUNK_SIZE;
        if (chunkMaxZ <= minZ || chunkMinZ >= maxZ) return false;
    }
    return true;
}

const _clamp = { posX: 0, posZ: 0, sizeX: 0, sizeZ: 0 };

function clampedChunk(
    gx: number,
    gz: number,
    minX: number,
    maxX: number,
    minZ: number,
    maxZ: number,
    bounded: boolean,
): typeof _clamp {
    let cMinX = gx * CHUNK_SIZE;
    let cMaxX = cMinX + CHUNK_SIZE;
    let cMinZ = gz * CHUNK_SIZE;
    let cMaxZ = cMinZ + CHUNK_SIZE;

    if (bounded) {
        if (minX !== maxX) {
            cMinX = Math.max(cMinX, minX);
            cMaxX = Math.min(cMaxX, maxX);
        }
        if (minZ !== maxZ) {
            cMinZ = Math.max(cMinZ, minZ);
            cMaxZ = Math.min(cMaxZ, maxZ);
        }
    }

    _clamp.posX = (cMinX + cMaxX) * 0.5;
    _clamp.posZ = (cMinZ + cMaxZ) * 0.5;
    _clamp.sizeX = cMaxX - cMinX;
    _clamp.sizeZ = cMaxZ - cMinZ;
    return _clamp;
}

function reconcile(state: State, waterEid: number, camX: number, camZ: number): void {
    const cx = worldToChunk(camX, CHUNK_SIZE);
    const cz = worldToChunk(camZ, CHUNK_SIZE);
    const radius = VIEW_DISTANCE / CHUNK_SIZE + 1;
    const r = Math.ceil(radius);
    const r2 = radius * radius;

    const wMinX = Water.minX[waterEid];
    const wMaxX = Water.maxX[waterEid];
    const wMinZ = Water.minZ[waterEid];
    const wMaxZ = Water.maxZ[waterEid];
    const bounded = wMinX !== wMaxX || wMinZ !== wMaxZ;

    _desired.clear();
    const level = Water.level[waterEid];
    for (let dz = -r; dz <= r; dz++) {
        for (let dx = -r; dx <= r; dx++) {
            if (dx * dx + dz * dz > r2) continue;
            const gx = cx + dx;
            const gz = cz + dz;
            if (bounded && !chunkInBounds(gx, gz, wMinX, wMaxX, wMinZ, wMaxZ)) continue;
            _desired.add(chunkKey(gx, gz));
        }
    }

    for (const eid of state.query([WaterChunk, pair(ChildOf.relation, waterEid)])) {
        const key = chunkKey(WaterChunk.cx[eid], WaterChunk.cz[eid]);
        if (_desired.has(key)) {
            _desired.delete(key);
        } else {
            state.removeEntity(eid);
        }
    }

    for (const key of _desired) {
        const [gxStr, gzStr] = key.split(",");
        const gx = Number(gxStr);
        const gz = Number(gzStr);
        const c = clampedChunk(gx, gz, wMinX, wMaxX, wMinZ, wMaxZ, bounded);
        const eid = state.addEntity();
        state.addComponent(eid, WaterChunk);
        state.addComponent(eid, Transform);
        state.addComponent(eid, Part);
        state.addComponent(eid, Mesh);
        state.addRelation(eid, ChildOf, waterEid);
        WaterChunk.cx[eid] = gx;
        WaterChunk.cz[eid] = gz;
        Transform.posX[eid] = c.posX;
        Transform.posY[eid] = level;
        Transform.posZ[eid] = c.posZ;
        Part.sizeX[eid] = c.sizeX;
        Part.sizeZ[eid] = c.sizeZ;
    }
}

const WaterSystem: System = {
    group: "simulation",
    annotations: { mode: "always" },

    update(state: State) {
        const res = WaterResource.from(state);
        if (!res) return;
        const cameraEid = findActiveCamera(state);
        const isPlay = state.scheduler.mode !== "edit";

        for (const waterEid of state.query([Water])) {
            if (cameraEid >= 0 && isPlay) {
                const camX = WorldTransform.data[cameraEid * 16 + 12];
                const camZ = WorldTransform.data[cameraEid * 16 + 14];
                reconcile(state, waterEid, camX, camZ);
            }

            const wMinX = Water.minX[waterEid];
            const wMaxX = Water.maxX[waterEid];
            const wMinZ = Water.minZ[waterEid];
            const wMaxZ = Water.maxZ[waterEid];
            const bounded = wMinX !== wMaxX || wMinZ !== wMaxZ;
            const level = Water.level[waterEid];

            for (const chunkEid of state.query([WaterChunk, pair(ChildOf.relation, waterEid)])) {
                Transform.posY[chunkEid] = level;
                const gx = WaterChunk.cx[chunkEid];
                const gz = WaterChunk.cz[chunkEid];
                const c = clampedChunk(gx, gz, wMinX, wMaxX, wMinZ, wMaxZ, bounded);
                syncChunk(waterEid, chunkEid, res, c.sizeX, c.sizeZ);
            }
        }
    },
};

export const WaterPlugin: Plugin = {
    name: "Water",
    systems: [WaterSystem],
    components: { Water },
    dependencies: [RenderPlugin],
    initialize(state) {
        registerWaterSurface();
        registerWaterMesh();
        state.setResource(WaterResource, {
            surface: waterSurface,
            mesh: waterMesh,
        });
    },
};
