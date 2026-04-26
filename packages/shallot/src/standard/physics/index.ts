import {
    resource,
    traits,
    onAdd,
    onRemove,
    Time,
    buf,
    capacity,
    type Plugin,
    type State,
    type System,
} from "../../engine";
import { createFieldProxy } from "../../engine/ecs/core";
import { Transform } from "../transforms";
import { transformWasm } from "../transforms/core";
import { Part, Mesh, Render, RenderPlugin } from "../render";
import { gbuf, binding, type GBuf, type Binding } from "../compute";
import { PartSizes, PartShapes } from "../render/core";
import { write } from "../../engine";
import {
    Compute,
    ComputePlugin,
    beginComputePass,
    type ComputeNode,
    type ExecutionContext,
} from "../compute";
import { interpolateWGSL } from "./interpolate.wgsl";
import { characterSweepWGSL, characterApplyWGSL } from "./character.wgsl";

import {
    type ProfileState,
    GpuProfile,
    createProfileState,
    allocSlot,
    resetProfile,
    resolveProfile,
    readProfile,
    drainProfile,
} from "../compute/core";
import {
    solverWGSL,
    coloringWGSL,
    maxConstraints,
    CONSTRAINT_BYTES,
    WARMSTART_BYTES,
    hashCapacity,
    JOINT_BYTES,
    MAX_COLORS,
    COLORING_ROUNDS,
    solverStateSize,
    HASH_REGION_OFFSET,
    colorDataSize,
    COUNTERS_SIZE,
    csrHeadsOffset,
    csrOffsetsOffset,
    SS_CONSTRAINT_COUNT,
    SS_USED_COLORS,
    SS_CONTACT_COUNT,
    SS_CONTACT_OVERFLOW,
    MAX_CONTACTS,
    CONTACT_STRIDE,
    CONTACTS_BYTES,
    NUM_PAIR_TYPES,
    BODY_COL_COUNT,
} from "./solver.wgsl";
import { bvhTraversalWGSL } from "./broadphase.wgsl";
import { narrowphaseWGSL, PAIR_TYPE_NAMES } from "./narrowphase.wgsl";
import {
    packWGSL,
    rebuildWGSL,
    syncTransformsWGSL,
    compactWGSL,
    emitContactsWGSL,
    prepareIndirectWGSL,
    BODY_BYTES,
    COMPACT_STRIDE,
    COMPACT_BYTES,
} from "./utility.wgsl";
import { createPrefixSum, rebuildPrefixSum, dispatchPrefixSum, disposePrefixSum } from "../radix";
import { createPhysicsLBVH, dispatchPhysicsLBVH, disposePhysicsLBVH } from "./lbvh";
import {
    resolveChanges,
    packJointData,
    checkOverflows,
    type JointInput,
    type SpringJointInput,
} from "./body";
import { packHullsForGPU, hullRegistry } from "./hull";
import { Shape } from "../../engine";

export {
    raycast,
    raySphere,
    rayCapsule,
    rayOBB,
    rayTriangle,
    rayMesh,
    generateRay,
    screenToRay,
} from "./raycast";
export type { Hit } from "./raycast";
export { hull, hullRegistry } from "./hull";
export type { ConvexHull, ConvexHullFace } from "./hull";
export { satHullHull, satHullBox, satHullSphere, satHullCapsule, boxToHull } from "./sat";
export type { SatContact, SatBody } from "./sat";

const BodyData = buf(Float32Array, 4, 0);

export const Body = {
    mass: createFieldProxy(BodyData, 4, 0),
    friction: createFieldProxy(BodyData, 4, 1),
    gravity: createFieldProxy(BodyData, 4, 2),
    group: createFieldProxy(BodyData, 4, 3),
};

traits(Body, {
    requires: [Part],
    defaults: () => ({ mass: 1, friction: 0.5, gravity: 1, group: 0 }),
});

const ForceData = buf(Float32Array, 6, 0);

export const Force = {
    forceX: createFieldProxy(ForceData, 6, 0),
    forceY: createFieldProxy(ForceData, 6, 1),
    forceZ: createFieldProxy(ForceData, 6, 2),
    torqueX: createFieldProxy(ForceData, 6, 3),
    torqueY: createFieldProxy(ForceData, 6, 4),
    torqueZ: createFieldProxy(ForceData, 6, 5),
};

traits(Force, {
    requires: [Body],
    defaults: () => ({ forceX: 0, forceY: 0, forceZ: 0, torqueX: 0, torqueY: 0, torqueZ: 0 }),
});

const ImpulseData = buf(Float32Array, 6, 0);

export const Impulse = {
    impulseX: createFieldProxy(ImpulseData, 6, 0),
    impulseY: createFieldProxy(ImpulseData, 6, 1),
    impulseZ: createFieldProxy(ImpulseData, 6, 2),
    angularImpulseX: createFieldProxy(ImpulseData, 6, 3),
    angularImpulseY: createFieldProxy(ImpulseData, 6, 4),
    angularImpulseZ: createFieldProxy(ImpulseData, 6, 5),
};

traits(Impulse, {
    requires: [Body],
    defaults: () => ({
        impulseX: 0,
        impulseY: 0,
        impulseZ: 0,
        angularImpulseX: 0,
        angularImpulseY: 0,
        angularImpulseZ: 0,
    }),
});

const VelocityData = buf(Float32Array, 6, 0);

export const Velocity = {
    linearX: createFieldProxy(VelocityData, 6, 0),
    linearY: createFieldProxy(VelocityData, 6, 1),
    linearZ: createFieldProxy(VelocityData, 6, 2),
    angularX: createFieldProxy(VelocityData, 6, 3),
    angularY: createFieldProxy(VelocityData, 6, 4),
    angularZ: createFieldProxy(VelocityData, 6, 5),
};

traits(Velocity, {
    requires: [Body],
    defaults: () => ({ linearX: 0, linearY: 0, linearZ: 0, angularX: 0, angularY: 0, angularZ: 0 }),
});

export const BallJoint = {
    bodyA: [] as number[],
    bodyB: [] as number[],
    anchorAX: [] as number[],
    anchorAY: [] as number[],
    anchorAZ: [] as number[],
    anchorBX: [] as number[],
    anchorBY: [] as number[],
    anchorBZ: [] as number[],
    stiffness: [] as number[],
    fracture: [] as number[],
};

traits(BallJoint, {
    defaults: () => ({
        anchorAX: 0,
        anchorAY: 0,
        anchorAZ: 0,
        anchorBX: 0,
        anchorBY: 0,
        anchorBZ: 0,
        stiffness: 0,
        fracture: 0,
    }),
});

export const SpringJoint = {
    bodyA: [] as number[],
    bodyB: [] as number[],
    anchorAX: [] as number[],
    anchorAY: [] as number[],
    anchorAZ: [] as number[],
    anchorBX: [] as number[],
    anchorBY: [] as number[],
    anchorBZ: [] as number[],
    restLength: [] as number[],
    stiffness: [] as number[],
    fracture: [] as number[],
};

traits(SpringJoint, {
    defaults: () => ({
        anchorAX: 0,
        anchorAY: 0,
        anchorAZ: 0,
        anchorBX: 0,
        anchorBY: 0,
        anchorBZ: 0,
        restLength: 0,
        stiffness: 0,
        fracture: 0,
    }),
});

const CharacterData = buf(Float32Array, 4, 0);

export const Character = {
    speed: createFieldProxy(CharacterData, 4, 0),
    maxSlope: createFieldProxy(CharacterData, 4, 1),
    jumpHeight: createFieldProxy(CharacterData, 4, 2),
    grounded: createFieldProxy(CharacterData, 4, 3),
    mass: [] as number[],
    gravity: [] as number[],
    coyoteTime: [] as number[],
    moveX: [] as number[],
    moveZ: [] as number[],
    jump: [] as number[],
};

traits(Character, {
    requires: [Body],
    defaults: () => ({
        speed: 6,
        maxSlope: 0.7,
        jumpHeight: 2.5,
        grounded: 0,
        mass: 70,
        gravity: 50,
        coyoteTime: 0.1,
        moveX: 0,
        moveZ: 0,
        jump: 0,
    }),
});

const PARAMS_SIZE = 64;

import type { PhysicsGPU, SolverParams } from "./gpu";

const DEFAULT_PARAMS: SolverParams = {
    dt: 1 / 60,
    gravity: -10,
    iterations: 4,
    alpha: 0.99,
    betaLin: 100000,
    betaAng: 100,
    gamma: 0.999,
};

/** marks a kinematic body as a moving platform — its displacement imparts velocity to contacts */
export const Move = {};
traits(Move, { requires: [Body] });

/** impulse is the positive normal impulse magnitude; bodyA/B are GPU body indices */
export interface Contact {
    bodyA: number;
    bodyB: number;
    posX: number;
    posY: number;
    posZ: number;
    normalX: number;
    normalY: number;
    normalZ: number;
    impulse: number;
}

/** double-buffered contact queue; contacts live 2 physics ticks before being dropped */
export interface Contacts {
    prevData: Uint32Array;
    prevCount: number;
    prevOverflow: number;
    prevTick: number;
    currentData: Uint32Array;
    currentCount: number;
    currentOverflow: number;
    currentTick: number;
}

export interface ContactReader {
    lastTick: number;
}

/** create a fresh cursor; pass to readContacts to iterate new contacts since last read */
export function contactReader(): ContactReader {
    return { lastTick: -1 };
}

const contactBits = new ArrayBuffer(4);
const contactU32 = new Uint32Array(contactBits);
const contactI32 = new Int32Array(contactBits);
const contactF32 = new Float32Array(contactBits);

const contactScratchObj: Contact = {
    bodyA: 0,
    bodyB: 0,
    posX: 0,
    posY: 0,
    posZ: 0,
    normalX: 0,
    normalY: 0,
    normalZ: 0,
    impulse: 0,
};

function decodeContact(data: Uint32Array, i: number, out: Contact): Contact {
    const base = i * 9;
    out.bodyA = data[base];
    contactU32[0] = data[base + 1];
    out.bodyB = contactI32[0];
    contactU32[0] = data[base + 2];
    out.posX = contactF32[0];
    contactU32[0] = data[base + 3];
    out.posY = contactF32[0];
    contactU32[0] = data[base + 4];
    out.posZ = contactF32[0];
    contactU32[0] = data[base + 5];
    out.normalX = contactF32[0];
    contactU32[0] = data[base + 6];
    out.normalY = contactF32[0];
    contactU32[0] = data[base + 7];
    out.normalZ = contactF32[0];
    contactU32[0] = data[base + 8];
    out.impulse = contactF32[0];
    return out;
}

/**
 * iterate contacts newer than reader's cursor; callback value is valid only during the call.
 * contacts older than 2 physics ticks are dropped — read every frame to avoid misses.
 */
export function readContacts(
    contacts: Contacts,
    reader: ContactReader,
    cb: (contact: Contact) => void,
): void {
    if (contacts.prevTick > reader.lastTick) {
        for (let i = 0; i < contacts.prevCount; i++)
            cb(decodeContact(contacts.prevData, i, contactScratchObj));
    }
    if (contacts.currentTick > reader.lastTick) {
        for (let i = 0; i < contacts.currentCount; i++)
            cb(decodeContact(contacts.currentData, i, contactScratchObj));
    }
    reader.lastTick = contacts.currentTick;
}

export const Physics = resource<PhysicsGPU>("physics");

export const Contacts = resource<Contacts>("contacts");

function pushChange(gpu: PhysicsGPU, value: number): void {
    if (gpu.pendingChangeCount >= gpu.pendingChanges.length) {
        const next = new Int32Array(gpu.pendingChanges.length * 2);
        next.set(gpu.pendingChanges);
        gpu.pendingChanges = next;
    }
    gpu.pendingChanges[gpu.pendingChangeCount++] = value;
}

async function createGPU(device: GPUDevice): Promise<PhysicsGPU> {
    const StorageDst = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const StorageDstSrc = StorageDst | GPUBufferUsage.COPY_SRC;
    const soaStorage = StorageDst;

    const bodyBuffer = gbuf(device, "physics-bodies", StorageDstSrc, (c) => c * BODY_BYTES);
    const bodyBufferPrev = gbuf(device, "physics-bodies-prev", StorageDst, (c) => c * BODY_BYTES);
    const bodyColsBuffer = gbuf(
        device,
        "physics-bodyCols",
        StorageDst,
        (c) => c * BODY_COL_COUNT * 16,
    );

    const constraintsBuffer = gbuf(
        device,
        "physics-constraints",
        StorageDstSrc,
        () => maxConstraints() * CONSTRAINT_BYTES,
    );
    const prevConstraintsBuffer = gbuf(
        device,
        "physics-prevConstraints",
        StorageDstSrc,
        () => maxConstraints() * CONSTRAINT_BYTES,
    );

    const warmstartBuffer = gbuf(
        device,
        "physics-warmstarts",
        StorageDst,
        () => hashCapacity() * WARMSTART_BYTES,
    );

    const solverStateBuffer = gbuf(
        device,
        "physics-solverState",
        StorageDstSrc,
        () => solverStateSize() + colorDataSize(),
    );
    const initData = new Uint32Array(hashCapacity()).fill(0xffffffff);
    device.queue.writeBuffer(solverStateBuffer.buffer, HASH_REGION_OFFSET, initData);

    const joints = {
        buffer: device.createBuffer({
            label: "physics-joints",
            size: 16 * JOINT_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        }),
        capacity: 16,
    };

    const csrCountsBuffer = gbuf(
        device,
        "physics-csrCounts",
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        (c) => (c + 1) * 4,
    );

    const indirectBuffer = device.createBuffer({
        label: "physics-indirect",
        size: (MAX_COLORS + 1 + NUM_PAIR_TYPES) * 12,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
    });

    const paramsBuffer = device.createBuffer({
        label: "physics-params",
        size: PARAMS_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const unpackTransformBuffer = gbuf(
        device,
        "physics-unpackTransform",
        StorageDstSrc,
        (c) => c * 8 * 4,
    );

    const sizesBuffer = gbuf(device, "physics-sizes", soaStorage, (c) => c * 16);
    const shapesBuffer = gbuf(device, "physics-shapes", soaStorage, (c) => Math.ceil(c / 4) * 4);
    const bodyPropsBuffer = gbuf(device, "physics-bodyProps", soaStorage, (c) => c * 16);
    const eidsBuffer = gbuf(device, "physics-eids", soaStorage, (c) => c * 4);
    const forceBuffer = gbuf(device, "physics-forces", StorageDst, (c) => c * 32);
    const packParamsBuffer = device.createBuffer({
        label: "physics-packParams",
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const rebuildParamsBuffer = device.createBuffer({
        label: "physics-rebuildParams",
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const pairBuffer = gbuf(
        device,
        "physics-pairs",
        GPUBufferUsage.STORAGE,
        () => NUM_PAIR_TYPES * maxConstraints() * 8,
    );

    const hullDataBuffer = {
        buffer: device.createBuffer({
            label: "physics-hullData",
            size: 32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        }),
    };
    const hullIdsBuffer = gbuf(device, "physics-hullIds", soaStorage, (c) => c * 4);

    const solverModule = device.createShaderModule({ code: solverWGSL });
    const coloringModule = device.createShaderModule({ code: coloringWGSL });
    const bvhTraversalModule = device.createShaderModule({ code: bvhTraversalWGSL });
    const narrowphaseModules = Array.from({ length: NUM_PAIR_TYPES }, (_, i) =>
        device.createShaderModule({ code: narrowphaseWGSL(i) }),
    );
    const packModule = device.createShaderModule({ code: packWGSL });
    const rebuildModule = device.createShaderModule({ code: rebuildWGSL });
    const prepareIndirectModule = device.createShaderModule({ code: prepareIndirectWGSL });
    const syncTransformsModule = device.createShaderModule({ code: syncTransformsWGSL });
    const compactModule = device.createShaderModule({ code: compactWGSL });
    const emitContactsModule = device.createShaderModule({ code: emitContactsWGSL });
    const characterSweepModule = device.createShaderModule({ code: characterSweepWGSL });
    const characterApplyModule = device.createShaderModule({ code: characterApplyWGSL });

    const vis = GPUShaderStage.COMPUTE;
    const baseEntries = [
        { binding: 0, visibility: vis, buffer: { type: "storage" as const } },
        { binding: 1, visibility: vis, buffer: { type: "uniform" as const } },
        { binding: 2, visibility: vis, buffer: { type: "storage" as const } },
        { binding: 3, visibility: vis, buffer: { type: "storage" as const } },
        { binding: 4, visibility: vis, buffer: { type: "storage" as const } },
        { binding: 5, visibility: vis, buffer: { type: "storage" as const } },
        { binding: 6, visibility: vis, buffer: { type: "read-only-storage" as const } },
        { binding: 7, visibility: vis, buffer: { type: "read-only-storage" as const } },
        { binding: 8, visibility: vis, buffer: { type: "read-only-storage" as const } },
    ];
    const bodyBindGroupLayout = device.createBindGroupLayout({ entries: baseEntries });
    const solverBindGroupLayout = device.createBindGroupLayout({
        entries: [
            ...baseEntries,
            { binding: 9, visibility: vis, buffer: { type: "read-only-storage" as const } },
            { binding: 10, visibility: vis, buffer: { type: "storage" as const } },
        ],
    });

    const solverLayout = device.createPipelineLayout({
        bindGroupLayouts: [solverBindGroupLayout],
    });

    const pairBindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: vis, buffer: { type: "storage" } },
            { binding: 1, visibility: vis, buffer: { type: "read-only-storage" as const } },
        ],
    });
    const bvhNarrowLayout = device.createPipelineLayout({
        bindGroupLayouts: [bodyBindGroupLayout, pairBindGroupLayout],
    });

    const rebuildBindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: vis, buffer: { type: "storage" } },
            { binding: 1, visibility: vis, buffer: { type: "storage" } },
            { binding: 2, visibility: vis, buffer: { type: "read-only-storage" } },
            { binding: 3, visibility: vis, buffer: { type: "uniform" } },
        ],
    });
    const rebuildLayout = device.createPipelineLayout({
        bindGroupLayouts: [rebuildBindGroupLayout],
    });

    const emitContactsBindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: vis, buffer: { type: "read-only-storage" as const } },
            { binding: 1, visibility: vis, buffer: { type: "read-only-storage" as const } },
            { binding: 2, visibility: vis, buffer: { type: "storage" as const } },
            { binding: 3, visibility: vis, buffer: { type: "storage" as const } },
        ],
    });
    const emitContactsLayout = device.createPipelineLayout({
        bindGroupLayouts: [emitContactsBindGroupLayout],
    });

    const characterBindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: vis, buffer: { type: "storage" as const } },
            { binding: 1, visibility: vis, buffer: { type: "read-only-storage" as const } },
            { binding: 2, visibility: vis, buffer: { type: "read-only-storage" as const } },
            { binding: 3, visibility: vis, buffer: { type: "read-only-storage" as const } },
            { binding: 4, visibility: vis, buffer: { type: "storage" as const } },
            { binding: 5, visibility: vis, buffer: { type: "read-only-storage" as const } },
            { binding: 6, visibility: vis, buffer: { type: "uniform" as const } },
            { binding: 7, visibility: vis, buffer: { type: "storage" as const } },
        ],
    });
    const characterLayout = device.createPipelineLayout({
        bindGroupLayouts: [characterBindGroupLayout],
    });

    const solverEntryPoints = [
        "warmstartBodies",
        "detectJoints",
        "initBodyCache",
        "cacheContactC",
        "solveDual",
        "advanceIteration",
        "computeVelocities",
        "writebackWarmstarts",
        "solvePrimal",
        "advanceColor",
        "resetColor",
        "syncBodyCols",
    ] as const;

    const coloringEntryPoints = [
        "clearColorBuffers",
        "countBodyConstraints",
        "scatterBodyConstraints",
        "buildAdjacencyList",
        "graphColor",
        "countColors",
        "prefixSumColors",
        "sortBodiesByColor",
    ] as const;

    const [
        solverPipelineEntries,
        coloringPipelineEntries,
        bvhTraversalPipeline,
        narrowphasePipelines,
        packPipeline,
        clearHashPipeline,
        rebuildPipeline,
        prepareIndirectPipeline,
        syncTransformsPipeline,
        compactPipeline,
        emitContactsPipeline,
        lbvh,
        csrPrefixSum,
        characterSweepPipeline,
        characterApplyPipeline,
    ] = await Promise.all([
        Promise.all(
            solverEntryPoints.map(async (ep) => {
                const pl = await device.createComputePipelineAsync({
                    label: ep,
                    layout: solverLayout,
                    compute: { module: solverModule, entryPoint: ep },
                });
                return [ep, pl] as const;
            }),
        ),
        Promise.all(
            coloringEntryPoints.map(async (ep) => {
                const pl = await device.createComputePipelineAsync({
                    label: ep,
                    layout: solverLayout,
                    compute: { module: coloringModule, entryPoint: ep },
                });
                return [ep, pl] as const;
            }),
        ),
        device.createComputePipelineAsync({
            label: "broadphase",
            layout: bvhNarrowLayout,
            compute: { module: bvhTraversalModule, entryPoint: "broadphase" },
        }),
        Promise.all(
            narrowphaseModules.map((module, i) =>
                device.createComputePipelineAsync({
                    label: `narrowphase-${PAIR_TYPE_NAMES[i]}`,
                    layout: bvhNarrowLayout,
                    compute: { module, entryPoint: "narrowphase" },
                }),
            ),
        ),
        device.createComputePipelineAsync({
            label: "packBodies",
            layout: "auto",
            compute: { module: packModule, entryPoint: "packBodies" },
        }),
        device.createComputePipelineAsync({
            label: "clearHash",
            layout: rebuildLayout,
            compute: { module: rebuildModule, entryPoint: "clearHash" },
        }),
        device.createComputePipelineAsync({
            label: "rebuildWarm",
            layout: rebuildLayout,
            compute: { module: rebuildModule, entryPoint: "rebuildWarm" },
        }),
        device.createComputePipelineAsync({
            label: "prepareIndirect",
            layout: "auto",
            compute: { module: prepareIndirectModule, entryPoint: "main" },
        }),
        device.createComputePipelineAsync({
            label: "syncTransforms",
            layout: "auto",
            compute: { module: syncTransformsModule, entryPoint: "syncTransforms" },
        }),
        device.createComputePipelineAsync({
            label: "readback",
            layout: "auto",
            compute: { module: compactModule, entryPoint: "readback" },
        }),
        device.createComputePipelineAsync({
            label: "emitContacts",
            layout: emitContactsLayout,
            compute: { module: emitContactsModule, entryPoint: "emitContacts" },
        }),
        createPhysicsLBVH(device, bodyBuffer, paramsBuffer),
        createPrefixSum(device, csrCountsBuffer.buffer, capacity() + 1),
        device.createComputePipelineAsync({
            label: "characterSweep",
            layout: characterLayout,
            compute: { module: characterSweepModule, entryPoint: "characterSweep" },
        }),
        device.createComputePipelineAsync({
            label: "characterApply",
            layout: characterLayout,
            compute: { module: characterApplyModule, entryPoint: "characterApply" },
        }),
    ]);

    const pipelines = Object.fromEntries([
        ...solverPipelineEntries,
        ...coloringPipelineEntries,
    ]) as Record<
        (typeof solverEntryPoints)[number] | (typeof coloringEntryPoints)[number],
        GPUComputePipeline
    >;

    const rebuildBindGroup = binding(device, rebuildBindGroupLayout, () => [
        {
            binding: 0,
            resource: {
                buffer: solverStateBuffer.buffer,
                offset: HASH_REGION_OFFSET,
                size: hashCapacity() * 4,
            },
        },
        { binding: 1, resource: { buffer: warmstartBuffer.buffer } },
        { binding: 2, resource: { buffer: prevConstraintsBuffer.buffer } },
        { binding: 3, resource: { buffer: rebuildParamsBuffer } },
    ]);

    const prepareIndirectBindGroup = binding(
        device,
        prepareIndirectPipeline.getBindGroupLayout(0),
        () => [
            { binding: 0, resource: { buffer: solverStateBuffer.buffer } },
            { binding: 1, resource: { buffer: indirectBuffer } },
        ],
    );

    const narrowBindGroupEntries = () => [
        { binding: 0, resource: { buffer: bodyBuffer.buffer } },
        { binding: 1, resource: { buffer: paramsBuffer } },
        { binding: 2, resource: { buffer: constraintsBuffer.buffer } },
        { binding: 3, resource: { buffer: warmstartBuffer.buffer } },
        { binding: 4, resource: { buffer: joints.buffer } },
        { binding: 5, resource: { buffer: solverStateBuffer.buffer } },
        { binding: 6, resource: { buffer: lbvh.lbvh.treeNodes } },
        { binding: 7, resource: { buffer: lbvh.lbvh.sortedIds } },
        { binding: 8, resource: { buffer: lbvh.bodyAABBs.buffer } },
    ];
    const narrowBindGroup = binding(device, bodyBindGroupLayout, narrowBindGroupEntries);
    const solverBindGroup = binding(device, solverBindGroupLayout, () => [
        ...narrowBindGroupEntries(),
        { binding: 9, resource: { buffer: forceBuffer.buffer } },
        { binding: 10, resource: { buffer: bodyColsBuffer.buffer } },
    ]);

    const pairBindGroup = binding(device, pairBindGroupLayout, () => [
        { binding: 0, resource: { buffer: pairBuffer.buffer } },
        { binding: 1, resource: { buffer: hullDataBuffer.buffer } },
    ]);

    const packBindGroup = binding(device, packPipeline.getBindGroupLayout(0), () => [
        { binding: 0, resource: { buffer: sizesBuffer.buffer } },
        { binding: 1, resource: { buffer: shapesBuffer.buffer } },
        { binding: 2, resource: { buffer: bodyPropsBuffer.buffer } },
        { binding: 3, resource: { buffer: eidsBuffer.buffer } },
        { binding: 4, resource: { buffer: bodyBuffer.buffer } },
        { binding: 5, resource: { buffer: packParamsBuffer } },
        { binding: 6, resource: { buffer: unpackTransformBuffer.buffer } },
        { binding: 7, resource: { buffer: hullIdsBuffer.buffer } },
    ]);

    const syncTransformsBindGroup = binding(
        device,
        syncTransformsPipeline.getBindGroupLayout(0),
        () => [
            { binding: 0, resource: { buffer: bodyBuffer.buffer } },
            { binding: 1, resource: { buffer: eidsBuffer.buffer } },
            { binding: 2, resource: { buffer: unpackTransformBuffer.buffer } },
            { binding: 3, resource: { buffer: packParamsBuffer } },
        ],
    );

    const compactBuffer = gbuf(
        device,
        "physics-compact",
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        () => capacity() * COMPACT_BYTES,
    );
    const compactParamsBuffer = device.createBuffer({
        label: "physics-compact-params",
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const compactBindGroup = binding(device, compactPipeline.getBindGroupLayout(0), () => [
        { binding: 0, resource: { buffer: bodyBuffer.buffer } },
        { binding: 1, resource: { buffer: compactBuffer.buffer } },
        { binding: 2, resource: { buffer: compactParamsBuffer } },
    ]);

    const contactsBuffer = device.createBuffer({
        label: "physics-contacts",
        size: CONTACTS_BYTES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const emitContactsBindGroup = binding(device, emitContactsBindGroupLayout, () => [
        { binding: 0, resource: { buffer: bodyBuffer.buffer } },
        { binding: 1, resource: { buffer: constraintsBuffer.buffer } },
        { binding: 2, resource: { buffer: solverStateBuffer.buffer } },
        { binding: 3, resource: { buffer: contactsBuffer } },
    ]);

    const characterBuffer = device.createBuffer({
        label: "physics-character-data",
        size: 64,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const characterIndicesBuffer = device.createBuffer({
        label: "physics-character-indices",
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const characterParamsBuffer = device.createBuffer({
        label: "physics-character-params",
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const characterGroundBuffer = device.createBuffer({
        label: "physics-character-ground",
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const characterSweepBindGroup = binding(device, characterBindGroupLayout, () => [
        { binding: 0, resource: { buffer: bodyBuffer.buffer } },
        { binding: 1, resource: { buffer: lbvh.lbvh.treeNodes } },
        { binding: 2, resource: { buffer: lbvh.lbvh.sortedIds } },
        { binding: 3, resource: { buffer: lbvh.bodyAABBs.buffer } },
        { binding: 4, resource: { buffer: characterBuffer } },
        { binding: 5, resource: { buffer: characterIndicesBuffer } },
        { binding: 6, resource: { buffer: characterParamsBuffer } },
        { binding: 7, resource: { buffer: characterGroundBuffer } },
    ]);
    const characterReadbackStaging = device.createBuffer({
        label: "physics-character-readback",
        size: 64,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const paramsData = new ArrayBuffer(PARAMS_SIZE);
    const profile: ProfileState | null = device.features.has("timestamp-query")
        ? createProfileState(device, 1024)
        : null;

    return {
        device,
        lbvh,
        warmstartPipeline: pipelines.warmstartBodies,
        clearHashPipeline,
        rebuildPipeline,
        bvhTraversalPipeline,
        narrowphasePipelines,
        pairBuffer,
        pairBindGroup,
        detectJointsPipeline: pipelines.detectJoints,
        initBodyCachePipeline: pipelines.initBodyCache,
        cacheContactCPipeline: pipelines.cacheContactC,
        dualPipeline: pipelines.solveDual,
        advancePipeline: pipelines.advanceIteration,
        velocityPipeline: pipelines.computeVelocities,
        writebackPipeline: pipelines.writebackWarmstarts,
        clearColorPipeline: pipelines.clearColorBuffers,
        countBodyConstraintsPipeline: pipelines.countBodyConstraints,
        scatterBodyConstraintsPipeline: pipelines.scatterBodyConstraints,
        buildAdjacencyPipeline: pipelines.buildAdjacencyList,
        graphColorPipeline: pipelines.graphColor,
        countColorsPipeline: pipelines.countColors,
        prefixSumColorsPipeline: pipelines.prefixSumColors,
        sortBodiesPipeline: pipelines.sortBodiesByColor,
        primalPipeline: pipelines.solvePrimal,
        advanceColorPipeline: pipelines.advanceColor,
        resetColorPipeline: pipelines.resetColor,
        syncBodyColsPipeline: pipelines.syncBodyCols,
        prepareIndirectPipeline,
        prepareIndirectBindGroup,
        packPipeline,
        syncTransformsPipeline,
        syncTransformsBindGroup,
        compactPipeline,
        compactBindGroup,
        compactBuffer,
        compactParamsBuffer,
        emitContactsPipeline,
        emitContactsBindGroup,
        contactsBuffer,
        bodyBuffer,
        bodyBufferPrev,
        bodyColsBuffer,
        constraintsBuffer,
        prevConstraintsBuffer,
        rebuildParamsBuffer,
        warmstartBuffer,
        solverStateBuffer,
        jointsBuffer: joints.buffer,
        jointSlot: joints,
        paramsBuffer,
        indirectBuffer,
        csrCountsBuffer,
        csrPrefixSum,
        unpackTransformBuffer,
        sizesBuffer,
        shapesBuffer,
        bodyPropsBuffer,
        eidsBuffer,
        packParamsBuffer,
        solverBindGroup,
        narrowBindGroup,
        rebuildBindGroup,
        packBindGroup,
        hullDataBuffer,
        hullIdsBuffer,
        forceBuffer,
        paramsData,
        paramsView: new DataView(paramsData),
        physicsActive: false,
        params: { ...DEFAULT_PARAMS },
        bodyEids: [],
        jointCount: 0,
        jointsNeedUpload: false,
        pendingChanges: new Int32Array(1024),
        pendingChangeCount: 0,
        cachedCapacity: capacity(),
        profile,
        debugReadbackData: new Uint32Array(COUNTERS_SIZE / 4),
        transformReadbackData: new Float32Array(0),
        contactScratch: new Uint32Array(MAX_CONTACTS * CONTACT_STRIDE),
        contactScratchCount: 0,
        contactScratchOverflow: 0,
        readbackStaging: device.createBuffer({
            label: "physics-readback-staging",
            size: COUNTERS_SIZE + capacity() * COMPACT_BYTES + CONTACTS_BYTES,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        }),
        readbackPending: false,
        readbackReady: false,
        readbackTick: 0,
        readbackGeneration: 0,
        readbackBodyCount: 0,
        lastSyncTick: -1,
        bodyGeneration: 0,
        characterSweepPipeline,
        characterApplyPipeline,
        characterSweepBindGroup,
        characterBuffer,
        characterIndicesBuffer,
        characterParamsBuffer,
        characterGroundBuffer,
        characterCount: 0,
        characters: [],
        characterVerticalVelocity: new Map(),
        characterCoyoteTimers: new Map(),
        characterReadbackStaging,
        characterReadbackPending: false,
    };
}

const rebuildScratch = new Uint32Array(4);

let eidsScratch = new Uint32Array(capacity());
const packParamsScratch = new Uint32Array(4);
let cachedEidsCapacity = capacity();

function ensureEidsScratch(): void {
    const c = capacity();
    if (c !== cachedEidsCapacity) {
        cachedEidsCapacity = c;
        eidsScratch = new Uint32Array(c);
    }
}

let hullIdsScratch = new Uint32Array(0);

function uploadHullData(gpu: PhysicsGPU): void {
    const packed = packHullsForGPU();
    const dataBytes = packed.data.byteLength;
    if (dataBytes === 0) return;
    if (dataBytes > gpu.hullDataBuffer.buffer.size) {
        gpu.hullDataBuffer.buffer.destroy();
        gpu.hullDataBuffer.buffer = gpu.device.createBuffer({
            label: "physics-hullData",
            size: dataBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        gpu.pairBindGroup.invalidate();
    }
    gpu.device.queue.writeBuffer(
        gpu.hullDataBuffer.buffer,
        0,
        packed.data as Uint32Array<ArrayBuffer>,
    );
}

function uploadHullIds(gpu: PhysicsGPU): void {
    const bodyCount = gpu.bodyEids.length;
    if (bodyCount === 0) return;
    if (hullIdsScratch.length < bodyCount) {
        hullIdsScratch = new Uint32Array(bodyCount);
    }
    for (let i = 0; i < bodyCount; i++) {
        const eid = gpu.bodyEids[i];
        if (Part.shape[eid] === Shape.Mesh) {
            const meshId = Mesh.geometry[eid];
            hullIdsScratch[i] = hullRegistry.getByName(String(meshId)) ?? 0;
        } else {
            hullIdsScratch[i] = 0;
        }
    }
    gpu.device.queue.writeBuffer(gpu.hullIdsBuffer.buffer, 0, hullIdsScratch, 0, bodyCount);
}

function syncJoints(state: State, gpu: PhysicsGPU): void {
    const eids = gpu.bodyEids;
    const eidToIndex = new Map<number, number>();
    for (let i = 0; i < eids.length; i++) eidToIndex.set(eids[i], i);

    const ballJoints: JointInput[] = [];
    for (const eid of state.query([BallJoint])) {
        ballJoints.push({
            anchorAX: BallJoint.anchorAX[eid] ?? 0,
            anchorAY: BallJoint.anchorAY[eid] ?? 0,
            anchorAZ: BallJoint.anchorAZ[eid] ?? 0,
            bodyA: BallJoint.bodyA[eid],
            anchorBX: BallJoint.anchorBX[eid] ?? 0,
            anchorBY: BallJoint.anchorBY[eid] ?? 0,
            anchorBZ: BallJoint.anchorBZ[eid] ?? 0,
            bodyB: BallJoint.bodyB[eid],
            stiffness: BallJoint.stiffness[eid] ?? 0,
            fracture: BallJoint.fracture[eid] ?? 0,
        });
    }

    const springJoints: SpringJointInput[] = [];
    for (const eid of state.query([SpringJoint])) {
        springJoints.push({
            anchorAX: SpringJoint.anchorAX[eid] ?? 0,
            anchorAY: SpringJoint.anchorAY[eid] ?? 0,
            anchorAZ: SpringJoint.anchorAZ[eid] ?? 0,
            bodyA: SpringJoint.bodyA[eid],
            anchorBX: SpringJoint.anchorBX[eid] ?? 0,
            anchorBY: SpringJoint.anchorBY[eid] ?? 0,
            anchorBZ: SpringJoint.anchorBZ[eid] ?? 0,
            bodyB: SpringJoint.bodyB[eid],
            stiffness: SpringJoint.stiffness[eid] ?? 0,
            fracture: SpringJoint.fracture[eid] ?? 0,
            restLength: SpringJoint.restLength[eid] ?? 0,
        });
    }

    const count = ballJoints.length + springJoints.length;
    gpu.jointCount = count;
    if (count === 0) return;

    const slot = gpu.jointSlot;
    if (count > slot.capacity) {
        slot.buffer.destroy();
        slot.capacity = count;
        slot.buffer = gpu.device.createBuffer({
            label: "physics-joints",
            size: count * JOINT_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        gpu.jointsBuffer = slot.buffer;
        gpu.solverBindGroup.invalidate();
        gpu.narrowBindGroup.invalidate();
    }

    const jointData = packJointData(ballJoints, springJoints, eidToIndex);
    gpu.device.queue.writeBuffer(slot.buffer, 0, jointData as Float32Array<ArrayBuffer>);
}

function processBodyChanges(gpu: PhysicsGPU, state: State, encoder: GPUCommandEncoder): void {
    if (gpu.pendingChangeCount === 0 && !gpu.jointsNeedUpload) return;

    const { removeOps, addEids } = resolveChanges(
        gpu.pendingChanges,
        gpu.pendingChangeCount,
        gpu.bodyEids,
        capacity(),
    );
    gpu.pendingChangeCount = 0;
    const addCount = addEids.length;
    const firstNewIndex = gpu.bodyEids.length - addCount;

    const bodyCount = gpu.bodyEids.length;
    const queue = gpu.device.queue;

    if (addCount > 0) {
        const entityCount = state.max + 1;
        ensureEidsScratch();
        write(queue, gpu.sizesBuffer.buffer, 0, PartSizes, entityCount);
        write(queue, gpu.shapesBuffer.buffer, 0, PartShapes, entityCount);
        write(queue, gpu.bodyPropsBuffer.buffer, 0, BodyData, entityCount);
        uploadHullData(gpu);
        uploadHullIds(gpu);
    }

    if (addCount > 0 || removeOps.length > 0) {
        ensureEidsScratch();
        for (let i = 0; i < bodyCount; i++) eidsScratch[i] = gpu.bodyEids[i];
        queue.writeBuffer(gpu.eidsBuffer.buffer, 0, eidsScratch, 0, bodyCount);
        gpu.bodyGeneration++;
    }

    packParamsScratch[0] = bodyCount;
    packParamsScratch[1] = capacity();
    packParamsScratch[2] = firstNewIndex;
    queue.writeBuffer(gpu.packParamsBuffer, 0, packParamsScratch);

    // Apply swap-remove copies BEFORE pack: pack writes new bodies at firstNewIndex onward,
    // which can collide with a removeOp's lastIdx (the slot pack just wrote into would be
    // copied over the swap target, scrambling slot identity).
    for (let i = 0; i < removeOps.length; i++) {
        const { removedIdx, lastIdx } = removeOps[i];
        encoder.copyBufferToBuffer(
            gpu.bodyBuffer.buffer,
            lastIdx * BODY_BYTES,
            gpu.bodyBuffer.buffer,
            removedIdx * BODY_BYTES,
            BODY_BYTES,
        );
        encoder.copyBufferToBuffer(
            gpu.bodyBufferPrev.buffer,
            lastIdx * BODY_BYTES,
            gpu.bodyBufferPrev.buffer,
            removedIdx * BODY_BYTES,
            BODY_BYTES,
        );
    }

    if (addCount > 0) {
        // Pack runs over all bodies: full init for new (i >= offset), shape refresh for
        // existing. Existing bodies' shape-derived fields (halfExtents, moments, radius)
        // would otherwise stay stale when Part.size* changes (e.g., pile ramp arena resize).
        const pass = beginComputePass(encoder);
        pass.setPipeline(gpu.packPipeline);
        pass.setBindGroup(0, gpu.packBindGroup.group);
        pass.dispatchWorkgroups(Math.ceil(bodyCount / 64));
        pass.end();
    }

    if (removeOps.length > 0 || gpu.jointsNeedUpload) {
        syncJoints(state, gpu);
        gpu.jointsNeedUpload = false;
    }
}

function uploadParams(gpu: PhysicsGPU, bodyCount: number): void {
    const view = gpu.paramsView;
    view.setFloat32(0, gpu.params.dt, true);
    view.setFloat32(4, gpu.params.gravity, true);
    view.setUint32(8, gpu.params.iterations, true);
    view.setFloat32(12, gpu.params.alpha, true);
    view.setFloat32(16, gpu.params.betaLin, true);
    view.setFloat32(20, gpu.params.gamma, true);
    view.setUint32(24, bodyCount, true);
    view.setUint32(28, gpu.jointCount, true);
    view.setUint32(32, capacity(), true);
    view.setUint32(36, maxConstraints() / capacity(), true);
    view.setUint32(40, hashCapacity() / capacity(), true);
    view.setFloat32(44, gpu.params.betaAng, true);
    view.setUint32(48, 0, true);
    view.setUint32(52, 0, true);
    view.setUint32(56, 0, true);
    view.setUint32(60, 0, true);
    gpu.device.queue.writeBuffer(gpu.paramsBuffer, 0, gpu.paramsData);
}

let forceScratch = new Float32Array(0);
const forceEidToIndex = new Map<number, number>();
const consumedEids: number[] = [];

function packForces(gpu: PhysicsGPU, state: State): void {
    const bodyCount = gpu.bodyEids.length;
    if (bodyCount === 0) return;

    const eids = gpu.bodyEids;
    const map = forceEidToIndex;
    map.clear();
    for (let i = 0; i < eids.length; i++) map.set(eids[i], i);

    const needed = bodyCount * 8;
    if (forceScratch.length < needed) forceScratch = new Float32Array(needed);
    else forceScratch.fill(0, 0, needed);

    let hasForce = false;
    for (const eid of state.query([Force])) {
        const idx = map.get(eid);
        if (idx === undefined) continue;
        hasForce = true;
        const off = idx * 8;
        forceScratch[off] = Force.forceX[eid];
        forceScratch[off + 1] = Force.forceY[eid];
        forceScratch[off + 2] = Force.forceZ[eid];
        forceScratch[off + 3] = Force.torqueX[eid];
        forceScratch[off + 4] = Force.torqueY[eid];
        forceScratch[off + 5] = Force.torqueZ[eid];
    }

    consumedEids.length = 0;
    const invDt = 1 / gpu.params.dt;
    for (const eid of state.query([Impulse])) {
        const idx = map.get(eid);
        if (idx === undefined) continue;
        hasForce = true;
        const off = idx * 8;
        forceScratch[off] += Impulse.impulseX[eid] * invDt;
        forceScratch[off + 1] += Impulse.impulseY[eid] * invDt;
        forceScratch[off + 2] += Impulse.impulseZ[eid] * invDt;
        forceScratch[off + 3] += Impulse.angularImpulseX[eid] * invDt;
        forceScratch[off + 4] += Impulse.angularImpulseY[eid] * invDt;
        forceScratch[off + 5] += Impulse.angularImpulseZ[eid] * invDt;
        consumedEids.push(eid);
    }
    for (const eid of consumedEids) state.removeComponent(eid, Impulse);

    consumedEids.length = 0;
    for (const eid of state.query([Velocity])) {
        const idx = map.get(eid);
        if (idx === undefined) continue;
        hasForce = true;
        const off = idx * 8;
        forceScratch[off] = Velocity.linearX[eid];
        forceScratch[off + 1] = Velocity.linearY[eid];
        forceScratch[off + 2] = Velocity.linearZ[eid];
        forceScratch[off + 3] = Velocity.angularX[eid];
        forceScratch[off + 4] = Velocity.angularY[eid];
        forceScratch[off + 5] = Velocity.angularZ[eid];
        forceScratch[off + 6] = 1.0;
        consumedEids.push(eid);
    }
    for (const eid of consumedEids) state.removeComponent(eid, Velocity);

    if (!hasForce) return;
    gpu.device.queue.writeBuffer(gpu.forceBuffer.buffer, 0, forceScratch, 0, needed);
}

let moveScratch = new Float32Array(0);

function syncTransforms(gpu: PhysicsGPU, state: State): void {
    if (gpu.bodyEids.length === 0) return;
    const S = capacity();
    const queue = gpu.device.queue;
    const buf = gpu.unpackTransformBuffer.buffer;
    type F32 = Float32Array<ArrayBuffer>;

    queue.writeBuffer(buf, 0, Transform.posX as F32, 0, S);
    queue.writeBuffer(buf, S * 4, Transform.posY as F32, 0, S);
    queue.writeBuffer(buf, 2 * S * 4, Transform.posZ as F32, 0, S);
    queue.writeBuffer(buf, 3 * S * 4, Transform.quatX as F32, 0, S);
    queue.writeBuffer(buf, 4 * S * 4, Transform.quatY as F32, 0, S);
    queue.writeBuffer(buf, 5 * S * 4, Transform.quatZ as F32, 0, S);
    queue.writeBuffer(buf, 6 * S * 4, Transform.quatW as F32, 0, S);

    if (moveScratch.length < S) moveScratch = new Float32Array(S);
    else moveScratch.fill(0);
    for (const eid of gpu.bodyEids) {
        if (state.hasComponent(eid, Move)) moveScratch[eid] = 1;
    }
    queue.writeBuffer(buf, 7 * S * 4, moveScratch, 0, S);
}

const CHAR_DATA_STRIDE = 8;
const CHAR_DATA_BYTES = CHAR_DATA_STRIDE * 4;
let charDataScratch = new Float32Array(0);
let charIndicesScratch = new Uint32Array(0);
const charParamsScratch = new Uint32Array(1);

const charEidToBody = new Map<number, number>();

function rebuildCharacterIndices(gpu: PhysicsGPU): void {
    const eids = gpu.characters;
    const count = eids.length;
    gpu.characterCount = count;
    if (count === 0) return;

    const bodyEids = gpu.bodyEids;
    charEidToBody.clear();
    for (let i = 0; i < bodyEids.length; i++) charEidToBody.set(bodyEids[i], i);

    if (charIndicesScratch.length < count) {
        charIndicesScratch = new Uint32Array(count);
    }
    if (charDataScratch.length < count * CHAR_DATA_STRIDE) {
        charDataScratch = new Float32Array(count * CHAR_DATA_STRIDE);
    }
    charDataScratch.fill(0, 0, count * CHAR_DATA_STRIDE);

    for (let i = 0; i < count; i++) {
        const eid = eids[i];
        charIndicesScratch[i] = charEidToBody.get(eid) ?? 0;
        charDataScratch[i * CHAR_DATA_STRIDE] = Character.maxSlope[eid];
        charDataScratch[i * CHAR_DATA_STRIDE + 5] = Character.mass[eid];
    }

    const indicesSize = count * 4;
    if (gpu.characterIndicesBuffer.size < indicesSize) {
        gpu.characterIndicesBuffer.destroy();
        gpu.characterIndicesBuffer = gpu.device.createBuffer({
            label: "physics-character-indices",
            size: indicesSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        gpu.characterSweepBindGroup.invalidate();
    }
    gpu.device.queue.writeBuffer(gpu.characterIndicesBuffer, 0, charIndicesScratch, 0, count);

    const dataSize = count * CHAR_DATA_BYTES;
    if (gpu.characterBuffer.size < dataSize) {
        gpu.characterBuffer.destroy();
        gpu.characterBuffer = gpu.device.createBuffer({
            label: "physics-character-data",
            size: dataSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        gpu.characterSweepBindGroup.invalidate();
    }

    const groundSize = count * 4;
    if (gpu.characterGroundBuffer.size < groundSize) {
        gpu.characterGroundBuffer.destroy();
        gpu.characterGroundBuffer = gpu.device.createBuffer({
            label: "physics-character-ground",
            size: groundSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        gpu.characterSweepBindGroup.invalidate();
        const initData = new Uint32Array(count);
        initData.fill(0xffffffff);
        gpu.device.queue.writeBuffer(gpu.characterGroundBuffer, 0, initData);
    }

    charParamsScratch[0] = count;
    gpu.device.queue.writeBuffer(gpu.characterParamsBuffer, 0, charParamsScratch);
}

function uploadCharacterData(gpu: PhysicsGPU): void {
    if (gpu.characterCount === 0) return;
    gpu.device.queue.writeBuffer(
        gpu.characterBuffer,
        0,
        charDataScratch,
        0,
        gpu.characterCount * CHAR_DATA_STRIDE,
    );
}

function updateCharacterInput(gpu: PhysicsGPU): void {
    const dt = gpu.params.dt;
    const vvel = gpu.characterVerticalVelocity;
    const coyote = gpu.characterCoyoteTimers;

    for (let ci = 0; ci < gpu.characters.length; ci++) {
        const eid = gpu.characters[ci];
        const grounded = Character.grounded[eid] > 0.5;
        const coyoteTime = Character.coyoteTime[eid] ?? 0.1;
        let coyoteTimer = coyote.get(eid) ?? 0;

        const g = Character.gravity[eid] ?? 50;
        const jumpVel = Math.sqrt(2 * g * Character.jumpHeight[eid]);
        const wantJump = Character.jump[eid] > 0.5;

        let vy = vvel.get(eid) ?? 0;
        if (grounded) {
            coyoteTimer = coyoteTime;
            vy = 0;
            if (wantJump) {
                vy = jumpVel;
                coyoteTimer = 0;
            }
        } else {
            coyoteTimer -= dt;
            vy -= g * dt;
            if (coyoteTimer > 0 && wantJump) {
                vy = jumpVel;
                coyoteTimer = 0;
            }
        }
        vvel.set(eid, vy);
        coyote.set(eid, coyoteTimer);

        const off = ci * CHAR_DATA_STRIDE;
        charDataScratch[off + 2] = Character.moveX[eid] * dt;
        charDataScratch[off + 3] = vy * dt;
        charDataScratch[off + 4] = Character.moveZ[eid] * dt;
    }
}

function requestCharacterReadback(gpu: PhysicsGPU): void {
    if (gpu.characterCount === 0 || gpu.characterReadbackPending) return;

    const byteSize = gpu.characterCount * CHAR_DATA_BYTES;
    if (gpu.characterReadbackStaging.size < byteSize) {
        gpu.characterReadbackStaging.destroy();
        gpu.characterReadbackStaging = gpu.device.createBuffer({
            label: "physics-character-readback",
            size: byteSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
    }

    gpu.characterReadbackPending = true;
    const gen = gpu.bodyGeneration;
    const count = gpu.characterCount;
    const eids = gpu.characters.slice();

    const enc = gpu.device.createCommandEncoder();
    enc.copyBufferToBuffer(gpu.characterBuffer, 0, gpu.characterReadbackStaging, 0, byteSize);
    gpu.device.queue.submit([enc.finish()]);

    gpu.characterReadbackStaging.mapAsync(GPUMapMode.READ, 0, byteSize).then(
        () => {
            if (gpu.bodyGeneration !== gen) {
                gpu.characterReadbackStaging.unmap();
                gpu.characterReadbackPending = false;
                return;
            }
            const mapped = gpu.characterReadbackStaging.getMappedRange(0, byteSize);
            const u32 = new Uint32Array(mapped);
            for (let i = 0; i < count && i < eids.length; i++) {
                Character.grounded[eids[i]] = u32[i * CHAR_DATA_STRIDE + 1] > 0 ? 1 : 0;
            }
            gpu.characterReadbackStaging.unmap();
            gpu.characterReadbackPending = false;
        },
        () => {
            gpu.characterReadbackPending = false;
        },
    );
}

function beginPass(
    encoder: GPUCommandEncoder,
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    workgroups: number,
    timestampWrites?: GPUComputePassTimestampWrites,
): void {
    const pass = beginComputePass(encoder, timestampWrites);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroups);
    pass.end();
}

function beginPassIndirect(
    encoder: GPUCommandEncoder,
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    indirectBuffer: GPUBuffer,
    indirectOffset: number,
    timestampWrites?: GPUComputePassTimestampWrites,
): void {
    const pass = beginComputePass(encoder, timestampWrites);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroupsIndirect(indirectBuffer, indirectOffset);
    pass.end();
}

function dispatch(gpu: PhysicsGPU, bodyCount: number, encoder: GPUCommandEncoder): void {
    const bodyWG = Math.ceil(bodyCount / 64);
    const p = gpu.profile;
    if (p) {
        drainProfile(p);
        resetProfile(p);
    }

    const hashWG = Math.ceil(hashCapacity() / 64);
    const rebuildGroup = gpu.rebuildBindGroup.group;

    const ts = (name: string) => (p ? allocSlot(p, name) : undefined);

    rebuildScratch[0] = 0;
    rebuildScratch[1] = hashCapacity();
    rebuildScratch[2] = 0;
    rebuildScratch[3] = 0;
    gpu.device.queue.writeBuffer(gpu.rebuildParamsBuffer, 0, rebuildScratch);

    const prevConstraintCount = gpu.debugReadbackData[SS_CONSTRAINT_COUNT];
    const prevCopySize =
        prevConstraintCount > 0
            ? prevConstraintCount * CONSTRAINT_BYTES
            : maxConstraints() * CONSTRAINT_BYTES;
    encoder.copyBufferToBuffer(
        gpu.solverStateBuffer.buffer,
        SS_CONSTRAINT_COUNT * 4,
        gpu.rebuildParamsBuffer,
        0,
        4,
    );
    beginPass(encoder, gpu.prepareIndirectPipeline, gpu.prepareIndirectBindGroup.group, 1);
    beginPass(encoder, gpu.syncTransformsPipeline, gpu.syncTransformsBindGroup.group, bodyWG);
    if (gpu.characterCount > 0) {
        beginPass(
            encoder,
            gpu.characterApplyPipeline,
            gpu.characterSweepBindGroup.group,
            Math.ceil(gpu.characterCount / 64),
            ts("phys:characterApply"),
        );
    }
    encoder.copyBufferToBuffer(
        gpu.bodyBuffer.buffer,
        0,
        gpu.bodyBufferPrev.buffer,
        0,
        bodyCount * BODY_BYTES,
    );
    encoder.copyBufferToBuffer(
        gpu.constraintsBuffer.buffer,
        0,
        gpu.prevConstraintsBuffer.buffer,
        0,
        prevCopySize,
    );
    beginPass(encoder, gpu.clearHashPipeline, rebuildGroup, hashWG, ts("phys:rebuild"));
    beginPassIndirect(
        encoder,
        gpu.rebuildPipeline,
        rebuildGroup,
        gpu.indirectBuffer,
        MAX_COLORS * 12,
        ts("phys:rebuild"),
    );

    dispatchPhysicsLBVH(gpu.lbvh, encoder, gpu.device, bodyCount, ts);

    encoder.clearBuffer(gpu.solverStateBuffer.buffer, 0, COUNTERS_SIZE);
    const solverGroup = gpu.solverBindGroup.group;
    const narrowGroup = gpu.narrowBindGroup.group;
    const pairGroup = gpu.pairBindGroup.group;
    const bvhPass = beginComputePass(encoder, ts("phys:broadphase"));
    bvhPass.setPipeline(gpu.bvhTraversalPipeline);
    bvhPass.setBindGroup(0, narrowGroup);
    bvhPass.setBindGroup(1, pairGroup);
    bvhPass.dispatchWorkgroups(bodyWG);
    bvhPass.end();

    beginPass(encoder, gpu.prepareIndirectPipeline, gpu.prepareIndirectBindGroup.group, 1);
    const npPass = beginComputePass(encoder, ts("phys:narrowphase"));
    for (let t = 0; t < NUM_PAIR_TYPES; t++) {
        npPass.setPipeline(gpu.narrowphasePipelines[t]);
        npPass.setBindGroup(0, narrowGroup);
        npPass.setBindGroup(1, pairGroup);
        npPass.dispatchWorkgroupsIndirect(gpu.indirectBuffer, (MAX_COLORS + 1 + t) * 12);
    }
    npPass.end();
    beginPass(
        encoder,
        gpu.detectJointsPipeline,
        solverGroup,
        Math.ceil(gpu.jointSlot.capacity / 64),
        ts("phys:broadphase"),
    );
    beginPass(encoder, gpu.warmstartPipeline, solverGroup, bodyWG, ts("phys:warmstart"));
    encoder.clearBuffer(gpu.forceBuffer.buffer, 0, bodyCount * 32);
    beginPass(encoder, gpu.prepareIndirectPipeline, gpu.prepareIndirectBindGroup.group, 1);
    beginPass(encoder, gpu.initBodyCachePipeline, solverGroup, bodyWG, ts("phys:warmstart"));
    beginPassIndirect(
        encoder,
        gpu.cacheContactCPipeline,
        solverGroup,
        gpu.indirectBuffer,
        MAX_COLORS * 12,
        ts("phys:warmstart"),
    );

    beginPass(encoder, gpu.clearColorPipeline, solverGroup, bodyWG, ts("phys:coloring"));
    beginPassIndirect(
        encoder,
        gpu.countBodyConstraintsPipeline,
        solverGroup,
        gpu.indirectBuffer,
        MAX_COLORS * 12,
        ts("phys:coloring"),
    );
    encoder.copyBufferToBuffer(
        gpu.solverStateBuffer.buffer,
        solverStateSize() + csrHeadsOffset() * 4,
        gpu.csrCountsBuffer.buffer,
        0,
        (capacity() + 1) * 4,
    );
    {
        const pass = beginComputePass(encoder, ts("phys:coloring"));
        dispatchPrefixSum(gpu.csrPrefixSum, pass);
        pass.end();
    }
    encoder.copyBufferToBuffer(
        gpu.csrCountsBuffer.buffer,
        0,
        gpu.solverStateBuffer.buffer,
        solverStateSize() + csrOffsetsOffset() * 4,
        (capacity() + 1) * 4,
    );
    encoder.copyBufferToBuffer(
        gpu.csrCountsBuffer.buffer,
        0,
        gpu.solverStateBuffer.buffer,
        solverStateSize() + csrHeadsOffset() * 4,
        (capacity() + 1) * 4,
    );
    beginPassIndirect(
        encoder,
        gpu.scatterBodyConstraintsPipeline,
        solverGroup,
        gpu.indirectBuffer,
        MAX_COLORS * 12,
        ts("phys:coloring"),
    );
    const iterations = gpu.params.iterations;

    beginPass(encoder, gpu.buildAdjacencyPipeline, solverGroup, bodyWG, ts("phys:coloring"));
    for (let r = 0; r < COLORING_ROUNDS; r++) {
        beginPass(encoder, gpu.graphColorPipeline, solverGroup, bodyWG, ts("phys:coloring"));
    }
    beginPass(encoder, gpu.countColorsPipeline, solverGroup, bodyWG, ts("phys:coloring"));
    beginPass(encoder, gpu.prefixSumColorsPipeline, solverGroup, 1, ts("phys:coloring"));
    beginPass(
        encoder,
        gpu.prepareIndirectPipeline,
        gpu.prepareIndirectBindGroup.group,
        1,
        ts("phys:coloring"),
    );
    beginPass(encoder, gpu.sortBodiesPipeline, solverGroup, bodyWG, ts("phys:coloring"));

    beginPass(encoder, gpu.syncBodyColsPipeline, solverGroup, bodyWG, ts("phys:solve"));

    // Skip empty-color dispatches based on prior-frame's used-color count + safety margin.
    // Mac dispatches cost ~23 µs each; saving 3-5 empty colors saves ~0.3 ms/substep on Mac.
    // Safety margin tolerates color count drifting up by 1-2 between frames; readback may be 1-2 frames stale.
    const lastUsedColors = gpu.debugReadbackData[SS_USED_COLORS] | 0;
    const colorsToRun = lastUsedColors > 0 ? Math.min(MAX_COLORS, lastUsedColors + 2) : MAX_COLORS;

    for (let it = 0; it < iterations; it++) {
        for (let c = 0; c < colorsToRun; c++) {
            beginPassIndirect(
                encoder,
                gpu.primalPipeline,
                solverGroup,
                gpu.indirectBuffer,
                c * 12,
                ts("phys:solve"),
            );
            if (c < colorsToRun - 1) {
                beginPass(encoder, gpu.advanceColorPipeline, solverGroup, 1);
            }
        }
        beginPass(encoder, gpu.resetColorPipeline, solverGroup, 1);

        beginPassIndirect(
            encoder,
            gpu.dualPipeline,
            solverGroup,
            gpu.indirectBuffer,
            MAX_COLORS * 12,
            ts("phys:dual"),
        );
        beginPass(encoder, gpu.advancePipeline, solverGroup, 1, ts("phys:dual"));
    }
    beginPass(encoder, gpu.velocityPipeline, solverGroup, bodyWG, ts("phys:dual"));

    beginPassIndirect(
        encoder,
        gpu.writebackPipeline,
        solverGroup,
        gpu.indirectBuffer,
        MAX_COLORS * 12,
        ts("phys:writeback"),
    );

    beginPassIndirect(
        encoder,
        gpu.emitContactsPipeline,
        gpu.emitContactsBindGroup.group,
        gpu.indirectBuffer,
        MAX_COLORS * 12,
        ts("phys:contacts"),
    );

    if (gpu.characterCount > 0) {
        beginPass(
            encoder,
            gpu.characterSweepPipeline,
            gpu.characterSweepBindGroup.group,
            Math.ceil(gpu.characterCount / 64),
            ts("phys:characterSweep"),
        );
        for (let i = 0; i < gpu.characterCount; i++) {
            const bodyIdx = charIndicesScratch[i];
            encoder.copyBufferToBuffer(
                gpu.bodyBuffer.buffer,
                bodyIdx * BODY_BYTES,
                gpu.bodyBufferPrev.buffer,
                bodyIdx * BODY_BYTES,
                BODY_BYTES,
            );
        }
    }

    if (p) resolveProfile(encoder, p);

    gpu.physicsActive = true;

    gpu.device.pushErrorScope("validation");
    gpu.device.queue.submit([encoder.finish()]);
    gpu.device.popErrorScope().then((error) => {
        if (error) console.error("PHYSICS VALIDATION ERROR:", error.message);
    });
}

const compactParamsScratch = new Uint32Array(1);

function requestReadback(gpu: PhysicsGPU, tick: number): void {
    if (gpu.readbackPending) return;
    const bodyCount = gpu.bodyEids.length;
    if (bodyCount === 0) return;

    gpu.readbackPending = true;
    gpu.readbackGeneration = gpu.bodyGeneration;
    gpu.readbackBodyCount = bodyCount;
    gpu.readbackTick = tick;

    const compactByteSize = bodyCount * COMPACT_BYTES;
    const contactsOffset = COUNTERS_SIZE + compactByteSize;
    const totalSize = contactsOffset + CONTACTS_BYTES;

    if (gpu.readbackStaging.size < totalSize) {
        gpu.readbackStaging.destroy();
        gpu.readbackStaging = gpu.device.createBuffer({
            label: "physics-readback-staging",
            size: totalSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
    }

    if (gpu.transformReadbackData.length < bodyCount * COMPACT_STRIDE) {
        gpu.transformReadbackData = new Float32Array(bodyCount * COMPACT_STRIDE);
    }

    compactParamsScratch[0] = bodyCount;
    gpu.device.queue.writeBuffer(gpu.compactParamsBuffer, 0, compactParamsScratch);

    const enc = gpu.device.createCommandEncoder();
    const bodyWG = Math.ceil(bodyCount / 64);
    beginPass(enc, gpu.compactPipeline, gpu.compactBindGroup.group, bodyWG);
    enc.copyBufferToBuffer(gpu.solverStateBuffer.buffer, 0, gpu.readbackStaging, 0, COUNTERS_SIZE);
    enc.copyBufferToBuffer(
        gpu.compactBuffer.buffer,
        0,
        gpu.readbackStaging,
        COUNTERS_SIZE,
        compactByteSize,
    );
    enc.copyBufferToBuffer(
        gpu.contactsBuffer,
        0,
        gpu.readbackStaging,
        contactsOffset,
        CONTACTS_BYTES,
    );
    gpu.device.queue.submit([enc.finish()]);

    gpu.readbackStaging.mapAsync(GPUMapMode.READ, 0, totalSize).then(
        () => {
            if (gpu.readbackGeneration !== gpu.bodyGeneration) {
                gpu.readbackStaging.unmap();
                gpu.readbackPending = false;
                return;
            }
            const mapped = gpu.readbackStaging.getMappedRange(0, totalSize);
            const bytes = new ArrayBuffer(totalSize);
            new Uint8Array(bytes).set(new Uint8Array(mapped));
            gpu.readbackStaging.unmap();

            gpu.debugReadbackData.set(new Uint32Array(bytes, 0, COUNTERS_SIZE / 4));
            gpu.transformReadbackData.set(
                new Float32Array(bytes, COUNTERS_SIZE, bodyCount * COMPACT_STRIDE),
            );
            gpu.contactScratch.set(
                new Uint32Array(bytes, contactsOffset, MAX_CONTACTS * CONTACT_STRIDE),
            );

            gpu.readbackReady = true;
            gpu.readbackPending = false;
        },
        () => {
            gpu.readbackPending = false;
        },
    );
}

function processReadback(state: State, gpu: PhysicsGPU): void {
    if (gpu.readbackGeneration !== gpu.bodyGeneration) return;
    const tick = gpu.readbackTick;
    const bodyCount = gpu.readbackBodyCount;

    checkOverflows(gpu.debugReadbackData, tick, bodyCount);

    const rawCount = gpu.debugReadbackData[SS_CONTACT_COUNT];
    const overflow = gpu.debugReadbackData[SS_CONTACT_OVERFLOW];
    gpu.contactScratchCount = Math.min(rawCount, MAX_CONTACTS);
    gpu.contactScratchOverflow = overflow;
    if (overflow > 0) {
        console.warn(
            `[phys] tick=${tick} CONTACT OVERFLOW: ${overflow} dropped (cap=${MAX_CONTACTS})`,
        );
    }

    const contacts = Contacts.from(state);
    if (contacts) {
        const tmp = contacts.prevData;
        contacts.prevData = contacts.currentData;
        contacts.prevCount = contacts.currentCount;
        contacts.prevOverflow = contacts.currentOverflow;
        contacts.prevTick = contacts.currentTick;
        contacts.currentData = tmp;
        contacts.currentData.set(gpu.contactScratch);
        contacts.currentCount = gpu.contactScratchCount;
        contacts.currentOverflow = overflow;
        contacts.currentTick = tick;
    }

    const data = gpu.transformReadbackData;
    const { posX, posY, posZ, quatX, quatY, quatZ, quatW } = transformWasm;
    const eids = gpu.bodyEids;

    const charEids = gpu.characters;
    for (let i = 0; i < bodyCount; i++) {
        const eid = eids[i];
        if (Body.mass[eid] <= 0 && !charEids.includes(eid)) continue;
        const o = i * COMPACT_STRIDE;
        posX[eid] = data[o];
        posY[eid] = data[o + 1];
        posZ[eid] = data[o + 2];
        quatX[eid] = data[o + 3];
        quatY[eid] = data[o + 4];
        quatZ[eid] = data[o + 5];
        quatW[eid] = data[o + 6];
    }
    gpu.lastSyncTick = tick;
}

export async function readBodies(gpu: PhysicsGPU): Promise<Float32Array> {
    const count = gpu.bodyEids.length;
    if (count === 0) return new Float32Array(0);
    const byteSize = count * BODY_BYTES;
    const staging = gpu.device.createBuffer({
        size: byteSize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = gpu.device.createCommandEncoder();
    enc.copyBufferToBuffer(gpu.bodyBuffer.buffer, 0, staging, 0, byteSize);
    gpu.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ, 0, byteSize);
    const mapped = staging.getMappedRange(0, byteSize);
    const data = new Float32Array(mapped.slice(0));
    staging.unmap();
    staging.destroy();
    return data;
}

const PhysicsSystem: System = {
    group: "fixed",

    dispose(state: State) {
        const gpu = Physics.from(state);
        if (!gpu) return;
        gpu.bodyBuffer.buffer.destroy();
        gpu.bodyBufferPrev.buffer.destroy();
        gpu.bodyColsBuffer.buffer.destroy();
        gpu.constraintsBuffer.buffer.destroy();
        gpu.prevConstraintsBuffer.buffer.destroy();
        gpu.rebuildParamsBuffer.destroy();
        gpu.warmstartBuffer.buffer.destroy();
        gpu.solverStateBuffer.buffer.destroy();
        gpu.jointsBuffer.destroy();
        gpu.paramsBuffer.destroy();
        gpu.indirectBuffer.destroy();
        gpu.csrCountsBuffer.buffer.destroy();
        gpu.unpackTransformBuffer.buffer.destroy();
        gpu.sizesBuffer.buffer.destroy();
        gpu.shapesBuffer.buffer.destroy();
        gpu.bodyPropsBuffer.buffer.destroy();
        gpu.eidsBuffer.buffer.destroy();
        gpu.packParamsBuffer.destroy();
        gpu.readbackStaging.destroy();
        gpu.compactBuffer.buffer.destroy();
        gpu.compactParamsBuffer.destroy();
        gpu.pairBuffer.buffer.destroy();
        gpu.hullDataBuffer.buffer.destroy();
        gpu.hullIdsBuffer.buffer.destroy();
        gpu.characterBuffer.destroy();
        gpu.characterIndicesBuffer.destroy();
        gpu.characterParamsBuffer.destroy();
        gpu.characterGroundBuffer.destroy();
        gpu.characterReadbackStaging.destroy();
        disposePrefixSum(gpu.csrPrefixSum);
        disposePhysicsLBVH(gpu.lbvh);
        if (gpu.profile) {
            gpu.profile.querySet.destroy();
            gpu.profile.resolveBuffer.destroy();
            gpu.profile.readBuffer.destroy();
        }
    },

    update(state: State) {
        const gpu = Physics.from(state);
        if (!gpu) return;

        if (capacity() !== gpu.cachedCapacity) {
            gpu.cachedCapacity = capacity();
            rebuildPrefixSum(gpu.csrPrefixSum, gpu.csrCountsBuffer.buffer, capacity() + 1);
            for (const eid of gpu.bodyEids) pushChange(gpu, eid + 1);
            gpu.bodyEids.length = 0;
        }

        if (gpu.readbackReady) {
            processReadback(state, gpu);
            gpu.readbackReady = false;
        }

        const encoder = gpu.device.createCommandEncoder();

        processBodyChanges(gpu, state, encoder);
        rebuildCharacterIndices(gpu);
        updateCharacterInput(gpu);
        uploadCharacterData(gpu);
        syncTransforms(gpu, state);
        packForces(gpu, state);

        const bodyCount = gpu.bodyEids.length;
        if (bodyCount === 0) {
            gpu.device.queue.submit([encoder.finish()]);
            return;
        }

        uploadParams(gpu, bodyCount);
        dispatch(gpu, bodyCount, encoder);

        if (gpu.profile) readProfile(gpu.profile);

        requestReadback(gpu, state.time.fixedTick);
        requestCharacterReadback(gpu);
    },
};

function createInterpolationNode(
    physics: { bodyBufferPrev: GBuf; bodyBuffer: GBuf; eidsBuffer: GBuf },
    matrices: GBuf,
    getAlpha: () => number,
    getBodyCount: () => number,
    isReady: () => boolean,
    hasPendingChanges: () => boolean,
): ComputeNode {
    let pipeline: GPUComputePipeline | null = null;
    let interpBinding: Binding | null = null;
    let paramsBuffer: GPUBuffer | null = null;
    let paramsData: Float32Array | null = null;
    let paramsU32: Uint32Array | null = null;

    return {
        name: "physics-interpolation",
        scope: "frame",
        inputs: ["matrices"],
        outputs: ["matrices"],

        async prepare(device: GPUDevice) {
            const module = device.createShaderModule({ code: interpolateWGSL });
            pipeline = await device.createComputePipelineAsync({
                label: "interpolate",
                layout: "auto",
                compute: { module, entryPoint: "interpolate" },
            });
            paramsBuffer = device.createBuffer({
                label: "interp-params",
                size: 8,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            const pb = paramsBuffer;
            interpBinding = binding(device, pipeline.getBindGroupLayout(0), () => [
                { binding: 0, resource: { buffer: physics.bodyBufferPrev.buffer } },
                { binding: 1, resource: { buffer: physics.bodyBuffer.buffer } },
                { binding: 2, resource: { buffer: physics.eidsBuffer.buffer } },
                { binding: 3, resource: { buffer: pb } },
                { binding: 4, resource: { buffer: matrices.buffer } },
            ]);
            paramsData = new Float32Array(2);
            paramsU32 = new Uint32Array(paramsData.buffer);
        },

        execute(ctx: ExecutionContext) {
            if (!pipeline || !interpBinding || !paramsBuffer || !paramsData || !paramsU32) return;
            if (!isReady()) return;
            if (hasPendingChanges()) return;

            const bodyCount = getBodyCount();
            if (bodyCount === 0) return;

            paramsData[0] = getAlpha();
            paramsU32[1] = bodyCount;
            ctx.queue.writeBuffer(paramsBuffer, 0, paramsData as Float32Array<ArrayBuffer>);

            const pass = beginComputePass(
                ctx.encoder,
                ctx.timestampWrites?.("physics-interpolation"),
            );
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, interpBinding.group);
            pass.dispatchWorkgroups(Math.ceil(bodyCount / 64));
            pass.end();
        },
    };
}

export const PhysicsPlugin: Plugin = {
    name: "Physics",
    dependencies: [ComputePlugin, RenderPlugin],
    systems: [PhysicsSystem],
    components: { Body, Force, Impulse, Velocity, BallJoint, SpringJoint, Character, Move },

    async initialize(state: State) {
        const compute = Compute.from(state);
        if (!compute) {
            console.error("PhysicsPlugin: Compute resource not available");
            return;
        }

        const gpu = await createGPU(compute.device);
        state.setResource(Physics, gpu);
        state.setResource(Contacts, {
            prevData: new Uint32Array(MAX_CONTACTS * CONTACT_STRIDE),
            prevCount: 0,
            prevOverflow: 0,
            prevTick: -1,
            currentData: new Uint32Array(MAX_CONTACTS * CONTACT_STRIDE),
            currentCount: 0,
            currentOverflow: 0,
            currentTick: -1,
        });

        state.observe(onAdd(Body), (eid) => pushChange(gpu, eid + 1));
        state.observe(onRemove(Body), (eid) => pushChange(gpu, -(eid + 1)));

        const markJointsDirty = () => {
            gpu.jointsNeedUpload = true;
        };
        state.observe(onAdd(BallJoint), markJointsDirty);
        state.observe(onRemove(BallJoint), markJointsDirty);
        state.observe(onAdd(SpringJoint), markJointsDirty);
        state.observe(onRemove(SpringJoint), markJointsDirty);

        state.observe(onAdd(Character), (eid) => {
            gpu.characters.push(eid);
        });
        state.observe(onRemove(Character), (eid) => {
            const idx = gpu.characters.indexOf(eid);
            if (idx >= 0) gpu.characters.splice(idx, 1);
            gpu.characterVerticalVelocity.delete(eid);
            gpu.characterCoyoteTimers.delete(eid);
        });

        const render = Render.from(state);
        if (render) {
            const interpolationNode = createInterpolationNode(
                gpu,
                render.matrices,
                () => state.scheduler.accumulator / Time.FIXED_DT,
                () => gpu.bodyEids.length,
                () => gpu.physicsActive,
                () => gpu.pendingChangeCount > 0,
            );
            compute.graph.add(interpolationNode);
        }

        if (gpu.profile) {
            GpuProfile.from(state)?.push(gpu.profile.durations);
        }
    },
};
