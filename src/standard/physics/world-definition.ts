import type { Resource, World } from "../../engine";
import type { Shape } from "./api/shape";
import { CONTACT_RECYCLE_DISTANCE } from "./common/constants";
import { defaultWorldDef, type WorldDef } from "./common/types";

export type WorldCustomFilterCallback = (world: World, shapeA: Shape, shapeB: Shape) => boolean;
export type WorldPreSolveCallback = (
    world: World,
    shapeA: Shape,
    shapeB: Shape,
    point: { x: number; y: number; z: number },
    normal: { x: number; y: number; z: number },
) => boolean;
export type WorldMixCallback = (
    world: World,
    valueA: number,
    materialIdA: bigint,
    valueB: number,
    materialIdB: bigint,
) => number;

/** The replaceable simulation's Box3D world settings and the stepped world policy. */
export type PhysicsWorldDefinitionConfig = Omit<
    WorldDef,
    "userData" | "frictionCallback" | "restitutionCallback"
> & {
    /** Solver substeps per fixed tick. Box3D's recommended default is four. */
    subStepCount: number;
    /** Whether contact and joint impulses are warm-started. */
    enableWarmStarting: boolean;
    /** Whether mesh contacts may use speculative points. */
    enableSpeculative: boolean;
    /** Contact separation at which a contact is recycled. */
    contactRecycleDistance: number;
    /** Optional collision-pair filter, called at the solver's serial callback point. */
    customFilterCallback: WorldCustomFilterCallback | null;
    /** Optional contact inspection, called at the solver's serial callback point. */
    preSolveCallback: WorldPreSolveCallback | null;
    /** Optional friction mixing rule; omitted means Box3D's geometric-mean default. */
    frictionCallback: WorldMixCallback | null;
    /** Optional restitution mixing rule; omitted means Box3D's maximum-value default. */
    restitutionCallback: WorldMixCallback | null;
};

/**
 * World-owned authoring resource for the complete standard-physics simulation definition.
 * Mutate it directly; standard physics compares it with its live solver settings before each
 * fixed step, so writes survive frames that contain no fixed tick. Host-owned data and task-system
 * callbacks are omitted; capacity only affects the world's initial reservation.
 */
export const PhysicsWorldDefinition: Resource<PhysicsWorldDefinitionConfig> = {
    create: () => {
        const definition = defaultWorldDef();
        return {
            ...definition,
            gravity: { ...definition.gravity },
            subStepCount: 4,
            enableWarmStarting: true,
            enableSpeculative: true,
            contactRecycleDistance: CONTACT_RECYCLE_DISTANCE,
            customFilterCallback: null,
            preSolveCallback: null,
            frictionCallback: null,
            restitutionCallback: null,
        };
    },
};

export function copyPhysicsWorldDefinition(
    definition: PhysicsWorldDefinitionConfig,
): PhysicsWorldDefinitionConfig {
    return {
        ...definition,
        gravity: { ...definition.gravity },
        capacity: definition.capacity ? { ...definition.capacity } : undefined,
    };
}
