import type { TgpuFn } from "typegpu";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import type { Resource, World } from "../../engine";

/** Inputs shared by every source of indirect light in standard materials and ambient fog. */
export const IndirectLightInput = d
    .struct({
        worldPosition: d.vec3f,
        normal: d.vec3f,
        view: d.vec3f,
        /** Material occlusion, or 1 for a volume sample; standard scales every source by this value. */
        materialOcclusion: d.f32,
        /** The view's selected global or camera ambient radiance, in cd/m². */
        ambientRadiance: d.vec3f,
    })
    .$name("IndirectLightInput");

/** One source returns indirect radiance for the supplied surface sample. Standard applies material occlusion. */
export type IndirectLightSource = TgpuFn<(input: typeof IndirectLightInput) => d.Vec3f>;

const ambientLightSource = tgpu
    .fn(
        [IndirectLightInput],
        d.vec3f,
    )((input) => {
        "use gpu";
        return std.mul(input.ambientRadiance, input.materialOcclusion);
    })
    .$name("ambientLightIndirect");

/** The source slot standard binds to the world's ambient light plus its registered sources. */
export const indirectLightSlot = tgpu.slot(ambientLightSource).$name("indirectLightSlot");

/** Evaluate the active sources; the pipeline root binds the world-specific source sum. */
export const indirectLight = tgpu
    .fn(
        [IndirectLightInput],
        d.vec3f,
    )((input) => {
        "use gpu";
        return indirectLightSlot.$(input);
    })
    .$name("indirectLight");

interface RegisteredSource {
    name: string;
    source: IndirectLightSource;
}

interface SourceRegistry {
    sources: RegisteredSource[];
    combined: IndirectLightSource | null;
    sealed: boolean;
}

const sourceRegistryKey = {
    create: (): SourceRegistry => ({ sources: [], combined: null, sealed: false }),
};

const sourceRegistryResource: Resource<SourceRegistry> = {
    create: (world) => world.resource(sourceRegistryKey),
};

function sourceRegistry(world: World): SourceRegistry {
    return world.resource(sourceRegistryResource);
}

/** Create the world-owned source list during standard renderer initialization. */
export function initializeIndirectLightSources(world: World): void {
    sourceRegistry(world);
}

/**
 * Register a TGSL source during plugin initialization. It is added after ambient light and receives the
 * world position, surface normal, view direction, material occlusion, and effective ambient radiance; its
 * return value is indirect radiance; standard multiplies each registered source by material occlusion,
 * then applies material albedo and camera exposure to the sum. Registration closes when standard warms its
 * pipelines.
 */
export function registerIndirectLightSource(
    world: World,
    name: string,
    source: IndirectLightSource,
): void {
    if (!name) throw new Error("indirect-light source name must not be empty");
    const registry = sourceRegistry(world);
    for (let i = 0; i < registry.sources.length; i++) {
        const prior = registry.sources[i]!;
        if (prior.name !== name) continue;
        if (prior.source === source) return;
        throw new Error(`indirect-light source "${name}" is already registered`);
    }
    if (registry.sealed) {
        throw new Error(
            `indirect-light source registration is closed after standard warm: "${name}"`,
        );
    }
    registry.sources.push({ name, source });
    registry.combined = null;
}

/** The ambient source followed by every world-registered source, composed outside frame execution. */
export function indirectLightSources(world: World): IndirectLightSource {
    const registry = sourceRegistry(world);
    if (registry.combined) return registry.combined;

    let combined = ambientLightSource;
    for (let i = 0; i < registry.sources.length; i++) {
        const { name, source } = registry.sources[i]!;
        const previous = combined;
        const suffix = name.replace(/[^a-zA-Z0-9_]/g, "_");
        combined = tgpu
            .fn(
                [IndirectLightInput],
                d.vec3f,
            )((input) => {
                "use gpu";
                return std.add(previous(input), std.mul(source(input), input.materialOcclusion));
            })
            .$name(`indirectLight_${i}_${suffix}`);
    }
    registry.combined = combined;
    return combined;
}

/** Freeze registrations once standard has finished collecting its pipeline inputs. */
export function sealIndirectLightSources(world: World): void {
    sourceRegistry(world).sealed = true;
}
