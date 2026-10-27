import type { TgpuBindGroupLayout, TgpuFn } from "typegpu";
import tgpu, { isTgpuFn } from "typegpu";
import * as d from "typegpu/data";
import type { World } from "../../engine";
import { validateBindingLimits } from "./binding-limits";
import { backgroundsKey } from "./contract-state";
import { engineLayout } from "./engine";
import { bindingEntry, type EntryFor, type MaterialBinding, refuseBindings } from "./material-type";
import type { BindResource } from "./pipelines";
import { shadowLayout } from "./shadow-layout";

function ownEntries<B extends Record<string, MaterialBinding>>(
    bindings: B,
): { [K in keyof B]: EntryFor<B[K]> } {
    return Object.fromEntries(
        Object.entries(bindings).map(([name, binding]) => [name, bindingEntry(binding)]),
    ) as { [K in keyof B]: EntryFor<B[K]> };
}

/** The resources a background's fragment shader reads from group 2. */
export type BackgroundLayout<B extends Record<string, MaterialBinding>> = TgpuBindGroupLayout<{
    [K in keyof B]: EntryFor<B[K]>;
}>;

/** Build a background's resource layout before authoring the shader that closes over it.
 * Per-stage budgets include the renderer's group-0 and group-1 entries. Exceeding a default WebGPU
 * binding limit is refused. */
export function backgroundLayout<B extends Record<string, MaterialBinding>>(
    bindings: B,
): BackgroundLayout<B> {
    refuseBindings("background", bindings);
    const layout = tgpu.bindGroupLayout(ownEntries(bindings)).$idx(2);
    validateBindingLimits("background layout", [engineLayout, shadowLayout, layout]);
    return layout as BackgroundLayout<B>;
}

/** A background shader's reconstructed normalized world-space view ray. */
export const BackgroundContext = d.struct({ dir: d.vec3f }).$name("BackgroundContext");
export type BackgroundFn = TgpuFn<(ctx: typeof BackgroundContext) => d.Vec3f>;

declare const backgroundHandleBrand: unique symbol;
/** A background's identity within its owning World. Names are display labels, not keys. */
export type BackgroundHandle = number & { readonly [backgroundHandleBrand]: true };

/** A fullscreen background recipe; StandardRenderer supplies geometry and opaque alpha. */
export interface Background<
    B extends Record<string, MaterialBinding> = Record<string, MaterialBinding>,
> {
    name: string;
    layout: BackgroundLayout<B>;
    fs: BackgroundFn;
    /** Live resources owned by this background; called at draw to observe replacements. */
    bindings?: (world: World) => Record<string, BindResource>;
}

export const Backgrounds: import("../../engine").Resource<ReadonlyMap<number, Background>> = {
    create: (world) => world.resource(backgroundsKey),
};

export function initializeBackgroundState(world: World): void {
    world.resource(backgroundsKey);
}

function assertOwnFn(label: string, fn: unknown): void {
    if (fn == null || isTgpuFn(fn)) return;
    throw new Error(
        `${label}: this isn't a TGSL function the engine can recognize. ` +
            "If it came from tgpu.fn, it resolved from a foreign copy of typegpu, not the one this " +
            "engine built against. Two physical copies in one bundle stamp different internal " +
            "markers even when the code is identical, so a shader built from a duplicate package " +
            "cannot resolve against the engine's metadata. Dedupe typegpu to a single copy; it is " +
            "the engine's peerDependency for exactly this reason. If this wasn't made with tgpu.fn, " +
            "build it with tgpu.fn(args, ret)(body).",
    );
}

/** Add a background for the lifetime of its owning World. Every call returns a distinct handle,
 * including calls with the same name. Cameras select it through CameraBackground.background.
 * Bindings resolve only from this recipe. */
export function registerBackground<B extends Record<string, MaterialBinding>>(
    world: World,
    spec: Background<B>,
): BackgroundHandle {
    assertOwnFn(`registerBackground "${spec.name}" fs`, spec.fs);
    const backgrounds = world.resource(backgroundsKey);
    const handle = backgrounds.size + 1;
    backgrounds.set(handle, { ...spec } as Background);
    return handle as BackgroundHandle;
}
