import type { TgpuBindGroupLayout, TgpuFn } from "typegpu";
import tgpu, { isTgpuFn } from "typegpu";
import type { WgslArray } from "typegpu/data";
import * as d from "typegpu/data";
import type { Registry, World } from "../../engine";
import { backgroundsKey } from "./contract-state";
import type { MaterialBinding } from "./material-type";

type EntryFor<B extends MaterialBinding> = B extends { type: "uniform" }
    ? { uniform: B["struct"]; visibility: ("vertex" | "fragment")[] }
    : B extends { type: "storage" | "attribute" }
      ? {
            storage: (count: number) => WgslArray<B["element"]>;
            access: "mutable" | "readonly";
            visibility: ("vertex" | "fragment")[];
        }
      : B extends { type: "texture-2d" }
        ? { texture: d.WgslTexture2d<d.F32>; visibility: ("vertex" | "fragment")[] }
        : B extends { type: "texture-2d-array" }
          ? { texture: d.WgslTexture2dArray<d.F32>; visibility: ("vertex" | "fragment")[] }
          : B extends { type: "texture-depth-2d" }
            ? { texture: d.WgslTextureDepth2d; visibility: ("vertex" | "fragment")[] }
            : B extends { type: "sampler" }
              ? { sampler: "filtering"; visibility: ("vertex" | "fragment")[] }
              : { sampler: "comparison"; visibility: ("vertex" | "fragment")[] };

const VISIBILITY: ("vertex" | "fragment")[] = ["vertex", "fragment"];

function bindingEntry<B extends MaterialBinding>(binding: B): EntryFor<B> {
    switch (binding.type) {
        case "uniform":
            return { uniform: binding.struct, visibility: VISIBILITY } as EntryFor<B>;
        case "attribute":
            return {
                storage: d.arrayOf(binding.element),
                access: "readonly",
                visibility: VISIBILITY,
            } as EntryFor<B>;
        case "storage":
            return {
                storage: d.arrayOf(binding.element),
                access: binding.access === "read_write" ? "mutable" : "readonly",
                visibility: binding.visibility
                    ? [...binding.visibility]
                    : binding.access === "read_write"
                      ? ["fragment"]
                      : VISIBILITY,
            } as EntryFor<B>;
        case "texture-2d":
            return { texture: d.texture2d(), visibility: VISIBILITY } as EntryFor<B>;
        case "texture-2d-array":
            return { texture: d.texture2dArray(), visibility: VISIBILITY } as EntryFor<B>;
        case "texture-depth-2d":
            return { texture: d.textureDepth2d(), visibility: VISIBILITY } as EntryFor<B>;
        case "sampler":
            return { sampler: "filtering", visibility: VISIBILITY } as EntryFor<B>;
        case "sampler-comparison":
            return { sampler: "comparison", visibility: VISIBILITY } as EntryFor<B>;
    }
}

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

/** Build a background's resource layout before authoring the shader that closes over it. */
export function backgroundLayout<B extends Record<string, MaterialBinding>>(
    bindings: B,
): BackgroundLayout<B> {
    return tgpu.bindGroupLayout(ownEntries(bindings)).$idx(2) as BackgroundLayout<B>;
}

/** A background shader's reconstructed normalized world-space view ray. */
export const BackgroundContext = d.struct({ dir: d.vec3f }).$name("BackgroundContext");
export type BackgroundFn = TgpuFn<(ctx: typeof BackgroundContext) => d.Vec3f>;

/** A fullscreen background recipe; StandardRenderer supplies geometry and opaque alpha. */
export interface Background<
    B extends Record<string, MaterialBinding> = Record<string, MaterialBinding>,
> {
    name: string;
    layout: BackgroundLayout<B>;
    fs: BackgroundFn;
}

export const Backgrounds: import("../../engine").Resource<Registry<Background>> = {
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

/** Register a background for the lifetime of its owning World. */
export function registerBackground<B extends Record<string, MaterialBinding>>(
    world: World,
    spec: Background<B>,
): number {
    assertOwnFn(`registerBackground "${spec.name}" fs`, spec.fs);
    return world.resource(backgroundsKey).register(spec as Background);
}
