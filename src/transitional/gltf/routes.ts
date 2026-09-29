import { Surfaces } from "../../core/rendering";
import { field, type State, type System, u32 } from "../../engine";
import type { Node } from "../../engine/scene";
import { Part } from "../part";
import { liveSkin, Skin } from "../skin";
import type { GltfHandle } from "./assets";

/**
 * per-instance material id: an index into the per-material palette (`materialData`) the textured glTF
 * surfaces read through the Part input record's material index, sampling
 * `albedo[materialData[id].layer]` without another GPU column or draw. Distinct from sear's `Material` (the per-instance PBR knobs): this is the
 * palette index, that is the shading params. A runtime-derived decoration. {@link GltfPlugin}'s route
 * sync owns it (the id is union-palette-relative, so scenes never author it).
 */
// `Textured` is a `derived` trait — a system owns it, never authored or serialized — because its id is
// recomputed per active set and can't be authored.
export const Textured = { id: field(u32) };

// a registered glTF primitive name as a scene authors it: `src.glb#index`, with an optional baked-clip
// variant `src.glb@clipN#index` (specName's shape). The capture groups are (src, clip).
const MESH_REF = /^(.+\.(?:glb|gltf))(?:@clip(\d+))?#\d+$/i;

/** one distinct glTF source a scene references by mesh name: the unit the pre-load resolve imports. */
export interface GltfRef {
    src: string;
    clip: number;
}

// the `mesh:` value of one part attr string, without the codec (registry-free — the mesh name is
// unresolvable before its asset loads, which is the whole point of scanning here)
function meshValue(attr: string): string | null {
    for (const prop of attr.split(";")) {
        const colon = prop.indexOf(":");
        if (colon === -1) continue;
        if (prop.slice(0, colon).trim() !== "mesh") continue;
        return prop.slice(colon + 1).trim();
    }
    return null;
}

/**
 * scan parsed scene nodes for glTF mesh references (`part="mesh: model.glb#0"`, clip variants
 * `model.glb@clip2#0`) and return the distinct `(src, clip)` sources: what the glTF preloader awaits
 * `loadGltf` for before the scene loads.
 */
export function scanRefs(nodes: Node[]): GltfRef[] {
    const refs = new Map<string, GltfRef>();
    for (const node of nodes) {
        for (const attr of node.attrs) {
            if (attr.name !== "part") continue;
            const mesh = meshValue(attr.value);
            if (!mesh) continue;
            const m = MESH_REF.exec(mesh);
            if (!m) continue;
            const src = m[1];
            const clip = m[2] ? Number(m[2]) : 0;
            refs.set(`${src}|${clip}`, { src, clip });
        }
    }
    return [...refs.values()];
}

const routesKey = Symbol("shallot.gltf-routes");

/** Mesh handles registered by this State; ids are local to its registry. */
export function routesFor(state: State): Map<number, GltfHandle> {
    return state.resource(routesKey, () => new Map());
}

/** the surfaces the importer owns. A Part sitting on one of these — or on sear's `default` — follows its
 *  mesh's route (the effective default surface of a glTF mesh is its imported route); any other surface is
 *  an author's explicit choice and wins. Exported so the coverage test can assert the registered surface
 *  set ⊇ this list (a new route surface added without a row here is a silent gap in the route-sync `owned`
 *  set). */
export const ROUTE_SURFACES = [
    "gltf-albedo",
    "gltf-albedo-clip",
    "gltf-albedo-blend",
    "skin",
    "skin-clip",
    "skin-blend",
    "skin-live",
    "skin-live-clip",
    "skin-live-blend",
] as const;

// drop an entity's Skin decoration, freeing its live palette block first — a no-op for a VAT skinned entity
// (it never allocated one), so it's safe to call whenever Skin comes off regardless of the prior route.
function dropSkin(state: State, eid: number): void {
    if (!state.has(eid, Skin)) return;
    liveSkin(state).free(eid);
    state.remove(eid, Skin);
}

/**
 * converge each Part onto its mesh's route: surface + `Textured`/`Skin` follow the handle, and a mesh
 * edited off a glTF handle drops them. Compare-before-write throughout: an unconditional slab set would
 * dirty every decorated entity every frame. The add/remove is sanctioned by the components'
 * `derived` trait (nothing serialized sees them).
 */
export const RouteSystem: System = {
    name: "GltfRoute",
    group: "simulation",
    update(state: State) {
        const routes = routesFor(state);
        if (routes.size === 0) return;
        const solid = Surfaces.id("default") ?? 0;
        const owned = new Set<number>();
        for (const name of ROUTE_SURFACES) {
            const id = Surfaces.id(name);
            if (id !== undefined) owned.add(id);
        }
        for (const eid of state.query([Part])) {
            const handle = routes.get(Part.mesh.get(eid));
            const surface = Part.surface.get(eid);
            if (!handle) {
                // the mesh moved off a glTF handle (a live edit) — drop the route's decorations
                if (owned.has(surface)) Part.surface.set(eid, solid);
                if (state.has(eid, Textured)) state.remove(eid, Textured);
                dropSkin(state, eid);
                continue;
            }
            if (surface !== handle.surface && (surface === solid || owned.has(surface))) {
                Part.surface.set(eid, handle.surface);
            }
            if (handle.skinned) {
                // a live→VAT swap frees the prior live palette block (a no-op for an entity that never
                // allocated one — `dropSkin`'s own comment certifies `liveSkin(state).free` safe on a VAT entity)
                liveSkin(state).free(eid);
                if (!state.has(eid, Skin)) state.add(eid, Skin);
                if (Skin.anim.y.get(eid) !== handle.material) Skin.anim.y.set(eid, handle.material);
                // fround: the slab stores f32, the handle holds the f64 bake — compare in f32 or the
                // mismatch re-dirties the lane every frame
                const duration = Math.fround(handle.duration);
                if (Skin.anim.w.get(eid) !== duration) Skin.anim.w.set(eid, duration);
                if (state.has(eid, Textured)) state.remove(eid, Textured);
            } else if (handle.live) {
                if (!state.has(eid, Skin)) state.add(eid, Skin);
                // allocate the instance's palette block (idempotent — returns the existing base) and publish
                // it in lane x for the surface; a producer poses it, unposed it renders the bind pose. w = 0
                // so SkinSystem's clip-advance skips it.
                const base = liveSkin(state).alloc(eid, handle.jointCount, state.stamp(eid));
                if (Skin.anim.x.get(eid) !== base) Skin.anim.x.set(eid, base);
                if (Skin.anim.y.get(eid) !== handle.material) Skin.anim.y.set(eid, handle.material);
                if (Skin.anim.w.get(eid) !== 0) Skin.anim.w.set(eid, 0);
                if (state.has(eid, Textured)) state.remove(eid, Textured);
            } else if (handle.textured) {
                if (!state.has(eid, Textured)) state.add(eid, Textured);
                if (Textured.id.get(eid) !== handle.material) Textured.id.set(eid, handle.material);
                dropSkin(state, eid);
            } else {
                if (state.has(eid, Textured)) state.remove(eid, Textured);
                dropSkin(state, eid);
            }
        }
    },
};
