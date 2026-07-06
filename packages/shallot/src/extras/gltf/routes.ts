import { type State, type System, u32 } from "../../engine";
import type { Node } from "../../engine/scene";
import { Part } from "../../standard/part";
import { Surfaces } from "../../standard/render/core";
import { slab } from "../../standard/slab";
import type { GltfHandle } from "./assets";
import { Skin } from "./skin";

// #doc:dev
// ### Declarative load + route sync
//
// A scene references a glTF primitive by its registered mesh name (`part="mesh: model.glb#0"`); two
// pieces make that reference self-sufficient. The **preloader** (GltfPlugin registers it into the scene
// `Preloads` seam) scans parsed nodes with `scanRefs` and awaits `loadGltf` for every distinct source
// before `load` resolves the names. Godot's shape: the resource path in the scene is the load trigger.
// The **route sync** (`RouteSystem`) then converges every Part whose mesh resolves to a handle onto the
// route `placeGltf` would have wired: the textured/skinned surface, plus the `Textured` / `Skin`
// decoration carrying the union-relative material id. Both components are `derived` traits (a system
// owns them, `serialize` and the editor never see them) because their ids are recomputed per active
// set and cannot be authored.

/**
 * per-instance material id: an index into the per-material palette (`materialData`) the textured glTF
 * surfaces read. A `slab(u32)` published as `"materialIndex"`, so a surface samples
 * `albedo[materialData[id].layer]`. The firehose seam for textures: one more per-entity GPU column, no
 * new draws. Distinct from sear's `Material` (the per-instance PBR knobs): this is the
 * palette index, that is the shading params. A runtime-derived decoration. {@link GltfPlugin}'s route
 * sync owns it (the id is union-palette-relative, so scenes never author it).
 */
export const Textured = { id: slab(u32, "materialIndex") };

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

/** mesh id → its handle, repopulated as each asset registers (per build: mesh ids die with the
 *  registry). `register` writes it, {@link RouteSystem} reads it; internal seam, not on the barrel. */
export const routes = new Map<number, GltfHandle>();

// the surfaces the importer owns. A Part sitting on one of these — or on sear's `default` — follows its
// mesh's route (the effective default surface of a glTF mesh is its imported route); any other surface is
// an author's explicit choice and wins.
const ROUTE_SURFACES = [
    "gltf-albedo",
    "gltf-albedo-clip",
    "gltf-albedo-blend",
    "skin",
    "skin-clip",
    "skin-blend",
];

/**
 * converge each Part onto its mesh's route: surface + `Textured`/`Skin` follow the handle, and a mesh
 * edited off a glTF handle drops them. Compare-before-write throughout: an unconditional slab set would
 * dirty every decorated entity every frame. `mode: "always"` so the editor viewport renders textures;
 * the add/remove is sanctioned by the components' `derived` trait (nothing document-facing sees them).
 */
export const RouteSystem: System = {
    name: "GltfRoute",
    group: "simulation",
    annotations: { mode: "always" },
    update(state: State) {
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
                if (state.has(eid, Skin)) state.remove(eid, Skin);
                continue;
            }
            if (surface !== handle.surface && (surface === solid || owned.has(surface))) {
                Part.surface.set(eid, handle.surface);
            }
            if (handle.skinned) {
                if (!state.has(eid, Skin)) state.add(eid, Skin);
                if (Skin.anim.y.get(eid) !== handle.material) Skin.anim.y.set(eid, handle.material);
                // fround: the slab stores f32, the handle holds the f64 bake — compare in f32 or the
                // mismatch re-dirties the lane every frame
                const duration = Math.fround(handle.duration);
                if (Skin.anim.w.get(eid) !== duration) Skin.anim.w.set(eid, duration);
                if (state.has(eid, Textured)) state.remove(eid, Textured);
            } else if (handle.textured) {
                if (!state.has(eid, Textured)) state.add(eid, Textured);
                if (Textured.id.get(eid) !== handle.material) Textured.id.set(eid, handle.material);
                if (state.has(eid, Skin)) state.remove(eid, Skin);
            } else {
                if (state.has(eid, Textured)) state.remove(eid, Textured);
                if (state.has(eid, Skin)) state.remove(eid, Skin);
            }
        }
    },
};
