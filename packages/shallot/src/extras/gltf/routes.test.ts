import { beforeEach, describe, expect, test } from "bun:test";
import { State } from "../../engine";
import { clear, register } from "../../engine/ecs/core";
import { parse } from "../../engine/scene";
import { Part, PartPlugin } from "../../standard/part";
import { Surfaces } from "../../standard/render/core";
import { Slab } from "../../standard/slab";
import type { GltfHandle } from "./assets";
import { GltfPlugin } from "./assets";
import { RouteSystem, routes, scanRefs, Textured } from "./routes";
import { Skin } from "./skin";

// Declarative-load scan + route sync, CPU-only: scanRefs is pure over parsed nodes, and RouteSystem is
// pure over a `new State()` + the module `routes` map (no device, no decode). The end-to-end declarative
// path on a real GPU (preload → load → textured render) is the gym `render` scenario's gltf modes.

describe("scanRefs", () => {
    test("collects distinct (src, clip) sources from part mesh refs", () => {
        const nodes = parse(`<scene>
            <a part="mesh: model.glb#0" transform />
            <a part="surface: unlit; mesh: model.glb#1" />
            <a part="mesh: env/level.gltf@clip2#3" />
        </scene>`);
        expect(scanRefs(nodes)).toEqual([
            { src: "model.glb", clip: 0 },
            { src: "env/level.gltf", clip: 2 },
        ]);
    });

    test("ignores non-glTF meshes, bare parts, and non-part attrs", () => {
        const nodes = parse(`<scene>
            <a part="mesh: cube" />
            <a part />
            <a text="mesh: fake.glb#0" />
            <a part="mesh: noindex.glb" />
        </scene>`);
        expect(scanRefs(nodes)).toEqual([]);
    });
});

describe("RouteSystem", () => {
    let albedo: number;
    let skinSurf: number;
    let solid: number;

    beforeEach(() => {
        clear();
        Surfaces.clear();
        routes.clear();
        solid = Surfaces.register({ name: "default" });
        albedo = Surfaces.register({ name: "gltf-albedo" });
        skinSurf = Surfaces.register({ name: "skin" });
        Surfaces.register({ name: "checker" });
        for (const p of [PartPlugin, GltfPlugin]) {
            for (const [n, c] of Object.entries(p.components ?? {})) {
                register(n, c, p.traits?.[n]);
            }
        }
        Slab.collect();
    });

    function handle(over: Partial<GltfHandle>): GltfHandle {
        return {
            name: "m.glb#0",
            mesh: 5,
            surface: albedo,
            material: 0,
            color: [1, 1, 1, 1],
            skinned: false,
            textured: false,
            duration: 0,
            ...over,
        };
    }

    function partEntity(state: State, mesh: number): number {
        const eid = state.create();
        state.add(eid, Part);
        Part.mesh.set(eid, mesh);
        return eid;
    }

    test("a textured handle routes the default surface and decorates with Textured", () => {
        routes.set(5, handle({ textured: true, material: 7 }));
        const state = new State();
        const eid = partEntity(state, 5);

        RouteSystem.update!(state);

        expect(Part.surface.get(eid)).toBe(albedo);
        expect(state.has(eid, Textured)).toBe(true);
        expect(Textured.id.get(eid)).toBe(7);
        expect(state.has(eid, Skin)).toBe(false);
    });

    test("an author's explicit non-glTF surface wins over the route", () => {
        routes.set(5, handle({ textured: true, material: 7 }));
        const state = new State();
        const eid = partEntity(state, 5);
        const checker = Surfaces.id("checker")!;
        Part.surface.set(eid, checker);

        RouteSystem.update!(state);

        expect(Part.surface.get(eid)).toBe(checker);
        expect(Textured.id.get(eid)).toBe(7); // the material id still syncs — a custom surface may read it
    });

    test("a skinned handle decorates with Skin (material + duration lanes only)", () => {
        routes.set(
            6,
            handle({ mesh: 6, surface: skinSurf, skinned: true, material: 3, duration: 2.5 }),
        );
        const state = new State();
        const eid = partEntity(state, 6);

        RouteSystem.update!(state);

        expect(Part.surface.get(eid)).toBe(skinSurf);
        expect(state.has(eid, Skin)).toBe(true);
        expect(Skin.anim.y.get(eid)).toBe(3);
        expect(Skin.anim.w.get(eid)).toBe(2.5);
        expect(Skin.anim.x.get(eid)).toBe(0); // play time is SkinSystem's
        expect(Skin.anim.z.get(eid)).toBe(0); // phase stays per-instance
    });

    test("a mesh edited off a glTF handle drops the decorations and the routed surface", () => {
        routes.set(5, handle({ textured: true, material: 7 }));
        const state = new State();
        const eid = partEntity(state, 5);
        RouteSystem.update!(state);
        expect(state.has(eid, Textured)).toBe(true);

        Part.mesh.set(eid, 99);
        RouteSystem.update!(state);

        expect(state.has(eid, Textured)).toBe(false);
        expect(Part.surface.get(eid)).toBe(solid);
    });

    test("a mesh edited between handles swaps the decoration kind", () => {
        routes.set(5, handle({ textured: true, material: 7 }));
        routes.set(
            6,
            handle({ mesh: 6, surface: skinSurf, skinned: true, material: 3, duration: 1 }),
        );
        const state = new State();
        const eid = partEntity(state, 5);
        RouteSystem.update!(state);

        Part.mesh.set(eid, 6);
        RouteSystem.update!(state);

        expect(state.has(eid, Textured)).toBe(false);
        expect(state.has(eid, Skin)).toBe(true);
        expect(Part.surface.get(eid)).toBe(skinSurf);
    });

    test("a converged entity re-dirties nothing (compare-before-write)", () => {
        routes.set(5, handle({ textured: true, material: 7 }));
        routes.set(
            6,
            handle({ mesh: 6, surface: skinSurf, skinned: true, material: 3, duration: 1 / 3 }),
        );
        const state = new State();
        partEntity(state, 5);
        partEntity(state, 6);
        RouteSystem.update!(state);

        const slabs = [Part.surface, Part.mesh, Textured.id, Skin.anim] as unknown as Slab[];
        for (const s of slabs) s.dirty.fill(0);
        RouteSystem.update!(state);

        for (const s of slabs) {
            expect(s.dirty.every((w) => w === 0)).toBe(true);
        }
    });
});
