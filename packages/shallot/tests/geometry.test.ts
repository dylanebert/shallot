import { describe, test, expect, beforeEach } from "bun:test";
import { State } from "../src";
import { clearRegistry } from "../src/engine/ecs/component";
import { Part, SurfaceType, RenderPlugin } from "../src/standard/render";
import { Shape } from "../src/engine/utils";
import { count } from "./helpers/state";

describe("geometry CRUD", () => {
    let state: State;

    beforeEach(() => {
        clearRegistry();
        state = new State();
        state.register(RenderPlugin);
    });

    test("Part defaults include surface", () => {
        const eid = state.addEntity();
        state.addComponent(eid, Part);

        expect(Part.surface[eid]).toBe(SurfaceType.Default);
    });

    test("Part added mid-session is queryable", () => {
        state.step();

        const eid = state.addEntity();
        state.addComponent(eid, Part);
        Part.shape[eid] = Shape.Sphere;

        expect(count(state, [Part])).toBe(1);
    });

    test("multiple entities with Part all queryable", () => {
        const a = state.addEntity();
        state.addComponent(a, Part);
        const b = state.addEntity();
        state.addComponent(b, Part);

        state.step();

        expect(count(state, [Part])).toBe(2);
    });

    test("Part.surface persists custom value", () => {
        const eid = state.addEntity();
        state.addComponent(eid, Part);
        Part.surface[eid] = SurfaceType.Normals;

        state.step();

        expect(Part.surface[eid]).toBe(SurfaceType.Normals);
    });

    test("removing Part excludes entity from Part queries", () => {
        const eid = state.addEntity();
        state.addComponent(eid, Part);
        state.step();

        expect(count(state, [Part])).toBe(1);

        state.removeComponent(eid, Part);

        expect(count(state, [Part])).toBe(0);
    });

    test("Part defaults applied on add", () => {
        const eid = state.addEntity();
        state.addComponent(eid, Part);

        expect(Part.shape[eid]).toBe(Shape.Box);
        expect(Part.surface[eid]).toBe(SurfaceType.Default);
        expect(Part.color[eid]).toBe(0xffffff);
        expect(Part.opacity[eid]).toBe(1.0);
        expect(Part.sizeX[eid]).toBe(1);
        expect(Part.sizeY[eid]).toBe(1);
        expect(Part.sizeZ[eid]).toBe(1);
        expect(Part.roughness[eid]).toBe(1.0);
        expect(Part.reflectivity[eid]).toBe(0.0);
    });

    test("Part property changes persist across steps", () => {
        const eid = state.addEntity();
        state.addComponent(eid, Part);
        Part.shape[eid] = Shape.Plane;
        Part.color[eid] = 0xff0000;
        Part.sizeX[eid] = 2;
        Part.sizeY[eid] = 3;
        Part.sizeZ[eid] = 4;

        state.step();
        state.step();

        expect(Part.shape[eid]).toBe(Shape.Plane);
        expect(Part.color[eid]).toBe(0xff0000);
        expect(Part.sizeX[eid]).toBe(2);
        expect(Part.sizeY[eid]).toBe(3);
        expect(Part.sizeZ[eid]).toBe(4);
    });
});
