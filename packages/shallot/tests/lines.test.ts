import { test, expect, describe, beforeEach } from "bun:test";
import { State, capacity, clearBuf } from "../src";
import { clearRegistry } from "../src/engine/ecs/component";
import { LineData, Line, LinesPlugin } from "../src/extras/lines";
import { Transform } from "../src/standard/transforms";
import { count, all } from "./helpers/state";

describe("Lines", () => {
    describe("LineData storage", () => {
        test("is sized by capacity system", () => {
            expect(LineData.chunks[0].length).toBe(capacity() * 12);
        });

        test("is Float32Array", () => {
            expect(LineData.chunks[0]).toBeInstanceOf(Float32Array);
        });
    });

    describe("Line proxy accessors", () => {
        const eid = 42;

        beforeEach(() => {
            clearBuf(LineData);
        });

        test("offsetX reads/writes at correct offset", () => {
            Line.offsetX[eid] = 5;
            expect(Line.offsetX[eid]).toBe(5);
        });

        test("offsetY reads/writes at correct offset", () => {
            Line.offsetY[eid] = 10;
            expect(Line.offsetY[eid]).toBe(10);
        });

        test("offsetZ reads/writes at correct offset", () => {
            Line.offsetZ[eid] = -3;
            expect(Line.offsetZ[eid]).toBe(-3);
        });

        test("thickness reads/writes correctly", () => {
            Line.thickness[eid] = 4;
            expect(Line.thickness[eid]).toBe(4);
        });

        test("opacity reads/writes correctly", () => {
            Line.opacity[eid] = 0.5;
            expect(Line.opacity[eid]).toBeCloseTo(0.5);
        });

        test("visible reads/writes correctly", () => {
            Line.visible[eid] = 1;
            expect(Line.visible[eid]).toBe(1);
            Line.visible[eid] = 0;
            expect(Line.visible[eid]).toBe(0);
        });

        test("overdraw reads/writes correctly", () => {
            Line.overdraw[eid] = 1;
            expect(Line.overdraw[eid]).toBe(1);
            Line.overdraw[eid] = 0;
            expect(Line.overdraw[eid]).toBe(0);
        });

        test("color converts hex to RGBA floats", () => {
            Line.color[eid] = 0xff0000;
            expect(Line.colorR[eid]).toBeCloseTo(1.0);
            expect(Line.colorG[eid]).toBeCloseTo(0.0);
            expect(Line.colorB[eid]).toBeCloseTo(0.0);
            expect(LineData.chunks[0][eid * 12 + 11]).toBeCloseTo(1.0);
        });

        test("color reads back as hex", () => {
            Line.color[eid] = 0x00ff00;
            expect(Line.color[eid]).toBe(0x00ff00);
        });

        test("white color stores correctly", () => {
            Line.color[eid] = 0xffffff;
            expect(Line.colorR[eid]).toBeCloseTo(1.0);
            expect(Line.colorG[eid]).toBeCloseTo(1.0);
            expect(Line.colorB[eid]).toBeCloseTo(1.0);
        });

        test("colorR/G/B access individual linear channels", () => {
            Line.color[eid] = 0xff8040;
            expect(Line.colorR[eid]).toBeCloseTo(1.0, 3);
            expect(Line.colorG[eid]).toBeCloseTo(0.2159, 3);
            expect(Line.colorB[eid]).toBeCloseTo(0.0513, 3);
        });

        test("fields at different eids are independent", () => {
            Line.thickness[10] = 3;
            Line.thickness[11] = 7;
            expect(Line.thickness[10]).toBe(3);
            expect(Line.thickness[11]).toBe(7);
        });
    });

    describe("Line component defaults", () => {
        let state: State;

        beforeEach(() => {
            clearRegistry();
            clearBuf(LineData);
            state = new State();
            state.register(LinesPlugin);
        });

        test("Line defaults to offset(1,0,0), white, 2px thick, visible", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Line);

            expect(Line.offsetX[eid]).toBe(1);
            expect(Line.offsetY[eid]).toBe(0);
            expect(Line.offsetZ[eid]).toBe(0);
            expect(Line.color[eid]).toBe(0xffffff);
            expect(Line.thickness[eid]).toBe(2);
            expect(Line.opacity[eid]).toBe(1);
            expect(Line.visible[eid]).toBe(1);
        });
    });

    describe("ECS integration", () => {
        let state: State;

        beforeEach(() => {
            clearRegistry();
            clearBuf(LineData);
            state = new State();
            state.register(LinesPlugin);
        });

        test("query matches entities with Line and Transform", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Line);
            state.addComponent(eid, Transform);

            expect(count(state, [Line, Transform])).toBe(1);
        });

        test("query excludes entities missing Transform", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Line);

            expect(count(state, [Line, Transform])).toBe(0);
        });

        test("query excludes entities missing Line", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Transform);

            expect(count(state, [Line, Transform])).toBe(0);
        });

        test("multiple Line+Transform entities all queryable", () => {
            for (let i = 0; i < 5; i++) {
                const eid = state.addEntity();
                state.addComponent(eid, Line);
                state.addComponent(eid, Transform);
            }
            expect(count(state, [Line, Transform])).toBe(5);
        });

        test("removing Line drops entity from query", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Line);
            state.addComponent(eid, Transform);
            expect(count(state, [Line, Transform])).toBe(1);

            state.removeComponent(eid, Line);
            expect(count(state, [Line, Transform])).toBe(0);
        });

        test("dynamically added Line mid-session is queryable", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Transform);
            state.step();

            state.addComponent(eid, Line);
            expect(count(state, [Line, Transform])).toBe(1);
        });

        test("defaults applied on dynamic add", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Transform);
            state.step();

            state.addComponent(eid, Line);
            expect(Line.thickness[eid]).toBe(2);
            expect(Line.visible[eid]).toBe(1);
            expect(Line.offsetX[eid]).toBe(1);
        });

        test("overdraw field distinguishes line types", () => {
            const a = state.addEntity();
            state.addComponent(a, Line);
            state.addComponent(a, Transform);
            Line.overdraw[a] = 0;

            const b = state.addEntity();
            state.addComponent(b, Line);
            state.addComponent(b, Transform);
            Line.overdraw[b] = 1;

            const entities = all(state, [Line, Transform]);
            expect(entities).toHaveLength(2);

            const regular = entities.filter((e) => !Line.overdraw[e]);
            const overdraw = entities.filter((e) => Line.overdraw[e]);
            expect(regular).toHaveLength(1);
            expect(overdraw).toHaveLength(1);
        });

        test("visible flag filters as expected by system logic", () => {
            const visible = state.addEntity();
            state.addComponent(visible, Line);
            state.addComponent(visible, Transform);

            const hidden = state.addEntity();
            state.addComponent(hidden, Line);
            state.addComponent(hidden, Transform);
            Line.visible[hidden] = 0;

            const entities = all(state, [Line, Transform]);
            const visibleEntities = entities.filter((e) => Line.visible[e]);
            expect(visibleEntities).toHaveLength(1);
            expect(visibleEntities[0]).toBe(visible);
        });
    });
});
