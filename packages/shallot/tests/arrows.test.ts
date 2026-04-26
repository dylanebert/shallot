import { test, expect, describe, beforeEach } from "bun:test";
import { State, capacity, clearBuf } from "../src";
import { clearRegistry } from "../src/engine/ecs/component";
import { Arrow, ArrowData, ArrowsPlugin } from "../src/extras/arrows";
import { Line, LinesPlugin } from "../src/extras/lines";
import { Transform } from "../src/standard/transforms";
import { count, all } from "./helpers/state";

describe("Arrows", () => {
    describe("ArrowData storage", () => {
        test("is sized by capacity system", () => {
            expect(ArrowData.chunks[0].length).toBe(capacity() * 4);
        });

        test("is Float32Array", () => {
            expect(ArrowData.chunks[0]).toBeInstanceOf(Float32Array);
        });
    });

    describe("Arrow proxy accessors", () => {
        const eid = 42;

        beforeEach(() => {
            clearBuf(ArrowData);
        });

        test("start reads/writes correctly", () => {
            Arrow.start[eid] = 1;
            expect(Arrow.start[eid]).toBe(1);
        });

        test("end reads/writes correctly", () => {
            Arrow.end[eid] = 0;
            expect(Arrow.end[eid]).toBe(0);
        });

        test("size reads/writes correctly", () => {
            Arrow.size[eid] = 2.5;
            expect(Arrow.size[eid]).toBeCloseTo(2.5);
        });

        test("fields at different eids are independent", () => {
            Arrow.size[10] = 3;
            Arrow.size[11] = 7;
            expect(Arrow.size[10]).toBe(3);
            expect(Arrow.size[11]).toBe(7);
        });
    });

    describe("Arrow component defaults", () => {
        let state: State;

        beforeEach(() => {
            clearRegistry();
            state = new State();
            state.register(LinesPlugin);
            state.register(ArrowsPlugin);
        });

        test("Arrow defaults to end arrow, size 1", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Arrow);

            expect(Arrow.start[eid]).toBe(0);
            expect(Arrow.end[eid]).toBe(1);
            expect(Arrow.size[eid]).toBe(1);
        });
    });

    describe("ECS integration", () => {
        let state: State;

        beforeEach(() => {
            clearRegistry();
            clearBuf(ArrowData);
            state = new State();
            state.register(LinesPlugin);
            state.register(ArrowsPlugin);
        });

        test("query matches entities with Arrow, Line, and Transform", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Arrow);
            state.addComponent(eid, Line);
            state.addComponent(eid, Transform);

            expect(count(state, [Arrow, Line, Transform])).toBe(1);
        });

        test("query excludes entities missing Line", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Arrow);
            state.addComponent(eid, Transform);

            expect(count(state, [Arrow, Line, Transform])).toBe(0);
        });

        test("query excludes entities missing Transform", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Arrow);
            state.addComponent(eid, Line);

            expect(count(state, [Arrow, Line, Transform])).toBe(0);
        });

        test("query excludes entities missing Arrow", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Line);
            state.addComponent(eid, Transform);

            expect(count(state, [Arrow, Line, Transform])).toBe(0);
        });

        test("multiple Arrow entities all queryable", () => {
            for (let i = 0; i < 4; i++) {
                const eid = state.addEntity();
                state.addComponent(eid, Arrow);
                state.addComponent(eid, Line);
                state.addComponent(eid, Transform);
            }
            expect(count(state, [Arrow, Line, Transform])).toBe(4);
        });

        test("removing Arrow drops entity from query", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Arrow);
            state.addComponent(eid, Line);
            state.addComponent(eid, Transform);
            expect(count(state, [Arrow, Line, Transform])).toBe(1);

            state.removeComponent(eid, Arrow);
            expect(count(state, [Arrow, Line, Transform])).toBe(0);
        });

        test("dynamically added Arrow mid-session is queryable", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Line);
            state.addComponent(eid, Transform);
            state.step();

            state.addComponent(eid, Arrow);
            expect(count(state, [Arrow, Line, Transform])).toBe(1);
        });

        test("defaults applied on dynamic add", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Line);
            state.addComponent(eid, Transform);
            state.step();

            state.addComponent(eid, Arrow);
            expect(Arrow.start[eid]).toBe(0);
            expect(Arrow.end[eid]).toBe(1);
            expect(Arrow.size[eid]).toBe(1);
        });

        test("visibility filtering uses Line.visible", () => {
            const visible = state.addEntity();
            state.addComponent(visible, Arrow);
            state.addComponent(visible, Line);
            state.addComponent(visible, Transform);

            const hidden = state.addEntity();
            state.addComponent(hidden, Arrow);
            state.addComponent(hidden, Line);
            state.addComponent(hidden, Transform);
            Line.visible[hidden] = 0;

            const entities = all(state, [Arrow, Line, Transform]);
            const visibleEntities = entities.filter((e) => Line.visible[e]);
            expect(visibleEntities).toHaveLength(1);
            expect(visibleEntities[0]).toBe(visible);
        });

        test("start and end flags produce correct instance counts", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Arrow);
            state.addComponent(eid, Line);
            state.addComponent(eid, Transform);

            Arrow.start[eid] = 1;
            Arrow.end[eid] = 1;

            const entities = all(state, [Arrow, Line, Transform]);
            let instanceCount = 0;
            for (const e of entities) {
                if (Arrow.start[e]) instanceCount++;
                if (Arrow.end[e]) instanceCount++;
            }
            expect(instanceCount).toBe(2);
        });

        test("start=0 end=0 produces zero instances", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Arrow);
            state.addComponent(eid, Line);
            state.addComponent(eid, Transform);

            Arrow.start[eid] = 0;
            Arrow.end[eid] = 0;

            const entities = all(state, [Arrow, Line, Transform]);
            let instanceCount = 0;
            for (const e of entities) {
                if (Arrow.start[e]) instanceCount++;
                if (Arrow.end[e]) instanceCount++;
            }
            expect(instanceCount).toBe(0);
        });
    });
});
