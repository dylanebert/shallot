import { test, expect, describe, beforeEach } from "bun:test";
import { State, Time } from "../src";
import type { System } from "../src";

describe("Scheduler", () => {
    let state: State;
    let executionOrder: string[];

    beforeEach(() => {
        state = new State();
        executionOrder = [];
    });

    describe("System Groups", () => {
        test("should run systems in group order: setup -> fixed -> simulation -> draw", () => {
            state.register({
                group: "draw",
                update: () => executionOrder.push("draw"),
            });
            state.register({
                group: "simulation",
                update: () => executionOrder.push("simulation"),
            });
            state.register({
                group: "fixed",
                update: () => executionOrder.push("fixed"),
            });
            state.register({
                group: "setup",
                update: () => executionOrder.push("setup"),
            });

            state.step(Time.FIXED_DT);

            expect(executionOrder.indexOf("setup")).toBeLessThan(executionOrder.indexOf("fixed"));
            expect(executionOrder.indexOf("fixed")).toBeLessThan(
                executionOrder.indexOf("simulation"),
            );
            expect(executionOrder.indexOf("simulation")).toBeLessThan(
                executionOrder.indexOf("draw"),
            );
        });

        test("should default to simulation group", () => {
            state.register({
                update: () => executionOrder.push("default"),
            });
            state.register({
                group: "simulation",
                update: () => executionOrder.push("explicit-sim"),
            });

            state.step(Time.FIXED_DT);

            expect(executionOrder).toContain("default");
            expect(executionOrder).toContain("explicit-sim");
        });
    });

    describe("Fixed Timestep", () => {
        test("should run fixed systems multiple times for large delta", () => {
            let fixedCount = 0;
            let simCount = 0;

            state.register({
                group: "fixed",
                update: () => fixedCount++,
            });
            state.register({
                group: "simulation",
                update: () => simCount++,
            });

            state.step(Time.FIXED_DT * 3);

            expect(fixedCount).toBe(3);
            expect(simCount).toBe(1);
        });

        test("should cap fixed steps to prevent spiral of death", () => {
            let fixedCount = 0;
            let simDelta = 0;

            state.register({
                group: "fixed",
                update: () => fixedCount++,
            });
            state.register({
                group: "simulation",
                update: (s: State) => {
                    simDelta = s.time.deltaTime;
                },
            });

            state.step(1.0);

            const maxDt = Time.FIXED_DT * Time.MAX_FIXED_STEPS;
            expect(fixedCount).toBe(Time.MAX_FIXED_STEPS);
            expect(simDelta).toBeCloseTo(maxDt, 10);
            expect(state.time.throttled).toBe(true);
            expect(state.time.rawDeltaTime).toBe(1.0);
        });

        test("should accumulate leftover time", () => {
            let fixedCount = 0;

            state.register({
                group: "fixed",
                update: () => fixedCount++,
            });

            state.step(0.015);
            expect(fixedCount).toBe(0);

            state.step(0.015);
            expect(fixedCount).toBe(1);
        });

        test("should provide fixedDeltaTime to fixed systems", () => {
            let capturedDelta = 0;

            state.register({
                group: "fixed",
                update: (s: State) => {
                    capturedDelta = s.time.deltaTime;
                },
            });

            state.step(0.1);

            expect(capturedDelta).toBe(Time.FIXED_DT);
        });

        test("should provide frame deltaTime to simulation systems", () => {
            let capturedDelta = 0;
            const frameDelta = 0.016;

            state.register({
                group: "simulation",
                update: (s: State) => {
                    capturedDelta = s.time.deltaTime;
                },
            });

            state.step(frameDelta);

            expect(capturedDelta).toBeCloseTo(frameDelta, 5);
        });
    });

    describe("Setup Lifecycle", () => {
        test("should call setup only once per system", () => {
            let setupCount = 0;
            let updateCount = 0;

            state.register({
                setup: () => setupCount++,
                update: () => updateCount++,
            });

            state.step();
            state.step();
            state.step();

            expect(setupCount).toBe(1);
            expect(updateCount).toBe(3);
        });

        test("should call setup before first update", () => {
            const order: string[] = [];

            state.register({
                setup: () => order.push("setup"),
                update: () => order.push("update"),
            });

            state.step();

            expect(order).toEqual(["setup", "update"]);
        });
    });

    describe("System Ordering Constraints", () => {
        test("should respect first constraint", () => {
            state.register({
                group: "simulation",
                update: () => executionOrder.push("normal"),
            });
            state.register({
                group: "simulation",
                first: true,
                update: () => executionOrder.push("first"),
            });

            state.step(Time.FIXED_DT);

            expect(executionOrder.indexOf("first")).toBeLessThan(executionOrder.indexOf("normal"));
        });

        test("should respect last constraint", () => {
            state.register({
                group: "simulation",
                last: true,
                update: () => executionOrder.push("last"),
            });
            state.register({
                group: "simulation",
                update: () => executionOrder.push("normal"),
            });

            state.step(Time.FIXED_DT);

            expect(executionOrder.indexOf("normal")).toBeLessThan(executionOrder.indexOf("last"));
        });

        test("should respect before constraint", () => {
            const systemB: System = {
                group: "simulation",
                update: () => executionOrder.push("B"),
            };
            const systemA: System = {
                group: "simulation",
                before: [systemB],
                update: () => executionOrder.push("A"),
            };

            state.register(systemB);
            state.register(systemA);

            state.step(Time.FIXED_DT);

            expect(executionOrder.indexOf("A")).toBeLessThan(executionOrder.indexOf("B"));
        });

        test("should respect after constraint", () => {
            const systemA: System = {
                group: "simulation",
                update: () => executionOrder.push("A"),
            };
            const systemB: System = {
                group: "simulation",
                after: [systemA],
                update: () => executionOrder.push("B"),
            };

            state.register(systemB);
            state.register(systemA);

            state.step(Time.FIXED_DT);

            expect(executionOrder.indexOf("A")).toBeLessThan(executionOrder.indexOf("B"));
        });

        test("should handle first + last combination", () => {
            state.register({
                group: "simulation",
                update: () => executionOrder.push("normal1"),
            });
            state.register({
                group: "simulation",
                first: true,
                update: () => executionOrder.push("first"),
            });
            state.register({
                group: "simulation",
                update: () => executionOrder.push("normal2"),
            });
            state.register({
                group: "simulation",
                last: true,
                update: () => executionOrder.push("last"),
            });

            state.step(Time.FIXED_DT);

            expect(executionOrder[0]).toBe("first");
            expect(executionOrder[executionOrder.length - 1]).toBe("last");
        });

        test("should throw on first + last on same system", () => {
            state.register({
                group: "simulation",
                first: true,
                last: true,
                update: () => {},
            });

            expect(() => state.step()).toThrow(
                "System cannot have both first and last constraints",
            );
        });

        test("should throw on circular dependency", () => {
            const systemA: System = {
                group: "simulation",
                update: () => {},
            };
            const systemB: System = {
                group: "simulation",
                after: [systemA],
                update: () => {},
            };
            (systemA as { after: System[] }).after = [systemB];

            state.register(systemA);
            state.register(systemB);

            expect(() => state.step()).toThrow("Circular dependency");
        });
    });

    describe("Cache Invalidation", () => {
        test("should invalidate cache when systems change", () => {
            const system1 = {
                group: "simulation" as const,
                update: () => executionOrder.push("system1"),
            };

            state.register(system1);
            state.step(Time.FIXED_DT);

            expect(executionOrder).toEqual(["system1"]);

            const system2 = {
                group: "simulation" as const,
                first: true,
                update: () => executionOrder.push("system2"),
            };
            state.register(system2);

            executionOrder = [];
            state.step(Time.FIXED_DT);

            expect(executionOrder).toEqual(["system2", "system1"]);
        });
    });

    describe("System Mode", () => {
        test("default mode is play — runs when Mode=play, skipped when Mode=edit", () => {
            state.scheduler.mode = "play";
            state.register({ update: () => executionOrder.push("default") });

            state.step();
            expect(executionOrder).toEqual(["default"]);

            executionOrder = [];
            state.scheduler.mode = "edit";
            state.step();
            expect(executionOrder).toEqual([]);
        });

        test("always mode runs regardless of Mode resource value", () => {
            state.register({
                annotations: { mode: "always" },
                update: () => executionOrder.push("always"),
            });

            state.scheduler.mode = "play";
            state.step();
            expect(executionOrder).toEqual(["always"]);

            executionOrder = [];
            state.scheduler.mode = "edit";
            state.step();
            expect(executionOrder).toEqual(["always"]);
        });

        test("edit mode runs when Mode=edit, skipped when Mode=play", () => {
            state.register({
                annotations: { mode: "edit" },
                update: () => executionOrder.push("edit-system"),
            });

            state.scheduler.mode = "edit";
            state.step();
            expect(executionOrder).toEqual(["edit-system"]);

            executionOrder = [];
            state.scheduler.mode = "play";
            state.step();
            expect(executionOrder).toEqual([]);
        });

        test("no Mode resource — all systems run (backward compat)", () => {
            state.register({ update: () => executionOrder.push("default") });
            state.register({
                annotations: { mode: "edit" },
                update: () => executionOrder.push("edit"),
            });
            state.register({
                annotations: { mode: "always" },
                update: () => executionOrder.push("always"),
            });

            state.step();
            expect(executionOrder).toContain("default");
            expect(executionOrder).toContain("edit");
            expect(executionOrder).toContain("always");
        });

        test("mode switch mid-run changes which systems execute", () => {
            state.register({
                annotations: { mode: "play" },
                update: () => executionOrder.push("play-sys"),
            });
            state.register({
                annotations: { mode: "edit" },
                update: () => executionOrder.push("edit-sys"),
            });
            state.register({
                annotations: { mode: "always" },
                update: () => executionOrder.push("always-sys"),
            });

            state.scheduler.mode = "play";
            state.step();
            expect(executionOrder).toEqual(["play-sys", "always-sys"]);

            executionOrder = [];
            state.scheduler.mode = "edit";
            state.step();
            expect(executionOrder).toEqual(["edit-sys", "always-sys"]);
        });

        test("mode gates systems across fresh States", () => {
            const log: string[] = [];
            const playSystem: System = {
                annotations: { mode: "play" },
                update: () => log.push("play-ran"),
            };

            state.scheduler.mode = "edit";
            state.register(playSystem);
            state.step();
            expect(log).toEqual([]);

            const state2 = new State();
            state2.scheduler.mode = "play";
            state2.register(playSystem);
            state2.step();
            expect(log).toEqual(["play-ran"]);
        });

        test("setup called once — on first actual run after mode enables system", () => {
            let setupCount = 0;
            state.register({
                annotations: { mode: "play" },
                setup: () => setupCount++,
                update: () => executionOrder.push("play"),
            });

            state.scheduler.mode = "edit";
            state.step();
            state.step();
            expect(setupCount).toBe(0);
            expect(executionOrder).toEqual([]);

            state.scheduler.mode = "play";
            state.step();
            expect(setupCount).toBe(1);
            expect(executionOrder).toEqual(["play"]);

            state.step();
            expect(setupCount).toBe(1);
        });
    });
});
