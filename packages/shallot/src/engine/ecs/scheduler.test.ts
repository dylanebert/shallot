import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import type { System } from "../..";
import { State, Time } from "../..";

describe("Scheduler", () => {
    let state: State;
    let executionOrder: string[];

    beforeEach(() => {
        state = new State();
        executionOrder = [];
    });

    describe("System Groups", () => {
        test("should run systems in group order: setup -> fixed -> simulation -> draw", () => {
            state.addSystem({
                group: "draw",
                update: () => executionOrder.push("draw"),
            });
            state.addSystem({
                group: "simulation",
                update: () => executionOrder.push("simulation"),
            });
            state.addSystem({
                group: "fixed",
                update: () => executionOrder.push("fixed"),
            });
            state.addSystem({
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
            state.addSystem({
                update: () => executionOrder.push("default"),
            });
            state.addSystem({
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

            state.addSystem({
                group: "fixed",
                update: () => fixedCount++,
            });
            state.addSystem({
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

            state.addSystem({
                group: "fixed",
                update: () => fixedCount++,
            });
            state.addSystem({
                group: "simulation",
                update: (s: State) => {
                    simDelta = s.time.deltaTime;
                },
            });

            state.step(1.0);

            const maxDt = Time.FIXED_DT * Time.MAX_FIXED_STEPS;
            expect(fixedCount).toBe(Time.MAX_FIXED_STEPS);
            // the clamp returns maxDt unchanged — same value, so equality is exact
            expect(simDelta).toBe(maxDt);
            expect(state.time.throttled).toBe(true);
            expect(state.time.rawDeltaTime).toBe(1.0);
        });

        test("should accumulate leftover time", () => {
            let fixedCount = 0;

            state.addSystem({
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

            state.addSystem({
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

            state.addSystem({
                group: "simulation",
                update: (s: State) => {
                    capturedDelta = s.time.deltaTime;
                },
            });

            state.step(frameDelta);

            // frameDelta is below the clamp cap, so it passes through unmodified
            expect(capturedDelta).toBe(frameDelta);
        });
    });

    describe("Time control", () => {
        test("default path: virtual clock equals real clock", () => {
            const dt = 0.016;
            state.step(dt);

            // at scale 1, unpaused: the virtual and real clocks are identical (the no-regression guarantee)
            expect(state.time.deltaTime).toBe(state.time.realDeltaTime);
            expect(state.time.elapsed).toBe(state.time.realElapsed);
            expect(state.time.deltaTime).toBe(dt);
            expect(state.time.scale).toBe(1);
            expect(state.time.paused).toBe(false);
        });

        test("pause freezes the virtual clock and fixed steps, real clock runs on", () => {
            let fixedCount = 0;
            state.addSystem({ group: "fixed", update: () => fixedCount++ });

            // prime the elapsed clocks
            state.step(0.02);
            const elapsed = state.time.elapsed;
            const realElapsed = state.time.realElapsed;
            fixedCount = 0;

            state.pause();
            state.step(0.02);

            expect(state.time.deltaTime).toBe(0);
            expect(state.time.elapsed).toBe(elapsed); // virtual frozen
            expect(fixedCount).toBe(0); // no fixed steps while paused
            expect(state.time.realDeltaTime).toBe(0.02); // real clock keeps ticking
            expect(state.time.realElapsed).toBeCloseTo(realElapsed + 0.02, 10);

            state.resume();
            state.step(0.02);
            expect(state.time.deltaTime).toBe(0.02);
            expect(state.time.elapsed).toBeCloseTo(elapsed + 0.02, 10);
        });

        test("timescale slows the virtual clock and halves fixed-tick frequency", () => {
            let fixedCount = 0;
            state.addSystem({ group: "fixed", update: () => fixedCount++ });

            const real = Time.FIXED_DT * 2;
            state.timescale(0.5);
            state.step(real);

            // scaled dt = real * 0.5 = one fixed step's worth, vs two at scale 1
            expect(state.time.deltaTime).toBeCloseTo(Time.FIXED_DT, 10);
            expect(state.time.realDeltaTime).toBeCloseTo(real, 10);
            expect(fixedCount).toBe(1);
        });

        test("negative timescale clamps to 0 (a freeze); 0 zeroes deltaTime like pause", () => {
            state.timescale(-1);
            state.step(0.016);
            expect(state.time.scale).toBe(0);
            expect(state.time.deltaTime).toBe(0);

            // pause is the resume-preserving path: it zeroes deltaTime without discarding the scale
            state.timescale(1);
            state.pause();
            state.step(0.016);
            expect(state.time.deltaTime).toBe(0);
            expect(state.time.scale).toBe(1);
        });

        test("pause from a fixed-group system does not cancel the current frame's queued steps", () => {
            let fixedCount = 0;
            state.addSystem({
                group: "fixed",
                update: (s: State) => {
                    fixedCount++;
                    s.pause(); // mid-frame — must not stop the steps already accumulated this frame
                },
            });

            state.step(Time.FIXED_DT * 3);

            expect(fixedCount).toBe(3); // all three queued steps ran
            expect(state.time.paused).toBe(true);

            // the freeze takes effect next frame
            fixedCount = 0;
            state.step(Time.FIXED_DT * 3);
            expect(fixedCount).toBe(0);
        });
    });

    describe("Setup Lifecycle", () => {
        test("should call setup only once per system", () => {
            let setupCount = 0;
            let updateCount = 0;

            state.addSystem({
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

            state.addSystem({
                setup: () => order.push("setup"),
                update: () => order.push("update"),
            });

            state.step();

            expect(order).toEqual(["setup", "update"]);
        });
    });

    describe("System Ordering Constraints", () => {
        test("should respect first constraint", () => {
            state.addSystem({
                group: "simulation",
                update: () => executionOrder.push("normal"),
            });
            state.addSystem({
                group: "simulation",
                first: true,
                update: () => executionOrder.push("first"),
            });

            state.step(Time.FIXED_DT);

            expect(executionOrder.indexOf("first")).toBeLessThan(executionOrder.indexOf("normal"));
        });

        test("should respect last constraint", () => {
            state.addSystem({
                group: "simulation",
                last: true,
                update: () => executionOrder.push("last"),
            });
            state.addSystem({
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

            state.addSystem(systemB);
            state.addSystem(systemA);

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

            state.addSystem(systemB);
            state.addSystem(systemA);

            state.step(Time.FIXED_DT);

            expect(executionOrder.indexOf("A")).toBeLessThan(executionOrder.indexOf("B"));
        });

        test("should handle first + last combination", () => {
            state.addSystem({
                group: "simulation",
                update: () => executionOrder.push("normal1"),
            });
            state.addSystem({
                group: "simulation",
                first: true,
                update: () => executionOrder.push("first"),
            });
            state.addSystem({
                group: "simulation",
                update: () => executionOrder.push("normal2"),
            });
            state.addSystem({
                group: "simulation",
                last: true,
                update: () => executionOrder.push("last"),
            });

            state.step(Time.FIXED_DT);

            expect(executionOrder[0]).toBe("first");
            expect(executionOrder[executionOrder.length - 1]).toBe("last");
        });

        test("should throw on first + last on same system", () => {
            state.addSystem({
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

            state.addSystem(systemA);
            state.addSystem(systemB);

            expect(() => state.step()).toThrow("Circular dependency");
        });
    });

    describe("Cache Invalidation", () => {
        test("should invalidate cache when systems change", () => {
            const system1 = {
                group: "simulation" as const,
                update: () => executionOrder.push("system1"),
            };

            state.addSystem(system1);
            state.step(Time.FIXED_DT);

            expect(executionOrder).toEqual(["system1"]);

            const system2 = {
                group: "simulation" as const,
                first: true,
                update: () => executionOrder.push("system2"),
            };
            state.addSystem(system2);

            executionOrder = [];
            state.step(Time.FIXED_DT);

            expect(executionOrder).toEqual(["system2", "system1"]);
        });
    });

    describe("System Mode", () => {
        test("default-annotated system runs when mode=play, skipped when mode=edit", () => {
            const sys: System = { update: () => executionOrder.push("default") };

            const sPlay = new State({ mode: "play" });
            sPlay.addSystem(sys);
            sPlay.step();
            expect(executionOrder).toEqual(["default"]);

            executionOrder = [];
            const sEdit = new State({ mode: "edit" });
            sEdit.addSystem(sys);
            sEdit.step();
            expect(executionOrder).toEqual([]);
        });

        test("always-annotated system runs regardless of mode", () => {
            const sys: System = {
                annotations: { mode: "always" },
                update: () => executionOrder.push("always"),
            };

            const sPlay = new State({ mode: "play" });
            sPlay.addSystem(sys);
            sPlay.step();
            expect(executionOrder).toEqual(["always"]);

            executionOrder = [];
            const sEdit = new State({ mode: "edit" });
            sEdit.addSystem(sys);
            sEdit.step();
            expect(executionOrder).toEqual(["always"]);
        });

        test("edit-annotated system runs when mode=edit, skipped when mode=play", () => {
            const sys: System = {
                annotations: { mode: "edit" },
                update: () => executionOrder.push("edit-system"),
            };

            const sEdit = new State({ mode: "edit" });
            sEdit.addSystem(sys);
            sEdit.step();
            expect(executionOrder).toEqual(["edit-system"]);

            executionOrder = [];
            const sPlay = new State({ mode: "play" });
            sPlay.addSystem(sys);
            sPlay.step();
            expect(executionOrder).toEqual([]);
        });

        test("undefined mode runs every system regardless of annotation", () => {
            state.addSystem({ update: () => executionOrder.push("default") });
            state.addSystem({
                annotations: { mode: "edit" },
                update: () => executionOrder.push("edit"),
            });
            state.addSystem({
                annotations: { mode: "always" },
                update: () => executionOrder.push("always"),
            });

            state.step();
            expect(executionOrder).toContain("default");
            expect(executionOrder).toContain("edit");
            expect(executionOrder).toContain("always");
        });

        test("setup runs only on the first step where the system is mode-active", () => {
            let setupCount = 0;
            const sys: System = {
                annotations: { mode: "play" },
                setup: () => setupCount++,
                update: () => executionOrder.push("play"),
            };

            const sEdit = new State({ mode: "edit" });
            sEdit.addSystem(sys);
            sEdit.step();
            sEdit.step();
            expect(setupCount).toBe(0);
            expect(executionOrder).toEqual([]);

            const sPlay = new State({ mode: "play" });
            sPlay.addSystem(sys);
            sPlay.step();
            expect(setupCount).toBe(1);
            expect(executionOrder).toEqual(["play"]);

            sPlay.step();
            expect(setupCount).toBe(1);
        });
    });

    // a reloaded system that throws must not wedge the frame loop (the editor's never-wedge bar):
    // it is paused after the first throw, reported once, and a swap (the next good edit) resumes it
    describe("Failure Recovery", () => {
        let errors: unknown[][];
        const origError = console.error;

        beforeEach(() => {
            errors = [];
            console.error = (...args: unknown[]) => {
                errors.push(args);
            };
        });

        afterEach(() => {
            console.error = origError;
        });

        test("a throwing update is paused after one throw, reported, and swap resumes it", () => {
            let badRuns = 0;
            const Bad: System = {
                name: "bad",
                update: () => {
                    badRuns++;
                    throw new Error("boom");
                },
            };
            state.addSystem(Bad);
            state.addSystem({ update: () => executionOrder.push("healthy") });

            state.step();
            state.step();
            expect(badRuns).toBe(1); // paused after the first throw
            expect(executionOrder).toEqual(["healthy", "healthy"]); // the loop survived
            expect(errors.length).toBe(1); // reported once, not per frame

            // the next good edit: a swap clears the pause and the new behavior runs
            state.swap(Bad, { update: () => executionOrder.push("fixed") });
            state.step();
            expect(executionOrder).toEqual(["healthy", "healthy", "fixed", "healthy"]);
        });

        test("a throwing setup is paused, then retried after a swap", () => {
            let setupRuns = 0;
            const Bad: System = {
                setup: () => {
                    setupRuns++;
                    throw new Error("setup boom");
                },
                update: () => executionOrder.push("bad-update"),
            };
            state.addSystem(Bad);

            state.step();
            state.step();
            expect(setupRuns).toBe(1);
            expect(executionOrder).toEqual([]); // update never ran past the failed setup
            expect(errors.length).toBe(1);

            // the swapped-in setup runs fresh — the failed one never marked initialized
            state.swap(Bad, {
                setup: () => executionOrder.push("setup-new"),
                update: () => executionOrder.push("update-new"),
            });
            state.step();
            expect(executionOrder).toEqual(["setup-new", "update-new"]);
        });
    });
});
