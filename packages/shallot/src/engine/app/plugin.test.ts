import { describe, expect, test } from "bun:test";
import { build, type Plugin, type System, serialize, sparse, stringify, swap, u32 } from "../..";
import { clear, getComponent, getTraits } from "../ecs/core";
import { Compute } from "../runtime";

describe("Plugin", () => {
    describe("withPlugins", () => {
        test("should register a plugin with systems", async () => {
            let updateCalled = false;
            const TestSystem: System = {
                group: "simulation",
                update: () => {
                    updateCalled = true;
                },
            };

            const plugin: Plugin = {
                name: "Test",
                systems: [TestSystem],
            };

            const { state } = await build({ plugins: [plugin], defaults: false });
            state.step();

            expect(updateCalled).toBe(true);
        });

        test("should register multiple systems from plugin", async () => {
            const callOrder: string[] = [];

            const SystemA: System = {
                group: "simulation",
                update: () => callOrder.push("A"),
            };
            const SystemB: System = {
                group: "simulation",
                update: () => callOrder.push("B"),
            };

            const plugin: Plugin = {
                name: "Test",
                systems: [SystemA, SystemB],
            };

            const { state } = await build({ plugins: [plugin], defaults: false });
            state.step();

            expect(callOrder).toContain("A");
            expect(callOrder).toContain("B");
        });

        test("should register multiple plugins", async () => {
            const callOrder: string[] = [];

            const pluginA: Plugin = {
                name: "A",
                systems: [{ group: "simulation", update: () => callOrder.push("A") }],
            };
            const pluginB: Plugin = {
                name: "B",
                systems: [{ group: "simulation", update: () => callOrder.push("B") }],
            };

            const { state } = await build({ plugins: [pluginA, pluginB], defaults: false });
            state.step();

            expect(callOrder).toContain("A");
            expect(callOrder).toContain("B");
        });

        test("should include components in plugin definition", async () => {
            const TestComp = { value: [] as number[] };

            const plugin: Plugin = {
                name: "TestComp",
                components: { TestComp },
            };

            await build({ plugins: [plugin], defaults: false });

            expect(getComponent("test-comp")).toBeDefined();
        });
    });

    describe("missing dependencies", () => {
        test("should skip initialize when dependency is missing", async () => {
            let initialized = false;

            const Base: Plugin = { name: "Base" };
            const Dependent: Plugin = {
                name: "Dependent",
                dependencies: [Base],
                initialize: () => {
                    initialized = true;
                },
            };

            await build({ plugins: [Dependent], defaults: false });
            expect(initialized).toBe(false);
        });

        test("should only skip plugin with missing dep, not its dependents", async () => {
            let aInit = false;
            let bInit = false;

            const Base: Plugin = { name: "Base" };
            const A: Plugin = {
                name: "A",
                dependencies: [Base],
                initialize: () => {
                    aInit = true;
                },
            };
            const B: Plugin = {
                name: "B",
                dependencies: [A],
                initialize: () => {
                    bInit = true;
                },
            };

            await build({ plugins: [A, B], defaults: false });
            expect(aInit).toBe(false);
            expect(bInit).toBe(true);
        });

        test("should not skip plugin when dependency is present", async () => {
            let initialized = false;

            const Base: Plugin = { name: "Base" };
            const Dependent: Plugin = {
                name: "Dependent",
                dependencies: [Base],
                initialize: () => {
                    initialized = true;
                },
            };

            await build({ plugins: [Base, Dependent], defaults: false });
            expect(initialized).toBe(true);
        });
    });

    // the plugin-swap half of the reload-safety tier (testing.md "Reload tier"); the component-identity
    // half lives in ecs/ecs.test.ts "stable component ids". The in-place hot-swap e2e died with the
    // editor — `swap()` coverage is these unit tests alone.
    describe("swap (hot reload)", () => {
        test("in-place swap preserves runtime state and applies new behavior", async () => {
            clear();
            const Counter = { n: sparse(u32) };
            const Step: System = {
                group: "simulation",
                update: (s) => {
                    for (const eid of s.query([Counter]))
                        Counter.n.set(eid, Counter.n.get(eid) + 1);
                },
            };
            const P1: Plugin = { name: "counter", components: { Counter }, systems: [Step] };

            const { state } = await build({ plugins: [P1], defaults: false });
            const eid = state.create();
            state.add(eid, Counter);
            state.step();
            state.step();
            expect(Counter.n.get(eid)).toBe(2);

            const Counter2 = { n: sparse(u32) };
            const Step2: System = {
                group: "simulation",
                update: (s) => {
                    for (const eid of s.query([Counter2]))
                        Counter2.n.set(eid, Counter2.n.get(eid) + 10);
                },
            };
            const P2: Plugin = {
                name: "counter",
                components: { Counter: Counter2 },
                systems: [Step2],
            };

            const result = await swap(state, [P1], [P2]);
            expect(result.ok).toBe(true);

            expect(Counter2.n.get(eid)).toBe(2);
            expect(state.has(eid, Counter2)).toBe(true);
            expect([...state.query([Counter2])]).toContain(eid);

            state.step();
            expect(Counter2.n.get(eid)).toBe(12);
        });

        // the editor keeps `prev` = the build-time plugins across consecutive swaps: a swap mutates
        // the live system objects in place, so the build-time objects stay the scheduler's identity
        test("a second swap pairs against the build-time plugins", async () => {
            clear();
            const Counter = { n: sparse(u32) };
            const P1: Plugin = {
                name: "counter",
                components: { Counter },
                systems: [
                    {
                        update: (s) => {
                            for (const e of s.query([Counter]))
                                Counter.n.set(e, Counter.n.get(e) + 1);
                        },
                    },
                ],
            };
            const { state } = await build({ plugins: [P1], defaults: false });
            const eid = state.create();
            state.add(eid, Counter);

            const Counter2 = { n: sparse(u32) };
            const P2: Plugin = {
                name: "counter",
                components: { Counter: Counter2 },
                systems: [
                    {
                        update: (s) => {
                            for (const e of s.query([Counter2]))
                                Counter2.n.set(e, Counter2.n.get(e) + 10);
                        },
                    },
                ],
            };
            expect((await swap(state, [P1], [P2])).ok).toBe(true);

            const Counter3 = { n: sparse(u32) };
            const P3: Plugin = {
                name: "counter",
                components: { Counter: Counter3 },
                systems: [
                    {
                        update: (s) => {
                            for (const e of s.query([Counter3]))
                                Counter3.n.set(e, Counter3.n.get(e) + 100);
                        },
                    },
                ],
            };
            expect((await swap(state, [P1], [P3])).ok).toBe(true);
            state.step();
            expect(Counter3.n.get(eid)).toBe(100);
        });

        test("a plugin-set change falls back to rebuild", async () => {
            clear();
            const P1: Plugin = { name: "one" };
            const { state } = await build({ plugins: [P1], defaults: false });
            const result = await swap(state, [P1], [P1, { name: "two" }]);
            expect(result.ok).toBe(false);
            expect(result.reason).toContain("plugin set");
        });

        test("a feature change falls back to rebuild", async () => {
            clear();
            const P1: Plugin = { name: "feat" };
            const { state } = await build({ plugins: [P1], defaults: false });
            const P2: Plugin = { name: "feat", features: ["subgroups"] };
            const result = await swap(state, [P1], [P2]);
            expect(result.ok).toBe(false);
            expect(result.reason).toContain("features");
        });

        test("a preferred-feature change falls back to rebuild (the device's granted set differs)", async () => {
            clear();
            const P1: Plugin = { name: "pref" };
            const { state } = await build({ plugins: [P1], defaults: false });
            const P2: Plugin = { name: "pref", preferredFeatures: ["subgroups"] };
            const result = await swap(state, [P1], [P2]);
            expect(result.ok).toBe(false);
            expect(result.reason).toContain("features");
        });

        test("a component schema change falls back to rebuild", async () => {
            clear();
            const A = { n: sparse(u32) };
            const P1: Plugin = { name: "shape", components: { A } };
            const { state } = await build({ plugins: [P1], defaults: false });

            const A2 = { n: sparse(u32), extra: sparse(u32) }; // added a field
            const P2: Plugin = { name: "shape", components: { A: A2 } };
            const result = await swap(state, [P1], [P2]);
            expect(result.ok).toBe(false);
            expect(result.reason).toContain("schema");
        });

        test("a system-set change falls back to rebuild", async () => {
            clear();
            const Sys: System = { update: () => {} };
            const P1: Plugin = { name: "sys", systems: [Sys] };
            const { state } = await build({ plugins: [P1], defaults: false });

            const P2: Plugin = { name: "sys", systems: [Sys, { update: () => {} }] };
            const result = await swap(state, [P1], [P2]);
            expect(result.ok).toBe(false);
            expect(result.reason).toContain("system set");
        });

        // initialize runs after systems are already swapped, so a throw there leaves the State
        // half-updated — the only safe recovery is the rebuild the ok:false path hands off to
        test("a mid-swap initialize throw reports a rebuild instead of throwing", async () => {
            clear();
            const Sys: System = { update: () => {} };
            const P1: Plugin = { name: "boom", systems: [Sys] };
            const { state } = await build({ plugins: [P1], defaults: false });

            const P2: Plugin = {
                name: "boom",
                systems: [{ update: () => {} }],
                initialize: () => {
                    throw new Error("init boom");
                },
            };
            const result = await swap(state, [P1], [P2]);
            expect(result.ok).toBe(false);
            expect(result.reason).toContain("initialize");
            expect(result.reason).toContain("init boom");
        });

        // a systemless skipped plugin has no scheduler trace for the liveness guard to catch, so
        // swap validates against the skip set build() exposes — built reality, not the config
        test("swap rejects a systemless plugin skipped at build, before half-applying it", async () => {
            clear();
            const Missing: Plugin = { name: "missing" };
            let inits = 0;
            const P1: Plugin = {
                name: "dependent",
                dependencies: [Missing],
                initialize: () => {
                    inits++;
                },
            };
            const app = await build({ plugins: [P1], defaults: false });
            expect(app.skipped).toContain("dependent");
            expect(inits).toBe(0);

            const P2: Plugin = {
                name: "dependent",
                dependencies: [Missing],
                initialize: () => {
                    inits++;
                },
            };
            const result = await swap(app.state, [P1], [P2], app.skipped);
            expect(result.ok).toBe(false);
            expect(result.reason).toContain("skipped");
            expect(inits).toBe(0); // the never-built initialize never ran on the live State
        });

        test("swap rejects a plugin whose systems never reached the scheduler (skipped at build)", async () => {
            clear();
            const Missing: Plugin = { name: "missing" };
            const P1: Plugin = {
                name: "dependent",
                dependencies: [Missing],
                systems: [{ update: () => {} }],
            };
            const { state } = await build({ plugins: [P1], defaults: false });

            const P2: Plugin = {
                name: "dependent",
                dependencies: [Missing],
                systems: [{ update: () => {} }],
            };
            const result = await swap(state, [P1], [P2]);
            expect(result.ok).toBe(false);
            expect(result.reason).toContain("not live");
        });

        // an ordering edge counts by *target*, not just length: `before: [A]` → `before: [B]`
        // reorders systems with no shape change, so the swap must catch the retarget. An in-plugin
        // ref resolves by slot (a reload recreates the object); an external ref by identity.
        test("a same-shape ordering-target change falls back to rebuild", async () => {
            clear();
            const A: System = { update: () => {} };
            const B: System = { update: () => {} };
            const C: System = { update: () => {}, before: [A] };
            const P1: Plugin = { name: "order", systems: [A, B, C] };
            const { state } = await build({ plugins: [P1], defaults: false });

            const A2: System = { update: () => {} };
            const B2: System = { update: () => {} };
            const C2: System = { update: () => {}, before: [B2] }; // same shape, retargeted
            const P2: Plugin = { name: "order", systems: [A2, B2, C2] };
            const result = await swap(state, [P1], [P2]);
            expect(result.ok).toBe(false);
            expect(result.reason).toContain("scheduling");
        });

        test("an added ordering edge falls back to rebuild", async () => {
            clear();
            const A: System = { update: () => {} };
            const B: System = { update: () => {} };
            const P1: Plugin = { name: "order", systems: [A, B] };
            const { state } = await build({ plugins: [P1], defaults: false });

            const A2: System = { update: () => {} };
            const B2: System = { update: () => {}, before: [A2] }; // gained an edge
            const P2: Plugin = { name: "order", systems: [A2, B2] };
            const result = await swap(state, [P1], [P2]);
            expect(result.ok).toBe(false);
            expect(result.reason).toContain("scheduling");
        });

        test("an external ordering-anchor retarget falls back to rebuild", async () => {
            clear();
            const Anchor1: System = { update: () => {} };
            const Anchor2: System = { update: () => {} };
            const Anchors: Plugin = { name: "anchors", systems: [Anchor1, Anchor2] };
            const Sys: System = { update: () => {}, after: [Anchor1] };
            const P1: Plugin = { name: "proj", systems: [Sys] };
            const { state } = await build({ plugins: [Anchors, P1], defaults: false });

            const Sys2: System = { update: () => {}, after: [Anchor2] };
            const P2: Plugin = { name: "proj", systems: [Sys2] };
            const result = await swap(state, [P1], [P2]);
            expect(result.ok).toBe(false);
            expect(result.reason).toContain("scheduling");
        });

        test("unchanged ordering across reloaded system objects still swaps", async () => {
            clear();
            const A: System = { update: () => {} };
            const C: System = { update: () => {}, before: [A] };
            const P1: Plugin = { name: "order", systems: [A, C] };
            const { state } = await build({ plugins: [P1], defaults: false });

            const A2: System = { update: () => {} };
            const C2: System = { update: () => {}, before: [A2] };
            const P2: Plugin = { name: "order", systems: [A2, C2] };
            expect((await swap(state, [P1], [P2])).ok).toBe(true);
        });

        test("an unchanged cross-plugin ordering ref survives a whole-set reload", async () => {
            clear();
            const ASys: System = { update: () => {} };
            const Pa: Plugin = { name: "a", systems: [ASys] };
            const BSys: System = { update: () => {}, after: [ASys] };
            const Pb: Plugin = { name: "b", systems: [BSys] };
            const { state } = await build({ plugins: [Pa, Pb], defaults: false });

            // a project reload hands in every plugin fresh; b's ref to a's system is a fresh
            // object too, resolved by its slot across the whole swapped set
            const ASys2: System = { update: () => {} };
            const Pa2: Plugin = { name: "a", systems: [ASys2] };
            const BSys2: System = { update: () => {}, after: [ASys2] };
            const Pb2: Plugin = { name: "b", systems: [BSys2] };
            expect((await swap(state, [Pa, Pb], [Pa2, Pb2])).ok).toBe(true);
        });

        test("a system scheduling change (same count) falls back to rebuild", async () => {
            clear();
            const Sys: System = { group: "simulation", update: () => {} };
            const P1: Plugin = { name: "sched", systems: [Sys] };
            const { state } = await build({ plugins: [P1], defaults: false });

            const Sys2: System = { group: "draw", update: () => {} }; // same count, moved group
            const P2: Plugin = { name: "sched", systems: [Sys2] };
            const result = await swap(state, [P1], [P2]);
            expect(result.ok).toBe(false);
            expect(result.reason).toContain("scheduling");
        });

        // swap re-runs initialize (registration-only + idempotent per the lifecycle contract), so an
        // edited initialize body repopulates module singletons with the reloaded code
        test("an initialize-body edit applies on swap", async () => {
            clear();
            let singleton = "";
            const P1: Plugin = {
                name: "init",
                initialize: () => {
                    singleton = "old";
                },
            };
            const { state } = await build({ plugins: [P1], defaults: false });
            expect(singleton).toBe("old");

            const P2: Plugin = {
                name: "init",
                initialize: () => {
                    singleton = "new";
                },
            };
            expect((await swap(state, [P1], [P2])).ok).toBe(true);
            expect(singleton).toBe("new");
        });

        // documented limit: a closure body can't be diffed, so a warm-body edit passes the swap
        // without applying — it lands on the next rebuild (see the swap JSDoc)
        test("a warm-body edit swaps without running the new warm", async () => {
            clear();
            let warmRuns = 0;
            const P1: Plugin = {
                name: "warmed",
                warm: () => {
                    warmRuns++;
                },
            };
            const { state } = await build({ plugins: [P1], defaults: false });
            expect(warmRuns).toBe(1);

            const P2: Plugin = {
                name: "warmed",
                warm: () => {
                    warmRuns += 100;
                },
            };
            expect((await swap(state, [P1], [P2])).ok).toBe(true);
            expect(warmRuns).toBe(1);
        });

        // documented limit, same class: the swapped system keeps its initialized mark, so an edited
        // setup body never runs on this State — only the new update goes live
        test("a setup-body edit swaps without re-running setup", async () => {
            clear();
            const log: string[] = [];
            const Sys: System = {
                setup: () => log.push("setup-old"),
                update: () => log.push("update-old"),
            };
            const P1: Plugin = { name: "lazy", systems: [Sys] };
            const { state } = await build({ plugins: [P1], defaults: false });
            state.step();
            expect(log).toEqual(["setup-old", "update-old"]);

            const Sys2: System = {
                setup: () => log.push("setup-new"),
                update: () => log.push("update-new"),
            };
            const P2: Plugin = { name: "lazy", systems: [Sys2] };
            expect((await swap(state, [P1], [P2])).ok).toBe(true);
            state.step();
            expect(log).toEqual(["setup-old", "update-old", "update-new"]);
        });

        // traits ride the re-registration: the registry serves the reloaded parse/format, and an
        // added exclude is enforced for adds through the reloaded handles.
        test("a traits parse/format change applies on swap", async () => {
            clear();
            const A = { value: sparse(u32) };
            const P1: Plugin = {
                name: "fmt",
                components: { A },
                traits: { A: { format: { value: (v) => `old-${v}` } } },
            };
            const { state } = await build({ plugins: [P1], defaults: false });
            expect(getTraits("a")?.format?.value(1)).toBe("old-1");

            const A2 = { value: sparse(u32) };
            const P2: Plugin = {
                name: "fmt",
                components: { A: A2 },
                traits: { A: { format: { value: (v) => `new-${v}` } } },
            };
            expect((await swap(state, [P1], [P2])).ok).toBe(true);
            expect(getTraits("a")?.format?.value(1)).toBe("new-1");
        });

        test("an added trait exclude is enforced after swap", async () => {
            clear();
            const A = { value: sparse(u32) };
            const B = { value: sparse(u32) };
            const P1: Plugin = { name: "excl", components: { A, B } };
            const { state } = await build({ plugins: [P1], defaults: false });

            const A2 = { value: sparse(u32) };
            const B2 = { value: sparse(u32) };
            const P2: Plugin = {
                name: "excl",
                components: { A: A2, B: B2 },
                traits: { A: { excludes: [B2] } },
            };
            expect((await swap(state, [P1], [P2])).ok).toBe(true);

            const eid = state.create();
            state.add(eid, B2);
            expect(() => state.add(eid, A2)).toThrow(/excluded/);

            // a handle held from before the reload resolves the current set — the
            // exclusion view keys on the stable id, not the object
            const other = state.create();
            state.add(other, B);
            expect(() => state.add(other, A)).toThrow(/excluded/);
        });

        test("a removed trait exclude stops being enforced after swap", async () => {
            clear();
            const A = { value: sparse(u32) };
            const B = { value: sparse(u32) };
            const P1: Plugin = {
                name: "excl",
                components: { A, B },
                traits: { A: { excludes: [B] } },
            };
            const { state } = await build({ plugins: [P1], defaults: false });

            const A2 = { value: sparse(u32) };
            const B2 = { value: sparse(u32) };
            const P2: Plugin = { name: "excl", components: { A: A2, B: B2 } };
            expect((await swap(state, [P1], [P2])).ok).toBe(true);

            const eid = state.create();
            state.add(eid, B2);
            state.add(eid, A2); // no throw — the exclusion view derives from current traits
            expect(state.has(eid, A2)).toBe(true);

            // the removal also reaches stale pre-reload handles
            const other = state.create();
            state.add(other, B);
            state.add(other, A);
            expect(state.has(other, A)).toBe(true);
        });

        // the rebuild-from-document fallback the swap rejections above hand off to: capture the
        // live State with `serialize`, rebuild from it with the SAME device, no page reload. A runtime
        // value on the authored entity survives (serialize reads it); a warm-derived entity re-derives
        // once, never doubled (serialize captures the authored set only). This is the non-editor path; the
        // editor's is the same `serialize`→`build` with its device reused via `ensureDevice`.
        test("rebuild from the serialized document preserves a runtime value, device reused", async () => {
            clear();
            const Counter = { n: sparse(u32) };
            const Derived = { of: sparse(u32) };
            const Tick: System = {
                group: "simulation",
                update: (s) => {
                    for (const e of s.query([Counter])) Counter.n.set(e, Counter.n.get(e) + 1);
                },
            };
            const P: Plugin = {
                name: "rebuildable",
                components: { Counter, Derived },
                systems: [Tick],
                warm: (s) => {
                    for (const c of s.query([Counter])) {
                        const d = s.create();
                        s.add(d, Derived);
                        Derived.of.set(d, c);
                    }
                },
            };

            const scene = `<scene><a id="hero" counter /></scene>`;
            const app = await build({ plugins: [P], defaults: false, scene });
            const device = Compute.device;

            app.state.step();
            app.state.step();
            app.state.step();
            const hero = app.state.only([Counter as never]);
            const runtimeN = Counter.n.get(hero);
            expect(runtimeN).toBeGreaterThan(0);
            expect([...app.state.query([Derived])].length).toBe(1);

            const xml = stringify(serialize(app.state));
            const rebuilt = await build({ plugins: [P], defaults: false, scene: xml, device });

            expect(Compute.device).toBe(device);
            const hero2 = rebuilt.state.only([Counter as never]);
            expect(Counter.n.get(hero2)).toBe(runtimeN);
            expect([...rebuilt.state.query([Derived])].length).toBe(1); // re-derived once, not doubled
        });
    });

    // a failed build must leave nothing live: plugins initialized so far dispose in reverse, then
    // the State (its system dispose hooks included), so the caller can rebuild against clean
    // module singletons
    describe("build failure", () => {
        test("a failed initialize disposes the initialized plugins in reverse, then the State", async () => {
            clear();
            const log: string[] = [];
            const A: Plugin = {
                name: "a",
                systems: [{ update: () => {}, dispose: () => log.push("sys-dispose") }],
                initialize: () => {
                    log.push("a-init");
                },
                dispose: () => log.push("a-dispose"),
            };
            const B: Plugin = {
                name: "b",
                initialize: () => {
                    log.push("b-init");
                },
                dispose: () => log.push("b-dispose"),
            };
            const C: Plugin = {
                name: "c",
                initialize: () => {
                    throw new Error("c boom");
                },
                dispose: () => log.push("c-dispose"),
            };

            await expect(build({ plugins: [A, B, C], defaults: false })).rejects.toThrow("c boom");
            expect(log).toEqual(["a-init", "b-init", "b-dispose", "a-dispose", "sys-dispose"]);
        });

        test("a dispose throw during cleanup is reported, never masks the build error", async () => {
            clear();
            const errors: unknown[][] = [];
            const origError = console.error;
            console.error = (...args: unknown[]) => {
                errors.push(args);
            };
            try {
                const A: Plugin = {
                    name: "a",
                    initialize: () => {},
                    dispose: () => {
                        throw new Error("dispose boom");
                    },
                };
                const B: Plugin = {
                    name: "b",
                    initialize: () => {
                        throw new Error("build boom");
                    },
                };
                await expect(build({ plugins: [A, B], defaults: false })).rejects.toThrow(
                    "build boom",
                );
                expect(errors.length).toBe(1);
            } finally {
                console.error = origError;
            }
        });
    });

    describe("plugin with full bundle", () => {
        test("should create a complete plugin with systems and components", async () => {
            let systemRan = false;

            const Health = { current: [] as number[], max: [] as number[] };
            const HealthSystem: System = {
                group: "simulation",
                update: () => {
                    systemRan = true;
                },
            };

            const HealthPlugin: Plugin = {
                name: "Health",
                systems: [HealthSystem],
                components: { Health },
            };

            const { state } = await build({ plugins: [HealthPlugin], defaults: false });

            expect(getComponent("health")).toBeDefined();

            state.step();
            expect(systemRan).toBe(true);
        });
    });
});
