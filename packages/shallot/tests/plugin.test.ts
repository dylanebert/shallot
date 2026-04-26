import { test, expect, describe } from "bun:test";
import { build, relation, type Plugin, type System } from "../src";
import { getComponent } from "../src/engine/ecs/component";

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

            const state = await build({ plugins: [plugin], defaults: false });
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

            const state = await build({ plugins: [plugin], defaults: false });
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

            const state = await build({ plugins: [pluginA, pluginB], defaults: false });
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

            expect(Object.keys(plugin.components!)).toHaveLength(1);
            expect(getComponent("test-comp")).toBeDefined();
        });

        test("should include relations in plugin definition", async () => {
            const TestRelation = relation("plugin-test-rel", { exclusive: true });

            const plugin: Plugin = {
                name: "TestRelation",
                relations: [TestRelation],
            };

            await build({ plugins: [plugin], defaults: false });

            expect(plugin.relations).toHaveLength(1);
            expect(plugin.relations![0].name).toBe("plugin-test-rel");
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

    describe("plugin with full bundle", () => {
        test("should create a complete plugin with systems, components, and relations", async () => {
            let systemRan = false;

            const Health = { current: [] as number[], max: [] as number[] };
            const OwnerRelation = relation("plugin-owner", { exclusive: true });
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
                relations: [OwnerRelation],
            };

            const state = await build({ plugins: [HealthPlugin], defaults: false });

            expect(HealthPlugin.systems).toHaveLength(1);
            expect(Object.keys(HealthPlugin.components!)).toHaveLength(1);
            expect(HealthPlugin.relations).toHaveLength(1);

            state.step();
            expect(systemRan).toBe(true);
        });
    });
});
