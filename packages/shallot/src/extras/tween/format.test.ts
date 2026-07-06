import { expect, test } from "bun:test";

// The scene formatter (`scripts/format.ts`) registers components but never runs plugin
// `initialize` — so the tween intern table is only what the module seeds at load. An authored
// `field` must survive `normalizeAttr`'s strip-defaults pass through that path.
//
// Regression: id 0 is the `Tween.field` default (the empty-path sentinel). Before the module
// seeded id 0 = "" at load, the first real path (`transform.pos.y`) took id 0, collided with the
// default, and the formatter stripped `field` off every tween — freezing the showcase. This runs
// in a fresh interpreter (no `build()`/`initialize` anywhere) to reproduce the formatter exactly;
// a same-process test is masked the moment any prior `build()` has seeded the table.
test("the scene formatter preserves an authored tween field", () => {
    const src = [
        `import { setupGlobals } from "bun-webgpu";`,
        `await setupGlobals();`,
        `const { normalizeAttr } = await import(${JSON.stringify(`${import.meta.dir}/../../engine/scene/core.ts`)});`,
        `const { register } = await import(${JSON.stringify(`${import.meta.dir}/../../engine/ecs/core.ts`)});`,
        `const { Tween, TweenPlugin } = await import(${JSON.stringify(`${import.meta.dir}/index.ts`)});`,
        `register("tween", Tween, TweenPlugin.traits.Tween);`,
        `process.stdout.write(normalizeAttr("tween", "field: transform.pos.y; to: 5") ?? "");`,
    ].join("\n");

    const out = Bun.spawnSync(["bun", "-e", src]).stdout.toString();
    expect(out).toContain("field: transform.pos.y");
});
