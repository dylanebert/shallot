import { beforeAll, expect, test } from "bun:test";
import { type App, build } from "@dylanebert/shallot";
import { clear, entries, getTraits, readFields } from "@dylanebert/shallot/ecs/core";
import { formatFields, parseFields } from "@dylanebert/shallot/scene/core";
import { SHALLOT_PLUGINS, STANDARD_PLUGINS } from "../plugins";

// the editor's component palette, minus the plugins that can't build on the WSL software adapter:
// Physics / Character / Player warm allocates the persistent contact store (capacity · contacts), which
// exceeds the adapter's storage-binding limit (testing.md "Physics/Character/Player can't build on the
// bun-webgpu adapter"). The catalog derives from the barrel now, so it includes them — filter by name so
// this CPU/schema round-trip stays buildable; their component reflection rides the real-GPU tier.
const GPU_FLOOR = new Set(["Physics", "Character", "Player"]);
const PLUGINS = [...STANDARD_PLUGINS, ...SHALLOT_PLUGINS].filter((p) => !GPU_FLOOR.has(p.name));

let app: App;

beforeAll(async () => {
    clear();
    // build() wires every plugin's component stores. On WSL this binds a software adapter; the
    // round-trip reads the CPU-side slab mirror (the authored value), not GPU output, so it stays
    // hardware-invariant — a bun-test-tier fact, not a real-GPU assertion.
    app = await build({ plugins: PLUGINS, defaults: false });
});
// no dispose: InputPlugin.dispose detaches window listeners that never attached headless (no `window`
// in bun test), and build() starts no frame loop — matching the other build()-based unit tests.

// the structural net under the inspector's reflection→UI path. Every registered component must survive
// `defaults → readFields → formatFields → parseFields` unchanged. readFields throwing (a GPU-resource
// getter, a lane miscount) or a format/parse drift fails here — at the registry level — before it can
// crash the inspector the moment that component is selected.
test("defaults → readFields → formatFields → parseFields is a fixed point for every registered component", () => {
    const state = app.state;
    let checked = 0;

    for (const { name, component } of entries()) {
        const eid = state.create();
        state.add(eid, component as never);

        // mirror ReadbackSystem exactly: it merges trait defaults under the live fields, formats that,
        // and writes the string to the node attr. A scene save/reload then parses it back. The three
        // calls below are the net — readFields throwing (a GPU-resource getter, a lane miscount) or
        // formatFields emitting an unparseable token (the Tween NaN class) fails right here.
        const defaults = getTraits(name)?.defaults?.() ?? {};
        const fields = readFields(component, eid);
        const formatted = formatFields(name, { ...defaults, ...fields });
        const reparsed = parseFields(name, formatted);

        // value fixed point: every field the formatted string carried parses back to the value
        // readFields held (lanes the formatter elided as trailing defaults simply re-default on parse)
        for (const [key, b] of Object.entries(reparsed)) {
            const a = fields[key];
            if (typeof a === "number" && typeof b === "number") {
                // formatNumber serializes non-integers via toPrecision(7); the round-trip error is
                // bounded by a half-ULP at the 7th significant figure
                expect(Math.abs(b - a)).toBeLessThanOrEqual(5e-7 * Math.max(1, Math.abs(a)));
            } else if (typeof a === "string" || typeof a === "number") {
                expect(b).toBe(a);
            }
        }
        checked++;
    }

    expect(checked).toBeGreaterThan(0);
});
