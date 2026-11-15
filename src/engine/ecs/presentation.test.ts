import { expect, test } from "bun:test";
import { component, f32 } from "./component";
import { World } from "./world";

const Display = component("presentation-test-display", { value: f32 }, { presentation: true });
const Simulation = component("presentation-test-simulation", { value: f32 });

test("restore preserves current presentation values and membership only on surviving identities", () => {
    const w = new World();
    w.registry.register(Display);
    w.registry.register(Simulation);
    const kept = w.create();
    const removed = w.create();
    const added = w.create();
    const recreated = w.create();
    const reused = w.create();
    const absent = w.create();
    for (const eid of [kept, removed, recreated, reused, absent]) w.add(eid, Display, { value: 1 });
    w.add(kept, Simulation, { value: 1 });
    const saved = w.snapshot();
    w.storage(Display).value.set(kept, 0.25);
    w.storage(Simulation).value.set(kept, 9);
    w.remove(removed, Display);
    w.add(added, Display, { value: 3 });
    w.destroy(recreated);
    w.destroy(reused);
    expect(w.create()).toBe(reused);
    w.add(reused, Display, { value: 99 });
    const destroyed = w.create();
    expect(destroyed).toBe(recreated);
    w.add(destroyed, Display, { value: 99 });
    const future = w.create();
    w.add(future, Display, { value: 99 });
    w.destroy(absent);
    const query = w.query([Display]);
    const events: [number, boolean][] = [];
    w.observeMembership(Display, (eid, present) => events.push([eid, present]));
    w.restore(saved);
    expect(w.get(kept, Display)?.value).toBe(0.25);
    expect(w.get(kept, Simulation)?.value).toBe(1);
    expect([...query].sort()).toEqual([kept, added].sort());
    expect(events).toEqual([
        [recreated, false],
        [reused, false],
        [future, false],
    ]);
    expect(w.has(absent, Display)).toBe(false);
    expect(w.has(recreated, Display)).toBe(false);
    expect(w.has(reused, Display)).toBe(false);
    expect(w.create()).toBe(future);
    expect(w.has(future, Display)).toBe(false);
    w.restore(saved);
    expect(w.get(kept, Display)?.value).toBe(0.25);
    w.dispose();
});
