import { expect, test } from "bun:test";
import { type Resource, World } from "./state";

function callerTypeControl(state: World): void {
    const number: Resource<number> = { create: () => 1 };
    // @ts-expect-error The declaration binds the result type, not the caller.
    const value: string = state.resource<string>(number);
    void value;
}
void callerTypeControl;

test("a declaration creates once per world, including an undefined value", () => {
    const state = new World();
    let calls = 0;
    const declaration: Resource<undefined> = {
        create: (owner) => {
            expect(owner).toBe(state);
            calls++;
            return undefined;
        },
    };
    expect(state.resource(declaration)).toBeUndefined();
    expect(state.resource(declaration)).toBeUndefined();
    expect(calls).toBe(1);
    state.dispose();
});

test("worlds and declarations isolate values even with the same creator", () => {
    const first = new World();
    const second = new World();
    let calls = 0;
    let cleanups = 0;
    const create = (state: World) => {
        calls++;
        state.onDispose(() => cleanups++);
        return {};
    };
    const a = { create };
    const b = { create };
    const value = first.resource(a);
    expect(first.resource(a)).toBe(value);
    expect(second.resource(a)).not.toBe(value);
    expect(first.resource(b)).not.toBe(value);
    expect(calls).toBe(3);
    first.dispose();
    expect(cleanups).toBe(2);
    second.dispose();
    expect(cleanups).toBe(3);
});

test("disposed worlds refuse cached and fresh declarations without invoking creators or caching", () => {
    const state = new World();
    let calls = 0;
    const create = () => ++calls;
    const cached = { create };
    const fresh = { create };
    state.resource(cached);
    state.dispose();
    for (const declaration of [cached, fresh, fresh]) {
        expect(() => state.resource(declaration)).toThrow("world is disposed");
    }
    expect(calls).toBe(1);
    // Inspect only the owner's cache, not caller-retained values.
    expect((state as unknown as { _resources: Map<unknown, unknown> })._resources.size).toBe(0);
});
