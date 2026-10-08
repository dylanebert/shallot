import { expect, test } from "bun:test";
import { type Resource, World } from "./world";

function callerTypeControl(world: World): void {
    const number: Resource<number> = { create: () => 1 };
    // @ts-expect-error The declaration binds the result type, not the caller.
    const value: string = world.resource<string>(number);
    void value;
}
void callerTypeControl;

test("a declaration creates once per world, including an undefined value", () => {
    const world = new World();
    let calls = 0;
    const declaration: Resource<undefined> = {
        create: (owner) => {
            expect(owner).toBe(world);
            calls++;
            return undefined;
        },
    };
    expect(world.resource(declaration)).toBeUndefined();
    expect(world.resource(declaration)).toBeUndefined();
    expect(calls).toBe(1);
    world.dispose();
});

test("worlds and declarations isolate values even with the same creator", () => {
    const first = new World();
    const second = new World();
    let calls = 0;
    let cleanups = 0;
    const create = (world: World) => {
        calls++;
        world.onDispose(() => cleanups++);
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

test("an explicit resource key carries a value across declarations but not across worlds", () => {
    const first = new World();
    const second = new World();
    const key = Symbol("reload-stable");
    let creates = 0;
    let cleanups = 0;
    const old: Resource<object> = {
        key,
        create(world) {
            creates++;
            world.onDispose(() => cleanups++);
            return {};
        },
    };
    const next: Resource<object> = {
        key,
        create() {
            throw new Error("carried resources do not reinitialize");
        },
    };
    const value = first.resource(old);
    expect(first.resource(next)).toBe(value);
    expect(second.resource(old)).not.toBe(value);
    expect(first.resource({ ...old, key: Symbol("reload-stable") })).not.toBe(value);
    expect(creates).toBe(3);
    first.dispose();
    expect(cleanups).toBe(2);
    second.dispose();
    expect(cleanups).toBe(3);
    expect(() => first.resource(next)).toThrow("world is disposed");
});

test("disposed worlds refuse cached and fresh declarations without invoking creators or caching", () => {
    const world = new World();
    let calls = 0;
    const create = () => ++calls;
    const cached = { create };
    const fresh = { create };
    world.resource(cached);
    world.dispose();
    for (const declaration of [cached, fresh, fresh]) {
        expect(() => world.resource(declaration)).toThrow("world is disposed");
    }
    expect(calls).toBe(1);
    // Inspect only the owner's cache, not caller-retained values.
    expect((world as unknown as { _resources: Map<unknown, unknown> })._resources.size).toBe(0);
});
