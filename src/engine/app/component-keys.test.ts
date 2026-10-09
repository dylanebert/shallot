import { expect, test } from "bun:test";
import { component, f32 } from "../ecs";
import { resolvePlugins } from "./compose";

test("different empty records under one key are refused, naming both plugins", () => {
    const Audio = { name: "Audio", components: [component("SharedTag", {})] };
    const Game = { name: "Game", components: [component("SharedTag", {})] };
    expect(() => resolvePlugins([Audio, Game])).toThrow(
        'component "SharedTag" is declared by plugin "Audio" and by another record in plugin "Game"; give one of them its own key',
    );
});

test("different field records under one key are refused", () => {
    const Audio = { name: "Audio", components: [component("SharedValue", { x: f32 })] };
    const Game = { name: "Game", components: [component("SharedValue", { y: f32 })] };
    expect(() => resolvePlugins([Audio, Game])).toThrow(
        'component "SharedValue" is declared by plugin "Audio" and by another record in plugin "Game"',
    );
});

test("one record listed by two plugins composes", () => {
    const Shared = component("SharedRecord", {});
    expect(() =>
        resolvePlugins([
            { name: "Owner", components: [Shared] },
            { name: "Includer", components: [Shared] },
        ]),
    ).not.toThrow();
});

test("different plugin objects with one name are refused", () => {
    expect(() => resolvePlugins([{ name: "Dup" }, { name: "Dup" }])).toThrow(
        'plugin "Dup" is listed more than once; give one of them its own name',
    );
});

test("plugin names are checked before component declarations", () => {
    expect(() => resolvePlugins([{ name: "Dup", components: [{}] }, { name: "Dup" }])).toThrow(
        'plugin "Dup" is listed more than once',
    );
});

test("one plugin reached both through a dependency and repeated input composes", () => {
    const Shared = { name: "Shared" };
    const Root = { name: "Root", dependencies: [Shared] };
    expect(() => resolvePlugins([Root, Shared, Shared])).not.toThrow();
});
