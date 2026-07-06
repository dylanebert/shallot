import { describe, expect, test } from "bun:test";
import { resolve } from "node:path";
import {
    enumOptions,
    findDefinitionLine,
    isAsConst,
    objectLiteral,
    paramNames,
    parseClassMembers,
    parseComponentFields,
    parseInterfaceFields,
    pluginRefs,
} from "./jsdoc";

const ORBIT = resolve(import.meta.dir, "../src/extras/orbit/index.ts");
const SPRITE = resolve(import.meta.dir, "../src/extras/sprite/pack.ts");
const GLTF = resolve(import.meta.dir, "../src/extras/gltf/assets.ts");
const INPUT = resolve(import.meta.dir, "../src/standard/input/index.ts");
const STATE = resolve(import.meta.dir, "../src/engine/ecs/state.ts");
const APP = resolve(import.meta.dir, "../src/engine/app/index.ts");

describe("isAsConst — the enum marker across the whole declaration", () => {
    test("single-line enum: the marker is on the def line", () => {
        expect(isAsConst(ORBIT, "OrbitMode")).toBe(true);
    });

    test("multi-line enum: the marker is on the object literal's closing line", () => {
        expect(isAsConst(SPRITE, "SpriteBillboard")).toBe(true);
    });

    test("a plain object literal (a component) is not an enum", () => {
        expect(isAsConst(SPRITE, "Sprite")).toBe(false);
    });

    test("a non-object export is not an enum", () => {
        expect(isAsConst(SPRITE, "SPRITE_BYTES")).toBe(false);
    });
});

describe("enumOptions — option/value pairs from an enum's literal", () => {
    test("parses numeric and string options, ignoring whitespace", () => {
        expect(enumOptions("{ Free: 0, Locked: 1 }")).toEqual([
            ["Free", "0"],
            ["Locked", "1"],
        ]);
        expect(enumOptions('{ Alpha: "a", Beta: "b" }')).toEqual([
            ["Alpha", '"a"'],
            ["Beta", '"b"'],
        ]);
    });

    test("OrbitMode reads from the real source via objectLiteral", () => {
        expect(enumOptions(objectLiteral(ORBIT, "OrbitMode"))).toEqual([
            ["Free", "0"],
            ["Locked", "1"],
        ]);
    });
});

describe("pluginRefs — the components/systems/dependencies a plugin bundles", () => {
    test("extracts identifiers from each section, taking the key of a `Name: Value` component", () => {
        const text = `{ name: "X", systems: [ASystem, BSystem], components: { Foo: Foo, Bar }, dependencies: [DepPlugin] }`;
        expect(pluginRefs(text)).toEqual({
            components: ["Foo", "Bar"],
            systems: ["ASystem", "BSystem"],
            dependencies: ["DepPlugin"],
        });
    });

    test("OrbitPlugin reads from the real source", () => {
        const refs = pluginRefs(objectLiteral(ORBIT, "OrbitPlugin"));
        expect(refs.components).toEqual(["Orbit"]);
        expect(refs.systems).toEqual(["OrbitSystem"]);
        expect(refs.dependencies).toEqual(["InputPlugin", "TransformsPlugin"]);
    });

    test("a missing section is an empty list, not a throw", () => {
        expect(pluginRefs("{ name: 'X' }")).toEqual({
            components: [],
            systems: [],
            dependencies: [],
        });
    });
});

describe("parseComponentFields — per-field descriptions from the field's doc comment", () => {
    test("orbit: every field carries its annotation, keyed by the declared name", () => {
        const docs = parseComponentFields(ORBIT, "Orbit");
        // exactly the Orbit schema — the brace-walk stops at its closing `}`, never spilling into the
        // code that follows it in the file.
        expect(Object.keys(docs).sort()).toEqual(
            [
                "yaw",
                "pitch",
                "distance",
                "size",
                "minPitch",
                "maxPitch",
                "minDistance",
                "maxDistance",
                "minSize",
                "maxSize",
                "smoothness",
                "flySmoothness",
                "sensitivity",
                "flySensitivity",
                "zoomSpeed",
                "orbitButton",
                "panButton",
                "flyButton",
                "pan",
                "flySpeed",
                "flyBoost",
                "flyMin",
                "flyMax",
                "mode",
                "target",
            ].sort(),
        );
    });

    test("the description is the comment's first line, verbatim", () => {
        const docs = parseComponentFields(ORBIT, "Orbit");
        expect(docs.yaw).toBe("horizontal orbit angle around the target, radians");
        expect(docs.orbitButton).toBe("mouse button that orbits: 0 left, 1 middle, 2 right");
    });

    test("a missing file is empty, not a throw", () => {
        expect(parseComponentFields(resolve(import.meta.dir, "nope.ts"), "Orbit")).toEqual({});
    });

    test("an unknown component in a real file is empty", () => {
        expect(parseComponentFields(ORBIT, "Nonexistent")).toEqual({});
    });
});

describe("paramNames — top-level parameter names from a signature", () => {
    const at = (file: string, name: string) =>
        paramNames(file, findDefinitionLine(file, name), name);

    test("splits on top-level commas only — a tuple/object-type param stays one param", () => {
        // placeGltf's `opts` is `{ pos?: [number, number, number]; … }`; its inner commas must not shred
        // the param list (the bug that mis-rendered the reference signature)
        expect(at(GLTF, "placeGltf")).toBe("state, handle, opts");
    });

    test("a plain multi-arg signature keeps each name", () => {
        expect(at(GLTF, "placeScene")).toBe("state, asset");
    });

    test("a no-arg call is an empty string", () => {
        expect(at(GLTF, "gltfCacheStats")).toBe("");
    });

    test("a missing file is null", () => {
        expect(paramNames(resolve(import.meta.dir, "nope.ts"), 1, "x")).toBeNull();
    });
});

describe("parseInterfaceFields — an @expand interface's fields and methods", () => {
    test("Inputs: both readonly property fields and method signatures, methods carrying params", () => {
        const members = parseInterfaceFields(INPUT, "Inputs");
        const byName = new Map(members.map((m) => [m.name, m]));
        expect([...byName.keys()].sort()).toEqual(
            [
                "mouse",
                "focused",
                "isKeyDown",
                "isKeyPressed",
                "isKeyReleased",
                "isKeyPressedWithin",
            ].sort(),
        );
        expect(byName.get("mouse")?.kind).toBe("field");
        expect(byName.get("isKeyDown")?.kind).toBe("method");
        expect(byName.get("isKeyDown")?.params).toBe("code");
        expect(byName.get("isKeyPressedWithin")?.params).toBe("code, seconds");
    });

    test("an optional method is a method, not dropped (`error?(...)`)", () => {
        const byName = new Map(parseInterfaceFields(APP, "Loading").map((m) => [m.name, m]));
        expect([...byName.keys()].sort()).toEqual(["error", "show", "update"]);
        expect(byName.get("error")?.kind).toBe("method");
        expect(byName.get("error")?.params).toBe("error");
    });

    test("a missing interface is empty, not a throw", () => {
        expect(parseInterfaceFields(INPUT, "Nonexistent")).toEqual([]);
    });
});

describe("parseClassMembers — a @expand class's members, interfaces excluded", () => {
    test("an interface of the same name bails (parseInterfaceFields owns it)", () => {
        // `Inputs` is both an interface and a const singleton; the class parser must not claim it and
        // strip its `readonly` property rows — the interface path renders those.
        expect(parseClassMembers(INPUT, "Inputs")).toEqual([]);
    });

    test("a real class still parses its methods", () => {
        const names = new Set(parseClassMembers(STATE, "State").map((m) => m.name));
        expect(names.has("query")).toBe(true);
        expect(names.has("create")).toBe(true);
    });
});
