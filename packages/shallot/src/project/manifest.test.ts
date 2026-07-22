import { describe, expect, test } from "bun:test";
import { normalize } from "./manifest";

describe("normalize", () => {
    test("parses a valid manifest", () => {
        const m = normalize(`{ "scene": "s.scene", "plugins": { "Orbit": true } }`);
        expect(m.scene).toBe("s.scene");
        expect(m.plugins).toEqual({ Orbit: true });
    });

    test("tolerates corrupt or absent storage", () => {
        expect(normalize(null)).toEqual({});
        expect(normalize("not json")).toEqual({});
        expect(normalize("[1,2,3]")).toEqual({}); // a non-object root
    });

    test("drops a non-string scene and a non-object plugins", () => {
        expect(normalize(`{ "scene": 5, "plugins": [1] }`)).toEqual({});
    });

    test("parses a numeric capacity (the serialized Config field) and drops a non-numeric one", () => {
        expect(normalize(`{ "scene": "s.scene", "capacity": 512 }`).capacity).toBe(512);
        expect(normalize(`{ "capacity": "lots" }`).capacity).toBeUndefined();
    });

    test("preserves $schema (the IDE pointer)", () => {
        const schema = "./node_modules/@dylanebert/shallot/shallot.schema.json";
        expect(normalize(`{ "$schema": "${schema}", "plugins": { "Orbit": true } }`).$schema).toBe(
            schema,
        );
    });
});
