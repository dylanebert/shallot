import { describe, expect, test } from "bun:test";
import { EXAMPLE_GATES } from "./example-gates";
import { selectExampleGates } from "./test-changed";

const dirs = (paths: string[]) => selectExampleGates(paths).map((row) => row.dir);

describe("changed-path example selector", () => {
    test("a recipe file selects exactly its registry row", () => {
        expect(dirs(["examples/recipes/moving-platform/src/plugin.ts"])).toEqual([
            "examples/recipes/moving-platform",
        ]);
    });

    test("an engine subsystem file selects the exact registry cone", () => {
        expect(dirs(["packages/shallot/src/standard/render/plugin.ts"])).toEqual(
            EXAMPLE_GATES.map((row) => row.dir),
        );
    });

    test("bun.lock selects the whole roster", () => {
        expect(dirs(["bun.lock"])).toEqual(EXAMPLE_GATES.map((row) => row.dir));
    });

    test("every package manifest selects the whole roster", () => {
        expect(dirs(["examples/showcase/visualization/package.json"])).toEqual(
            EXAMPLE_GATES.map((row) => row.dir),
        );
    });

    test("docs select nothing", () => {
        expect(dirs(["docs/selector.md"])).toEqual([]);
    });
});
