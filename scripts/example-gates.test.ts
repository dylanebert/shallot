import { expect, test } from "bun:test";
import { readdirSync } from "node:fs";
import { resolve } from "node:path";
import { EXAMPLE_GATES } from "./example-gates";
import { selectExampleGates } from "./test-changed";

// The selection ratchet. Every example row used to hardcode `packages/shallot-runtime/src/**`, so one
// runtime edit — a boot splash, a fog setting — selected all 39 headed rows, the whole scenario sweep and
// a Playwright suite. Rows now declare the modules their surviving check claims about, and this file is
// what keeps that true: the ceiling below is the measured maximum, and it only ever moves DOWN. Raising it
// is the regression this test exists to catch, so a change that needs a higher number is a change that
// re-widened a cone.

const RUNTIME_SRC = "packages/shallot-runtime/src";

/** measured 2026-09-08 over the whole runtime source tree: `standard/avbd` and `standard/physics`, the
 *  cone the seven physics recipes plus the collapse and sandbox showcases all claim about. Lower it when
 *  a cone narrows; never raise it. */
const MAX_ROWS_PER_RUNTIME_FILE = 9;

function runtimeFiles(): string[] {
    const root = resolve(import.meta.dir, "..", RUNTIME_SRC);
    return readdirSync(root, { recursive: true, withFileTypes: true })
        .filter((entry) => entry.isFile() && entry.name.endsWith(".ts"))
        .map((entry) =>
            `${RUNTIME_SRC}/${resolve(entry.parentPath, entry.name).slice(root.length + 1)}`
                .split("\\")
                .join("/"),
        );
}

test("no single runtime file selects more example rows than the ratchet allows", () => {
    const files = runtimeFiles();
    // the population itself is pinned: a narrowed walk would satisfy the ceiling by seeing nothing.
    expect(files.length).toBeGreaterThan(250);
    const worst = files
        .map((file) => ({ file, rows: selectExampleGates([file]).length }))
        .sort((a, b) => b.rows - a.rows);
    expect({
        file: worst[0].file,
        overCeiling: worst[0].rows > MAX_ROWS_PER_RUNTIME_FILE,
    }).toEqual({ file: worst[0].file, overCeiling: false });
    // and the ceiling is not slack: something must actually reach it, or it has stopped ratcheting.
    expect(worst[0].rows).toBe(MAX_ROWS_PER_RUNTIME_FILE);
});

test("a runtime module no example claims selects nothing", () => {
    // `standard/fog` is the census's own case: changing fog moved all 39 rows and no example asserts
    // anything about it. A row that starts claiming fog must declare the cone, not inherit it.
    expect(selectExampleGates([`${RUNTIME_SRC}/standard/fog/index.ts`])).toEqual([]);
});

test("every row's cone starts with its own directory", () => {
    for (const row of EXAMPLE_GATES)
        expect({ dir: row.dir, own: row.covers[0] }).toEqual({
            dir: row.dir,
            own: `${row.dir}/**`,
        });
});

test("no row inherits a blanket runtime cone", () => {
    // The exact globs the pre-cut roster appended to every row. A physics recipe may still declare
    // `packages/shallot-tumble/src/**`, because the tumble solver IS what its check reads through — the
    // defect was the blanket, not the path.
    const blanket = [`${RUNTIME_SRC}/**`, "packages/shallot-runtime/**"];
    for (const row of EXAMPLE_GATES)
        for (const cover of row.covers)
            expect({ dir: row.dir, cover, blanket: blanket.includes(cover) }).toEqual({
                dir: row.dir,
                cover,
                blanket: false,
            });
});
