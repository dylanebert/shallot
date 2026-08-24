// Tree-wide static arm for S6 item 3 (`shallot-boot-noise.md`): four in-repo `examples/showcase/**`
// forcers still zero-dispatched on a deploying site while `AGENTS.md`/`MIGRATION.md` forbid it — the
// Goal's own symptom, still live in a stranger's console, because S1 enumerated `packages/shallot/src`
// alone. This is the load-bearing half of criterion 6/7's re-opened oracle: criterion 2's only witness
// (`bin/console-warning.probes.ts`) runs under no gate and no CI, so nothing today catches a
// reintroduced zero-workgroup dispatch or a zero-count `.draw`. `prebuilt.test.ts`'s static arms are
// the precedent for a repo-shape assertion living beside the code it checks rather than inside a
// script nobody runs.
//
// Scanned **dynamically** by glob over the whole repo tree (the same `consumerDirs` shape
// `check-exports.ts` uses for its own cross-cutting walk) — never a fixed enumeration of "the four
// known sites". A fixed list is satisfiable by deletion: shrink the list (or the files it names) and
// the arm passes having checked nothing new. A glob over `**/*.ts` catches a fifth site anywhere in
// the scanned tree the moment it's written, which `findZeroDispatches`'s own fixture-driven tests
// below prove by construction (they never edit real source to produce a red).
import { describe, expect, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, relative, resolve } from "node:path";
import { Glob } from "bun";

const REPO_ROOT = resolve(import.meta.dir, "../../..");

// Mirrors `check-exports.ts`'s `consumerDirs`: every tree a shipped or example forcer can live in.
// `packages/shallot/tests` is included too — a fixture accidentally left in real source under here
// would still be caught.
const SCAN_DIRS = [
    "packages/shallot/src",
    "packages/shallot/bin",
    "packages/shallot/scripts",
    "packages/shallot/tests",
    "scripts",
    "examples",
    "evals",
];

// Built by concatenation, never as a literal, so this file itself is never a false positive when the
// scan root is the real repo (this file lives under `packages/shallot/tests`, one of `SCAN_DIRS`).
const ZERO_DISPATCH = `dispatchWorkgroups${"("}0${")"}`;
const ZERO_DRAW = `.draw${"("}0${")"}`;
// The three widened shapes, same concatenation discipline: a float-literal zero, and the two
// multi-argument forms whose match text stops at the first comma (the regex's own boundary).
const ZERO_DISPATCH_FLOAT = `dispatchWorkgroups${"("}0.0${")"}`;
const ZERO_DISPATCH_MULTI = `dispatchWorkgroups${"("}0${", y, z)"}`;
const ZERO_DISPATCH_MULTI_MATCH = `dispatchWorkgroups${"("}0${","}`;
const ZERO_DRAW_MULTI = `.draw${"("}0${", 1)"}`;
const ZERO_DRAW_MULTI_MATCH = `.draw${"("}0${","}`;

export interface ZeroDispatchViolation {
    file: string;
    match: string;
}

/** scan every `.ts` file under `root`'s `SCAN_DIRS` for a zero-workgroup dispatch or a zero-count
 *  `.draw` call — the two call shapes Dawn's `EmitWarningOnce` families warn on
 *  (`ComputePassEncoder::APIDispatchWorkgroups`, its draw-side siblings). `node_modules` and any
 *  `dist/` build output are skipped — this checks source, not a bundle a build step already baked.
 *  The zero it matches is the CALL'S FIRST ARGUMENT (workgroup count / vertex count), not the whole
 *  argument list — a multi-argument `dispatchWorkgroups` with a literal zero x, or a multi-argument
 *  `.draw` with a literal zero vertex count, warns exactly like the single-arg form because the first
 *  argument is what Dawn checks, so a literal `0` there trips this regardless of what follows. The literal itself may be an integer (`0`) or a float zero (`0.0`, `.0`) —
 *  whitespace around it still matches. The literal must be exactly `0`, `0.0` or `.0` sitting directly
 *  before the comma or close paren, so a real dispatch/draw sized off a runtime count
 *  (`dispatchWorkgroups(count)`, `.draw(vertexCount)`, or a multi-arg call with no zero literal in the
 *  first slot) never trips it. Known blind spots of a lexical scan, none of them present in-tree and
 *  each verified against this regex rather than assumed: `00`, `0x0`, `-0`, a comment interposed
 *  between the literal and its terminator (`dispatchWorkgroups(0 /* c *\/, y)`), and any zero reached
 *  through a name (`const ZERO = 0`). Only the last needs an AST pass to close, and no live instance
 *  of any of them justifies one today — so read a green here as "no literal zero-count call", never as
 *  "no zero-count call reachable". @internal */
export async function findZeroDispatches(root: string): Promise<ZeroDispatchViolation[]> {
    const violations: ZeroDispatchViolation[] = [];
    const zeroLiteral = String.raw`(?:0(?:\.0+)?|\.0+)`;
    const pattern = new RegExp(
        String.raw`\bdispatchWorkgroups\(\s*${zeroLiteral}\s*[,)]|\.draw\(\s*${zeroLiteral}\s*[,)]`,
        "g",
    );
    for (const dir of SCAN_DIRS) {
        const full = resolve(root, dir);
        if (!existsSync(full)) continue;
        const glob = new Glob("**/*.ts");
        for await (const path of glob.scan({ cwd: full })) {
            if (path.includes("node_modules") || path.includes("dist/")) continue;
            const fullPath = resolve(full, path);
            const content = await Bun.file(fullPath).text();
            const matches = content.match(pattern);
            if (!matches) continue;
            const relPath = relative(root, fullPath).replace(/\\/g, "/");
            for (const match of matches) violations.push({ file: relPath, match });
        }
    }
    return violations;
}

describe("no zero-workgroup dispatch or zero-count draw survives in the shipped tree", () => {
    // THE gate: the actual repo, scanned for real. Green today because S6 converted the four
    // `examples/showcase/**` forcers close's architectural pass found still dispatching
    // (`voxel/mesher.ts`, `voxel/generate.ts`, `roads/posts.ts`, `roads/terrain/generate.ts`) — a
    // fifth site anywhere under `SCAN_DIRS` reds this the moment it's written, source-tree wide.
    test("packages/shallot/src, examples/**, scripts/, bin/, and evals/ carry none", async () => {
        const violations = await findZeroDispatches(REPO_ROOT);
        expect(violations).toEqual([]);
    });

    // Non-vacuous witness (a): a fixture file carrying the exact forbidden dispatch shape, in a
    // scratch root under the OS tmpdir — never real source — proves the scanner reports a real
    // violation rather than one no input could ever produce.
    test("a fixture zero-workgroup dispatch is caught (not deletable-by-narrowing)", async () => {
        const root = mkdtempSync(join(tmpdir(), "no-zero-dispatch-"));
        try {
            const srcDir = join(root, "packages/shallot/src/standard/rogue");
            mkdirSync(srcDir, { recursive: true });
            writeFileSync(
                join(srcDir, "rogue.ts"),
                `export function force() {\n    pipeline.with(group).with(pass).${ZERO_DISPATCH};\n    return pipeline;\n}\n`,
            );
            const violations = await findZeroDispatches(root);
            expect(violations).toHaveLength(1);
            expect(violations[0].file).toBe("packages/shallot/src/standard/rogue/rogue.ts");
            expect(violations[0].match).toBe(ZERO_DISPATCH);
        } finally {
            rmSync(root, { recursive: true, force: true });
        }
    });

    // Non-vacuous witness (b): same proof for the render-side sibling — a zero-count draw call — in a
    // directory outside `packages/shallot` (`examples/`) — the scan is tree-wide, not `gpu.ts`-only.
    test("a fixture zero-count draw is caught, anywhere under the scanned tree", async () => {
        const root = mkdtempSync(join(tmpdir(), "no-zero-draw-"));
        try {
            const srcDir = join(root, "examples/showcase/rogue/src");
            mkdirSync(srcDir, { recursive: true });
            writeFileSync(
                join(srcDir, "rogue.ts"),
                `export function force() {\n    pass${ZERO_DRAW};\n    return pipeline;\n}\n`,
            );
            const violations = await findZeroDispatches(root);
            expect(violations).toHaveLength(1);
            expect(violations[0].file).toBe("examples/showcase/rogue/src/rogue.ts");
            expect(violations[0].match).toBe(ZERO_DRAW);
        } finally {
            rmSync(root, { recursive: true, force: true });
        }
    });

    // Non-vacuous witness (c): a float-literal zero (`0.0`) is the same warning-triggering shape as
    // the bare integer `0` — Dawn's check reads the workgroup count's value, not its literal syntax —
    // and the old regex's bare-digit `0` only missed this by accident of not handling a decimal point.
    test("a fixture float-literal zero dispatch is caught", async () => {
        const root = mkdtempSync(join(tmpdir(), "no-zero-dispatch-float-"));
        try {
            const srcDir = join(root, "packages/shallot/src/standard/float-zero");
            mkdirSync(srcDir, { recursive: true });
            writeFileSync(
                join(srcDir, "float-zero.ts"),
                `export function force() {\n    pipeline.with(group).with(pass).${ZERO_DISPATCH_FLOAT};\n    return pipeline;\n}\n`,
            );
            const violations = await findZeroDispatches(root);
            expect(violations).toHaveLength(1);
            expect(violations[0].file).toBe(
                "packages/shallot/src/standard/float-zero/float-zero.ts",
            );
            expect(violations[0].match).toBe(ZERO_DISPATCH_FLOAT);
        } finally {
            rmSync(root, { recursive: true, force: true });
        }
    });

    // Non-vacuous witness (d): a multi-argument dispatch with a literal `0` in the x slot is zero
    // workgroups regardless of y/z — the bare single-arg regex missed this class entirely.
    test("a fixture multi-argument dispatch with x=0 is caught", async () => {
        const root = mkdtempSync(join(tmpdir(), "no-zero-dispatch-multiarg-"));
        try {
            const srcDir = join(root, "packages/shallot/src/standard/multiarg-zero");
            mkdirSync(srcDir, { recursive: true });
            writeFileSync(
                join(srcDir, "multiarg-zero.ts"),
                `export function force() {\n    pipeline.with(group).with(pass).${ZERO_DISPATCH_MULTI};\n    return pipeline;\n}\n`,
            );
            const violations = await findZeroDispatches(root);
            expect(violations).toHaveLength(1);
            expect(violations[0].file).toBe(
                "packages/shallot/src/standard/multiarg-zero/multiarg-zero.ts",
            );
            expect(violations[0].match).toBe(ZERO_DISPATCH_MULTI_MATCH);
        } finally {
            rmSync(root, { recursive: true, force: true });
        }
    });

    // Non-vacuous witness (e): a multi-argument `.draw` with vertexCount 0 is zero draws regardless
    // of the instance count that follows, the render-side twin of witness (d).
    test("a fixture multi-argument draw with vertexCount=0 is caught", async () => {
        const root = mkdtempSync(join(tmpdir(), "no-zero-draw-multiarg-"));
        try {
            const srcDir = join(root, "examples/showcase/rogue-multiarg/src");
            mkdirSync(srcDir, { recursive: true });
            writeFileSync(
                join(srcDir, "rogue-multiarg.ts"),
                `export function force() {\n    pass${ZERO_DRAW_MULTI};\n    return pipeline;\n}\n`,
            );
            const violations = await findZeroDispatches(root);
            expect(violations).toHaveLength(1);
            expect(violations[0].file).toBe(
                "examples/showcase/rogue-multiarg/src/rogue-multiarg.ts",
            );
            expect(violations[0].match).toBe(ZERO_DRAW_MULTI_MATCH);
        } finally {
            rmSync(root, { recursive: true, force: true });
        }
    });

    // A real, non-zero dispatch/draw (the shape every legitimate per-frame call takes) must never
    // trip the arm — otherwise the check would be satisfiable only by having no dispatches at all,
    // which is not the property being asserted. Includes the multi-arg shape with no zero literal in
    // the first slot, the negative twin of witnesses (d) and (e).
    test("a non-zero dispatch or draw count never trips the arm", async () => {
        const root = mkdtempSync(join(tmpdir(), "no-zero-dispatch-real-"));
        try {
            const srcDir = join(root, "packages/shallot/src/standard/real");
            mkdirSync(srcDir, { recursive: true });
            writeFileSync(
                join(srcDir, "real.ts"),
                "export function frame() {\n" +
                    "    pipeline.with(group).with(pass).dispatchWorkgroups(count);\n" +
                    "    pass.draw(vertexCount);\n" +
                    "    pipeline.with(group).with(pass).dispatchWorkgroups(x, y, z);\n" +
                    "}\n",
            );
            const violations = await findZeroDispatches(root);
            expect(violations).toEqual([]);
        } finally {
            rmSync(root, { recursive: true, force: true });
        }
    });
});
