// check-site.ts clause 0 — a stale built artifact reads as a stale-refusal, not as a defect
//
// The founding red: a local `out/site/` built before the title fix at `bf1c25f`, never rebuilt,
// which clause 6 read as a present-tense scratch-shaped-<title> defect on a tree where the fix
// was already in — and a roadmap item was written from it. The fixture below is that artifact:
// demo pages clearing clauses 4/5 (relative paths, full RUM injection) and carrying the pre-fix
// scratch-shaped title, so clause 6 is armed and reds the moment it is reached.
//
// Both directions over the same fixture bytes — only the stamp differs:
//   - no/mismatched stamp: refuses as stale, clause 6 never speaks (exit 0 by default; exit 1
//     naming staleness, not the title, under SITE_OUT_REQUIRED=1 where the build just ran);
//   - stamp matching the current sources: clause 6 reaches the same pages and reds on the title.
// The second leg is what keeps the first from being a fail-open: a refusal that never lifts
// would green a real title defect too.
//
// Red-first witnessed 2026-08-27 by deleting clause 0 from `scripts/check-site.ts` and re-running
// this file: 4 pass / 2 fail, both failures reading the founding false red verbatim —
// `✗ 5 built demo page(s) carry a non-human-readable <title>` over the pre-fix fixture — while the
// fresh-artifact leg stayed green, so the red is the refusal going missing and not the fixture or
// the clause-6 mechanism going missing. The same mutation over the real local `out/site/` (built
// before the stamp existed) is the founding artifact itself: exit 0 with a stale-refusal note by
// default, exit 1 naming staleness under `SITE_OUT_REQUIRED=1`.

import { expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { ROSTER } from "../site/roster";
import {
    demoFingerprints,
    readStamp,
    type SiteMode,
    staleDemos,
    writeStamp,
} from "../site/site-stamp";
import { datadogInitSnippet, nonWorkspaceShallotDependencies } from "./build-site";

const repoRoot = resolve(import.meta.dir, "..");
const checkSite = resolve(repoRoot, "scripts/check-site.ts");

// the real repo's release version — the fixtures below stamp `prod` mode with it so clause 2's
// mode-branched pin check (new in the staging build mode) reads a matching version rather than
// failing ahead of the clause each fixture actually exercises.
const releaseVersion = (
    JSON.parse(readFileSync(resolve(repoRoot, "packages/shallot/package.json"), "utf8")) as {
        version: string;
    }
).version;
const PROD_MODE: SiteMode = { kind: "prod", version: releaseVersion };
const STAGING_MODE: SiteMode = { kind: "staging", pin: "file:/tmp/dylanebert-shallot-0.0.0.tgz" };

test("check-site clause 2 — rejects a fixed workspace extension version", () => {
    expect(
        nonWorkspaceShallotDependencies({
            dependencies: {
                "@dylanebert/shallot": "workspace:*",
                "@dylanebert/shallot-wave": "0.1.0",
            },
        }),
    ).toEqual([["@dylanebert/shallot-wave", "0.1.0"]]);
    expect(
        nonWorkspaceShallotDependencies({
            dependencies: { "@dylanebert/shallot-wave": "workspace:*" },
        }),
    ).toEqual([]);
});

/** A built-site fixture that clears clauses 4 and 5 and fails clause 6 — every demo root page
 * carries the pre-fix scratch-shaped <title> the site build used to synthesize. */
function preFixFixture(): string {
    const out = mkdtempSync(join(tmpdir(), "check-site-fixture-"));
    for (const { slug } of ROSTER) {
        mkdirSync(resolve(out, slug), { recursive: true });
        writeFileSync(
            resolve(out, slug, "index.html"),
            `<!doctype html>
<html lang="en">
    <head>
        <title>shallot-site-${slug}-1756000000000</title>
        <link rel="stylesheet" href="./assets/index.css" />
    </head>
    <body>
        <script type="module" src="./assets/index.js"></script>
${datadogInitSnippet()}    </body>
</html>
`,
        );
    }
    // the site index carries no RUM injection (clause 5's index leg)
    writeFileSync(resolve(out, "index.html"), `<!doctype html>\n<html><body></body></html>\n`);
    return out;
}

function runCheck(outDir: string, env: Record<string, string> = {}) {
    const proc = Bun.spawnSync(["bun", checkSite], {
        cwd: repoRoot,
        env: { ...process.env, SITE_OUT_DIR: outDir, ...env },
    });
    return {
        exitCode: proc.exitCode,
        out: proc.stdout.toString() + proc.stderr.toString(),
    };
}

test("check-site — an unstamped built artifact refuses as stale, not as a title defect", () => {
    const fixture = preFixFixture();
    try {
        const { exitCode, out } = runCheck(fixture);
        expect(out).toContain("stale");
        expect(out).not.toContain("non-human-readable");
        expect(exitCode).toBe(0);
    } finally {
        rmSync(fixture, { recursive: true, force: true });
    }
});

test("check-site — a stale artifact on the deploy path reds on staleness, not on the title", () => {
    const fixture = preFixFixture();
    try {
        // a stamp naming fingerprints that are not this tree's — the artifact is a build of some
        // other sources, which is exactly the founding defect's state
        writeStamp(
            fixture,
            Object.fromEntries(ROSTER.map(({ slug }) => [slug, "0".repeat(32)])),
            PROD_MODE,
        );
        const { exitCode, out } = runCheck(fixture, { SITE_OUT_REQUIRED: "1" });
        expect(out).toContain("stale");
        expect(out).not.toContain("non-human-readable");
        expect(exitCode).toBe(1);
    } finally {
        rmSync(fixture, { recursive: true, force: true });
    }
});

test("check-site — a fresh artifact is judged: clause 6 reds on the pre-fix title", () => {
    const fixture = preFixFixture();
    try {
        writeStamp(
            fixture,
            demoFingerprints(
                repoRoot,
                ROSTER.map((d) => d.slug),
            ),
            PROD_MODE,
        );
        const { exitCode, out } = runCheck(fixture);
        expect(out).toContain("non-human-readable");
        expect(exitCode).toBe(1);
    } finally {
        rmSync(fixture, { recursive: true, force: true });
    }
});

// --- the fingerprint's own moves, over a hermetic git tree -------------------------------
//
// `demoFingerprints` scopes each demo to what git tracks, so the subject needs a real checkout.
// A fixture repo (never this one — an arm that edits tracked files to observe a hash leaves the
// working tree as its residue) carries one demo plus the builder files the recipe reads.

function fixtureRepo(): string {
    const dir = mkdtempSync(join(tmpdir(), "site-stamp-repo-"));
    const run = (...args: string[]) => {
        const p = Bun.spawnSync(args, { cwd: dir });
        if (p.exitCode !== 0) throw new Error(`${args.join(" ")} failed: ${p.stderr.toString()}`);
    };
    run("git", "init", "-q");
    mkdirSync(resolve(dir, "examples/showcase/demo"), { recursive: true });
    mkdirSync(resolve(dir, "scripts"), { recursive: true });
    mkdirSync(resolve(dir, "site"), { recursive: true });
    mkdirSync(resolve(dir, "packages/shallot"), { recursive: true });
    mkdirSync(resolve(dir, "examples/showcase/ocean/src/ocean/src"), { recursive: true });
    writeFileSync(resolve(dir, "examples/showcase/demo/index.html"), "<html></html>\n");
    writeFileSync(
        resolve(dir, "examples/showcase/demo/package.json"),
        `{"dependencies":{"@dylanebert/shallot-wave":"workspace:*"}}\n`,
    );
    writeFileSync(
        resolve(dir, "examples/showcase/ocean/src/ocean/index.ts"),
        "export const ocean = 1;\n",
    );
    writeFileSync(resolve(dir, "scripts/build-site.ts"), "// builder\n");
    writeFileSync(resolve(dir, "site/rum-config.ts"), "// rum\n");
    writeFileSync(resolve(dir, "packages/shallot/package.json"), `{"version":"0.1.0"}\n`);
    run("git", "add", "-A");
    return dir;
}

test("site-stamp — the fingerprint moves on a demo source, a builder, and the release version", () => {
    const dir = fixtureRepo();
    try {
        const fp = () => demoFingerprints(dir, ["demo"]).demo;
        const base = fp();

        // the demo's own tracked source
        writeFileSync(resolve(dir, "examples/showcase/demo/index.html"), "<html> </html>\n");
        const afterDemo = fp();
        expect(afterDemo).not.toBe(base);
        writeFileSync(resolve(dir, "examples/showcase/demo/index.html"), "<html></html>\n");
        expect(fp()).toBe(base);

        // a builder file no showcase dir contains — the founding defect's own shape (`bf1c25f`
        // was a build-site.ts change, with every showcase file untouched)
        writeFileSync(resolve(dir, "scripts/build-site.ts"), "// builder v2\n");
        expect(fp()).not.toBe(base);
        writeFileSync(resolve(dir, "scripts/build-site.ts"), "// builder\n");

        // source from the workspace extension packed for this demo
        writeFileSync(
            resolve(dir, "examples/showcase/ocean/src/ocean/index.ts"),
            "export const ocean = 2;\n",
        );
        expect(fp()).not.toBe(base);
        writeFileSync(
            resolve(dir, "examples/showcase/ocean/src/ocean/index.ts"),
            "export const ocean = 1;\n",
        );
        expect(fp()).toBe(base);

        // the release version the ejected package.json is pinned to
        writeFileSync(resolve(dir, "packages/shallot/package.json"), `{"version":"0.2.0"}\n`);
        expect(fp()).not.toBe(base);
        writeFileSync(resolve(dir, "packages/shallot/package.json"), `{"version":"0.1.0"}\n`);
        expect(fp()).toBe(base);

        // untracked residue is out of scope — a `dist/` build leftover must not move the
        // fingerprint, or the relation would depend on which checkout computed it
        mkdirSync(resolve(dir, "examples/showcase/demo/dist"), { recursive: true });
        writeFileSync(resolve(dir, "examples/showcase/demo/dist/index.html"), "built\n");
        expect(fp()).toBe(base);
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
});

test("site-stamp — staleness is per demo dir, and an absent dir is not stale", () => {
    const dir = fixtureRepo();
    const out = mkdtempSync(join(tmpdir(), "site-stamp-out-"));
    try {
        // no demo dir built at all: nothing to judge, nothing stale (an unbuilt slot is clause
        // 4's own skip, not a staleness claim)
        expect(staleDemos(dir, out, ["demo"])).toEqual([]);

        mkdirSync(resolve(out, "demo"), { recursive: true });
        // present but unstamped
        expect(staleDemos(dir, out, ["demo"]).map((s) => s.slug)).toEqual(["demo"]);

        writeStamp(out, demoFingerprints(dir, ["demo"]), PROD_MODE);
        expect(staleDemos(dir, out, ["demo"])).toEqual([]);

        // sources move under a stamped artifact
        writeFileSync(resolve(dir, "examples/showcase/demo/index.html"), "<html> </html>\n");
        expect(staleDemos(dir, out, ["demo"]).map((s) => s.slug)).toEqual(["demo"]);
    } finally {
        rmSync(dir, { recursive: true, force: true });
        rmSync(out, { recursive: true, force: true });
    }
});

test("site-stamp — a stamp write merges over a prior build's other slots", () => {
    const out = mkdtempSync(join(tmpdir(), "site-stamp-merge-"));
    try {
        writeStamp(out, { a: "aaa", b: "bbb" }, PROD_MODE);
        writeStamp(out, { b: "ccc" }, PROD_MODE); // a `--demo b` rebuild
        expect(readStamp(out)?.demos).toEqual({ a: "aaa", b: "ccc" });
    } finally {
        rmSync(out, { recursive: true, force: true });
    }
});

// --- S1 (staging build mode): the stamp's mode branches clause 2's pin check and clause 5's
// env check two-sided ------------------------------------------------------------------------

/** A built-site fixture that clears every clause: relative paths, correct <title>, and the
 * given mode's RUM injection (`datadogInitSnippet(mode)`, `build-site.ts`'s own emitter — so
 * the fixture is byte-identical to what a real build of that mode would inject). */
function modeFixture(mode: "prod" | "staging"): string {
    const out = mkdtempSync(join(tmpdir(), `check-site-fixture-${mode}-`));
    for (const { slug } of ROSTER) {
        mkdirSync(resolve(out, slug), { recursive: true });
        writeFileSync(
            resolve(out, slug, "index.html"),
            `<!doctype html>
<html lang="en">
    <head>
        <title>${slug}</title>
        <link rel="stylesheet" href="./assets/index.css" />
    </head>
    <body>
        <script type="module" src="./assets/index.js"></script>
${datadogInitSnippet(mode)}    </body>
</html>
`,
        );
    }
    writeFileSync(resolve(out, "index.html"), `<!doctype html>\n<html><body></body></html>\n`);
    return out;
}

function stampFresh(fixture: string, mode: SiteMode) {
    writeStamp(
        fixture,
        demoFingerprints(
            repoRoot,
            ROSTER.map((d) => d.slug),
        ),
        mode,
    );
}

test("check-site — a staging artifact stamped staging passes clean", () => {
    const fixture = modeFixture("staging");
    try {
        stampFresh(fixture, STAGING_MODE);
        const { exitCode, out } = runCheck(fixture, { SITE_OUT_REQUIRED: "1" });
        expect(exitCode).toBe(0);
        expect(out).toContain("✓");
    } finally {
        rmSync(fixture, { recursive: true, force: true });
    }
});

test("check-site — a prod artifact stamped prod passes clean", () => {
    const fixture = modeFixture("prod");
    try {
        stampFresh(fixture, PROD_MODE);
        const { exitCode, out } = runCheck(fixture, { SITE_OUT_REQUIRED: "1" });
        expect(exitCode).toBe(0);
        expect(out).toContain("✓");
    } finally {
        rmSync(fixture, { recursive: true, force: true });
    }
});

// The two-sided mutation check the spec's Validation names directly: a staging artifact judged
// with the prod clause set must red, and vice versa — never pass on the union of both literals.
test("check-site — a staging artifact judged with the prod clause set reds", () => {
    const fixture = modeFixture("staging");
    try {
        stampFresh(fixture, PROD_MODE); // wrong clause set: mode says prod, content is staging
        const { exitCode, out } = runCheck(fixture, { SITE_OUT_REQUIRED: "1" });
        expect(exitCode).toBe(1);
        expect(out).toContain("missing the RUM env-derivation snippet for prod mode");
    } finally {
        rmSync(fixture, { recursive: true, force: true });
    }
});

test("check-site — a prod artifact judged with the staging clause set reds", () => {
    const fixture = modeFixture("prod");
    try {
        stampFresh(fixture, STAGING_MODE); // wrong clause set: mode says staging, content is prod
        const { exitCode, out } = runCheck(fixture, { SITE_OUT_REQUIRED: "1" });
        expect(exitCode).toBe(1);
        expect(out).toContain("missing the RUM env-derivation snippet for staging mode");
    } finally {
        rmSync(fixture, { recursive: true, force: true });
    }
});

test("check-site — a page carrying both mode's env literals reds on the two-sided check", () => {
    const out = mkdtempSync(join(tmpdir(), "check-site-fixture-bothmodes-"));
    try {
        for (const { slug } of ROSTER) {
            mkdirSync(resolve(out, slug), { recursive: true });
            writeFileSync(
                resolve(out, slug, "index.html"),
                `<!doctype html>
<html lang="en">
    <head>
        <title>${slug}</title>
        <link rel="stylesheet" href="./assets/index.css" />
    </head>
    <body>
        <script type="module" src="./assets/index.js"></script>
${datadogInitSnippet("prod")}    <!-- var ddEnv='staging'; -->
    </body>
</html>
`,
            );
        }
        writeFileSync(resolve(out, "index.html"), `<!doctype html>\n<html><body></body></html>\n`);
        stampFresh(out, PROD_MODE);
        const { exitCode, out: log } = runCheck(out, { SITE_OUT_REQUIRED: "1" });
        expect(exitCode).toBe(1);
        expect(log).toContain("carry the other mode's env");
    } finally {
        rmSync(out, { recursive: true, force: true });
    }
});

test("check-site — clause 2's artifact leg: a prod stamp naming a stale version reds", () => {
    const fixture = modeFixture("prod");
    try {
        stampFresh(fixture, { kind: "prod", version: "0.0.0-not-the-real-version" });
        const { exitCode, out } = runCheck(fixture, { SITE_OUT_REQUIRED: "1" });
        expect(exitCode).toBe(1);
        expect(out).toContain(
            "build stamp records prod mode pinned to v0.0.0-not-the-real-version",
        );
    } finally {
        rmSync(fixture, { recursive: true, force: true });
    }
});

test("check-site — clause 2's artifact leg: a staging stamp naming a non-tarball pin reds", () => {
    const fixture = modeFixture("staging");
    try {
        stampFresh(fixture, { kind: "staging", pin: "not-a-file-pin" });
        const { exitCode, out } = runCheck(fixture, { SITE_OUT_REQUIRED: "1" });
        expect(exitCode).toBe(1);
        expect(out).toContain('build stamp records staging mode with pin "not-a-file-pin"');
    } finally {
        rmSync(fixture, { recursive: true, force: true });
    }
});
