import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

// The site build's freshness relation — the one fact `out/site/` cannot state about itself.
//
// `check-site.ts`'s artifact clauses (4-7) read `out/site/` as present-tense truth about the
// tree they are run in. That relation did not exist: a demo dir built before a fix, never
// rebuilt, read as a live defect in the current sources. Measured 2026-08-27 — a local
// `out/site/collapse/` built before shallot `bf1c25f` (the `mkdtempSync` parent + slug leaf
// title fix) reported clause 6's scratch-shaped-<title> red on a tree where the fix was
// already in, and a roadmap item was written from it. The artifact's own mtime (22:52) sat
// *after* the fix commit (22:50), so a timestamp comparison against the commit would not have
// caught it either — mtime records when bytes were written, not what they were written from.
//
// So the relation is content-derived: the build records, per demo, a fingerprint of everything
// it built that demo *from*; the check recomputes the fingerprint from the current tree and
// compares. Equal means the artifact is a build of these sources and may be judged. Unequal
// (or unstamped) means the artifact is a build of some other sources and the artifact clauses
// have nothing to say about this tree — they refuse rather than report.
//
// The fingerprint's inputs are the demo's *build* inputs, no more:
//   - every git-tracked file under `examples/showcase/<slug>/`, path and content. Tracked, not
//     `readdirSync` — the same law `site/roster.ts` and `scripts/check-docs.ts` write for their
//     own populations: a scope that reads local `dist/`/`node_modules/` residue makes the
//     fingerprint depend on which checkout ran it. A tracked file absent from disk hashes as a
//     `missing` marker, so a deletion in progress reads stale rather than identical.
//   - the builders: `scripts/build-site.ts` (eject + inject) and the `site/rum-*` modules whose
//     literals land in every built page. A change here changes every demo's output with no
//     showcase file touched — the founding defect's own shape, since `bf1c25f` was a
//     `build-site.ts` change.
//   - the release version from `packages/shallot/package.json` — the pin `build-site.ts` writes
//     into each ejected `package.json`, so a version bump means the artifact was built against
//     a different published package.
//
// Not an input: `packages/shallot/**` source. Each demo is ejected and installed against the
// *published* package, so in-repo package edits do not reach the artifact until a release
// changes the version — which the version input already catches.

/** The stamp lives at the root of the output dir, next to `index.html`. Plain-named (not a
 * dotfile) so no static host's filter drops it and the artifact carries its own provenance. */
export const STAMP_FILE = "build-stamp.json";

/** Which build mode produced the artifact, and the `@dylanebert/shallot` pin that mode wrote into
 * every demo's ejected `package.json` — recorded because the pin itself never survives to be read
 * back: each demo is ejected into a scratch tree under /tmp and that tree is deleted once the demo
 * is built, so `check-site.ts`'s clause 2 has nothing else to read the actual pin from. Prod
 * carries the release version it pinned; staging carries the `file:<tgz>` pin `bun pm pack`
 * produced, so a mode mix-up (prod artifact stamped staging, or vice versa) reds on the pin shape
 * instead of passing silently. */
export type SiteMode = { kind: "prod"; version: string } | { kind: "staging"; pin: string };

function isSiteMode(v: unknown): v is SiteMode {
    if (typeof v !== "object" || v === null) return false;
    const m = v as Record<string, unknown>;
    if (m.kind === "prod") return typeof m.version === "string";
    if (m.kind === "staging") return typeof m.pin === "string";
    return false;
}

export interface SiteStamp {
    /** Bumped when the fingerprint recipe below changes: an old stamp then reads stale, which is
     * the correct answer — a fingerprint computed by a different recipe is not comparable. Bumped
     * 1 → 2 to add `mode` — a recipe-1 stamp predates mode recording and reads stale rather than
     * being read with a mode it never carried. */
    recipe: 2;
    /** The mode this build ran in, and the engine pin it used — see `SiteMode` above. Overwritten
     * on every write (unlike `demos`, which merges) since a build run has exactly one mode. */
    mode: SiteMode;
    /** slug → fingerprint of the sources that demo was built from. */
    demos: Record<string, string>;
}

const RECIPE: SiteStamp["recipe"] = 2;

/** The builder files whose contents reach every built page regardless of demo. */
const BUILDER_FILES = [
    "scripts/build-site.ts",
    "site/rum-config.ts",
    "site/rum-runtime.ts",
    "site/rum-sampler.ts",
    "site/roster.ts",
];

const SHOWCASE_PREFIX = "examples/showcase/";

function hashFile(hasher: Bun.CryptoHasher, rootDir: string, rel: string): void {
    const full = resolve(rootDir, rel);
    hasher.update(`\0${rel}\0`);
    if (existsSync(full)) {
        hasher.update(readFileSync(full));
    } else {
        hasher.update("missing");
    }
}

/** Tracked files under `examples/showcase/`, grouped by slug. Asking git makes the scope
 * identical in every checkout (`site/roster.ts`'s law, same reason). */
function trackedShowcaseFiles(rootDir: string): Map<string, string[]> {
    const tracked = Bun.spawnSync(["git", "ls-files", "-z", SHOWCASE_PREFIX], { cwd: rootDir });
    if (!tracked.success) {
        throw new Error(
            "`git ls-files` failed — the site build stamp needs a git checkout to scope each demo's sources.",
        );
    }
    const bySlug = new Map<string, string[]>();
    for (const path of tracked.stdout.toString().split("\0").filter(Boolean)) {
        const parts = path.slice(SHOWCASE_PREFIX.length).split("/");
        if (parts.length < 2) continue; // a file directly in showcase/ belongs to no demo
        const slug = parts[0];
        const list = bySlug.get(slug);
        if (list) list.push(path);
        else bySlug.set(slug, [path]);
    }
    for (const list of bySlug.values()) list.sort();
    return bySlug;
}

/** Fingerprints every roster slug's build inputs in one pass — one `git ls-files` and one read
 * of the shared builder inputs, whatever the roster's size. */
export function demoFingerprints(rootDir: string, slugs: string[]): Record<string, string> {
    const bySlug = trackedShowcaseFiles(rootDir);

    const shared = new Bun.CryptoHasher("sha256");
    shared.update(`recipe:${RECIPE}\0`);
    const pkgPath = resolve(rootDir, "packages/shallot/package.json");
    const version = existsSync(pkgPath)
        ? ((JSON.parse(readFileSync(pkgPath, "utf8")) as { version?: string }).version ?? "missing")
        : "missing";
    shared.update(`version:${version}\0`);
    for (const rel of BUILDER_FILES) hashFile(shared, rootDir, rel);
    const sharedDigest = shared.digest("hex");

    const out: Record<string, string> = {};
    for (const slug of slugs) {
        const hasher = new Bun.CryptoHasher("sha256");
        hasher.update(`${sharedDigest}\0${slug}\0`);
        for (const rel of bySlug.get(slug) ?? []) hashFile(hasher, rootDir, rel);
        out[slug] = hasher.digest("hex").slice(0, 32);
    }
    return out;
}

export function readStamp(outDirPath: string): SiteStamp | null {
    const path = resolve(outDirPath, STAMP_FILE);
    if (!existsSync(path)) return null;
    try {
        const parsed = JSON.parse(readFileSync(path, "utf8")) as Partial<SiteStamp>;
        if (
            parsed.recipe !== RECIPE ||
            typeof parsed.demos !== "object" ||
            parsed.demos === null ||
            !isSiteMode(parsed.mode)
        ) {
            return null;
        }
        return { recipe: RECIPE, mode: parsed.mode, demos: parsed.demos as Record<string, string> };
    } catch {
        return null; // an unparseable stamp is no stamp — the artifact reads stale
    }
}

/** Records the fingerprints of the demos this build wrote, merging over any prior stamp — a
 * `--demo <slug>` build only rebuilt one slot, so the other slots keep the stamp they earned.
 * `mode` is not merged — it names the mode this build ran in, and a build run has one mode. */
export function writeStamp(
    outDirPath: string,
    entries: Record<string, string>,
    mode: SiteMode,
): void {
    const prior = readStamp(outDirPath);
    const stamp: SiteStamp = {
        recipe: RECIPE,
        mode,
        demos: { ...(prior?.demos ?? {}), ...entries },
    };
    writeFileSync(resolve(outDirPath, STAMP_FILE), `${JSON.stringify(stamp, null, 4)}\n`);
}

export interface StaleDemo {
    slug: string;
    reason: string;
}

/** The demos present in `outDirPath` whose artifact is not a build of `rootDir`'s current
 * sources. An absent demo dir is not stale — a `--demo` filtered build only writes some slots. */
export function staleDemos(rootDir: string, outDirPath: string, slugs: string[]): StaleDemo[] {
    const present = slugs.filter((slug) => existsSync(resolve(outDirPath, slug)));
    if (present.length === 0) return [];
    const stamp = readStamp(outDirPath);
    if (!stamp) {
        return present.map((slug) => ({
            slug,
            reason: `no readable ${STAMP_FILE} — built before the artifact recorded its sources`,
        }));
    }
    const current = demoFingerprints(rootDir, present);
    const stale: StaleDemo[] = [];
    for (const slug of present) {
        const built = stamp.demos[slug];
        if (!built) {
            stale.push({ slug, reason: `no stamp entry — this slot was written by another build` });
        } else if (built !== current[slug]) {
            stale.push({
                slug,
                reason: `built from ${built}, sources now ${current[slug]}`,
            });
        }
    }
    return stale;
}
