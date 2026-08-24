import { resolve } from "node:path";

// The site roster — derived from what git tracks under `examples/showcase/`, not a
// `readdirSync` of the filesystem. A `readdirSync` reads whatever a particular checkout
// happens to have on disk: a directory holding nothing but ignored build residue (`dist/`,
// `node_modules/`, `test-results/`) still joins the roster, so the manifest-presence arm in
// `check-site.ts` reds on a checkout where the repo is correct — the same law
// `scripts/check-docs.ts` wrote for its doc set: a check whose coverage depends on local
// state is how a stale tree reads green. Asking git makes the scope identical in every
// checkout. Each showcase dir joins the site by being tracked; the title is the slug with
// each kebab segment capitalized (no override map until a slug needs one — every current
// slug is single-word, so the rule is exact today). `check-site.ts` gates the dirs' contracts
// (manifest presence, workspace-form shallot pin, no escaping import, no root-absolute
// path); set membership is by construction, so no separate roster-equals-disk clause
// survives. This module is the single import point: `build-site.ts`, `demos.ts`, and
// `check-site.ts` all import `ROSTER` and none of them learn about the filesystem — the
// change is the provenance of the array, not the shape consumers see.

export interface DemoEntry {
    slug: string;
    title: string;
}

const repoRoot = resolve(import.meta.dir, "..");

function deriveTitle(slug: string): string {
    return slug
        .split("-")
        .map((seg) => seg.charAt(0).toUpperCase() + seg.slice(1))
        .join(" ");
}

// The showcase set is what git tracks, not what the filesystem holds — the same law
// `scripts/check-docs.ts` wrote for its doc set. A `readdirSync` would read `dist/`,
// `node_modules/`, and `test-results/` residue from a deleted project; asking git makes the
// scope identical in every checkout.
const prefix = "examples/showcase/";
const tracked = Bun.spawnSync(["git", "ls-files", "-z", prefix], { cwd: repoRoot });
if (!tracked.success) {
    console.error(
        "✗ `git ls-files` failed — site/roster.ts needs a git checkout to scope the showcase set.",
    );
    process.exit(1);
}
const trackedFiles = tracked.stdout.toString().split("\0").filter(Boolean);
const slugs = [
    ...new Set(
        trackedFiles
            .filter((p) => p.startsWith(prefix))
            .map((p) => p.slice(prefix.length).split("/"))
            .filter((parts) => parts.length > 1) // skip files directly in showcase/ (e.g. .gitkeep)
            .map((parts) => parts[0]),
    ),
].sort();

if (slugs.length === 0) {
    console.error(
        "✗ `git ls-files 'examples/showcase/'` matched no project dir — the roster would be empty.",
    );
    process.exit(1);
}

export const ROSTER: DemoEntry[] = slugs.map((slug) => ({ slug, title: deriveTitle(slug) }));
