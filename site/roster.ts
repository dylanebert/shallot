import { readdirSync } from "node:fs";
import { resolve } from "node:path";

// The site roster — derived from `examples/showcase/` by enumeration, not a hand-maintained list.
// Each showcase dir joins the site by existing; the title is the slug with each kebab segment
// capitalized (no override map until a slug needs one — every current slug is single-word, so the
// rule is exact today). `check-site.ts` gates the dirs' contracts (manifest presence, workspace-form
// shallot pin, no escaping import, no root-absolute path); set membership is by construction, so no
// separate roster-equals-disk clause survives. This module is the single import point:
// `build-site.ts`, `demos.ts`, and `check-site.ts` all import `ROSTER` and none of them learn about
// the filesystem — the change is the provenance of the array, not the shape consumers see.

export interface DemoEntry {
    slug: string;
    title: string;
}

const repoRoot = resolve(import.meta.dir, "..");
const showcaseDir = resolve(repoRoot, "examples/showcase");

function deriveTitle(slug: string): string {
    return slug
        .split("-")
        .map((seg) => seg.charAt(0).toUpperCase() + seg.slice(1))
        .join(" ");
}

export const ROSTER: DemoEntry[] = readdirSync(showcaseDir, { withFileTypes: true })
    .filter((e) => e.isDirectory())
    .map((e) => e.name)
    .sort()
    .map((slug) => ({ slug, title: deriveTitle(slug) }));
