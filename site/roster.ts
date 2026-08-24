// The site roster — the six showcase demos. This is the single source of truth for which demos
// the site builds and serves. `check-site.ts` asserts this set equals the showcase dirs on disk,
// so a demo that ships without a roster entry (or vice versa) reds `bun check`.
//
// The roster is kept separate from `examples/AGENTS.md` because the two serve different readers:
// the roster is the build's source of truth and the thing `check-site.ts` set-gates, while
// `examples/AGENTS.md` lines are written for agent retrieval — different readers, so these are
// not copies to consolidate (the Locked decision). Rows carry a title, a play link, and a code
// link — nothing else; the demos speak for themselves.

export interface DemoEntry {
    slug: string;
    title: string;
}

export const ROSTER: DemoEntry[] = [
    {
        slug: "collapse",
        title: "Collapse",
    },
    {
        slug: "fountain",
        title: "Fountain",
    },
    {
        slug: "roads",
        title: "Roads",
    },
    {
        slug: "sandbox",
        title: "Sandbox",
    },
    {
        slug: "visualization",
        title: "Visualization",
    },
    {
        slug: "voxel",
        title: "Voxel",
    },
];
