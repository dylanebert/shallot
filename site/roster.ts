// The site roster — the six showcase demos, hand-written in site voice. This is the single source
// of truth for which demos the site builds and serves. `check-site.ts` asserts this set equals the
// showcase dirs on disk, so a demo that ships without a roster entry (or vice versa) reds `bun check`.
//
// Blurbs are written for a visitor, not derived from `examples/AGENTS.md` — different readers (the
// Locked decision: "site copy is written for a visitor; `examples/AGENTS.md` lines are written for
// agent retrieval — different readers, so these are not copies to consolidate").

export interface DemoEntry {
    slug: string;
    title: string;
    blurb: string;
}

export const ROSTER: DemoEntry[] = [
    {
        slug: "collapse",
        title: "Collapse",
        blurb: "A structure pancakes under rolling impact — AVBD rigid bodies in real time.",
    },
    {
        slug: "fountain",
        title: "Fountain",
        blurb: "A GPU particle fountain — thousands of sprites in a single draw call.",
    },
    {
        slug: "roads",
        title: "Roads",
        blurb: "Sketch a road network across terrain — capture, edit, and re-drive corridors.",
    },
    {
        slug: "sandbox",
        title: "Sandbox",
        blurb: "Walk a foggy world with physics, audio, and outlines — the full feature stack.",
    },
    {
        slug: "visualization",
        title: "Visualization",
        blurb: "Lines, arrows, SDF text, and tweens — the rendering primitives, on display.",
    },
    {
        slug: "voxel",
        title: "Voxel",
        blurb: "Carve a voxel world — marching-cubes meshing with real-time editing.",
    },
];
