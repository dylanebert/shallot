import { buildIndex, type DocIndex } from "./docs";

// the render-ready docs artifact, globbed at build (see packages/shallot/scripts/docs.ts).
// kept out of ./docs so that module stays pure + bun-testable without import.meta.glob.
const corpus = import.meta.glob("../../../../../docs/dist/**/*.md", {
    query: "?raw",
    import: "default",
    eager: true,
}) as Record<string, string>;

let cached: DocIndex | null = null;

/** the built doc index — pages + reference symbols — over docs/dist, memoized.
 * The docs panel calls this once, then renders {@link DocPage.blocks} and queries via search. */
export function docs(): DocIndex {
    if (!cached) cached = buildIndex(corpus);
    return cached;
}
