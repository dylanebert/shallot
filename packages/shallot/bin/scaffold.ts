// Scaffold fragments emitted into a recipe copied out by `shallot recipe` (bin/recipe.ts): the
// AGENTS.md/CLAUDE.md pointer stanza that hands an agent the engine's contract, and a standalone
// tsconfig. They live here — in the engine package, which ships `bin/` — because the two would-be
// sources can't import each other: `create-shallot` ships only `index.ts` and runs before the engine
// is installed (nothing to import), and the engine tarball omits `create-shallot`. So this is the one
// source for the copy-out, and `create-shallot`'s own AGENTS.md carries the identical
// `ENGINE_REFERENCE` stanza, guarded consistent by `recipe.test.ts`.

// The reach the recipes-install-ux distribution decision rests on: a stock harness (Claude Code,
// Cursor, Codex) never reads instruction files from node_modules, but it follows an explicit path from
// a file in the project it opens. A copied recipe therefore carries this at its root so an agent finds
// the installed engine's agent surface. Kept byte-identical to `create-shallot`'s scaffold stanza.
export const ENGINE_REFERENCE = `## Engine reference

The engine is the documentation. Read \`node_modules/@dylanebert/shallot/AGENTS.md\` for the full
contract (ECS, plugins, scenes, GPU, UI, and the \`shallot verify\` harness), and every public export
carries JSDoc. The examples index lives at \`node_modules/@dylanebert/shallot/examples/AGENTS.md\` — grep
it for the problem you have, then read that recipe's source, before writing a pattern from scratch.
\`bunx shallot recipe <name> [dir]\` copies a recipe out of the installed package into a runnable project
(bare: lists them).`;

/** the AGENTS.md / CLAUDE.md a copied recipe carries: what it is, how to run it, the engine pointer. */
export function recipeDoc(name: string): string {
    return `# ${name}

A shallot recipe — a minimal project demonstrating one concept, copied out of \`@dylanebert/shallot\`.

## Develop

\`\`\`bash
bun install
bunx shallot dev
\`\`\`

\`bun install\` fetches the engine. \`bunx shallot dev\` runs the project with hot reload. Read
\`shallot.json\` (the manifest: scene + plugin enablement) and \`src/*.ts\` (the plugins).

${ENGINE_REFERENCE}
`;
}

// A standalone tsconfig for a copied recipe: in the monorepo the recipes share a root config, so they
// carry none of their own; copied out, they need one for `bunx tsc` to resolve the engine + webgpu types.
export const RECIPE_TSCONFIG = `${JSON.stringify(
    {
        compilerOptions: {
            target: "ESNext",
            module: "ESNext",
            moduleResolution: "bundler",
            lib: ["ESNext", "DOM", "DOM.Iterable"],
            types: ["@webgpu/types"],
            strict: true,
            noEmit: true,
            skipLibCheck: true,
        },
        include: ["src"],
    },
    null,
    2,
)}\n`;
