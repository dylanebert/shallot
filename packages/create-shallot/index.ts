#!/usr/bin/env bun

import { existsSync, mkdirSync, writeFileSync } from "fs";
import { dirname, join, resolve } from "path";

/**
 * the project files keyed by relative path, with the project name interpolated. the single source of
 * truth for `bun create shallot <name>`.
 *
 * the project is pure data — a `shallot.json` manifest + plugin modules + `public/`, no vite
 * boilerplate. the CLI provides every harness over it: `shallot dev` runs it standalone with hot
 * reload, `shallot build` ships it (web + native targets). the emitted CLAUDE.md / AGENTS.md point an
 * agent at the installed engine's contract.
 */
export function template(name: string): Record<string, string> {
    return {
        "public/icon.svg": ICON,
        ".gitignore": "node_modules/\ndist/\nbuild/\n",
        "package.json":
            JSON.stringify(
                {
                    name,
                    version: "0.0.0",
                    private: true,
                    type: "module",
                    dependencies: { "@dylanebert/shallot": "latest" },
                    devDependencies: { typescript: "^7.0.2" },
                },
                null,
                2,
            ) + "\n",
        "tsconfig.json":
            JSON.stringify(
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
            ) + "\n",
        "shallot.json": MANIFEST,
        "src/env.d.ts": ENV,
        "src/spin.ts": SPIN,
        "public/scenes/scene.scene": SCENE,
        "README.md": readme(name),
        "AGENTS.md": agents(name),
        "CLAUDE.md": agents(name),
    };
}

/** write a template file map under dir, creating parent directories as needed. */
export function scaffold(dir: string, files: Record<string, string>): void {
    for (const [rel, content] of Object.entries(files)) {
        const path = join(dir, rel);
        mkdirSync(dirname(path), { recursive: true });
        writeFileSync(path, content);
    }
}

const ICON = `<svg id="Shallot" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 80">
  <title>shallot icon</title>
  <defs>
    <radialGradient id="baseGradient" cx="35%" cy="30%" r="70%" fx="25%" fy="20%">
      <stop offset="0%" stop-color="#F5D4B8"/>
      <stop offset="45%" stop-color="#E8A86B"/>
      <stop offset="100%" stop-color="#B87654"/>
    </radialGradient>
  </defs>
  <g transform="rotate(35 40.0 40.0)">
    <path d="M40,2 C44,10 66,28 66,46 C66,60 48,70 40,78 C32,70 14,60 14,46 C14,28 36,10 40,2 Z" fill="#E8A86B"/>
    <path d="M40,6 C37,14 22,28 20,44 C20,52 28,62 36,70 C34,58 26,46 26,38 C26,26 38,12 40,6 Z" fill="#D49560"/>
    <path d="M40,6 C43,14 58,28 60,44 C60,52 52,62 44,70 C46,58 54,46 54,38 C54,26 42,12 40,6 Z" fill="#D49560"/>
    <path d="M40,8 C40,20 40,50 40,72" stroke="#6B4230" stroke-width="1" stroke-opacity="0.4" fill="none" stroke-linecap="round"/>
    <path d="M40,78 C48,70 66,60 66,46 C61,58 44,70 40,73 Z" fill="#D49560"/>
    <path d="M40,2 C44,10 66,28 66,46 C66,60 48,70 40,78 C32,70 14,60 14,46 C14,28 36,10 40,2 Z" fill="none" stroke="#6B4230" stroke-width="2"/>
  </g>
</svg>
`;

const readme = (name: string) => `# ${name}

A shallot project.

## Develop

\`\`\`bash
bun install
bunx shallot dev
\`\`\`

\`bun install\` fetches the engine. \`bunx shallot dev\` runs the project with hot
reload. Edit \`src/spin.ts\` (a plugin) and \`shallot.json\` (the manifest: scene +
plugin enablement) in your IDE.

## Ship

\`\`\`bash
bunx shallot build
\`\`\`

Builds a web bundle to \`dist/\`. For native targets:
\`bunx shallot build --target windows|mac|linux\` (add \`--release\` for an optimized build).
`;

const agents = (name: string) => `# ${name}

A WebGPU game built on \`@dylanebert/shallot\`.

## Layout

- \`shallot.json\` — the manifest: which scene to open + which plugins to enable
- \`public/scenes/*.scene\` — the world as declarative XML (each \`<a>\` is an entity, each attribute a component)
- \`src/*.ts\` — your plugins (a plugin is data: components + systems)

## Build, run, verify

\`\`\`bash
bunx tsc --noEmit                                # typecheck — run after every change
bunx shallot dev                                 # run with hot reload while you work
bunx shallot build [--target windows|mac|linux]  # ship it (web bundle, or a native app)
bunx shallot verify                              # prove it: boot headless, check it renders, exit 0/nonzero
\`\`\`

\`shallot verify\` is the verification step — a self-terminating gate that boots the project in a real
headless browser, waits for a settled frame, and exits nonzero on failure. Nothing left running. \`--json\`
emits the full result; \`--screenshot <path>\` saves a frame. It drives Playwright, an optional one-time
install: \`bun add -d playwright && bunx playwright install chromium\` (exit code 3 names this command if
it's missing). By default it checks the scene rendered; to assert your own pass/fail (entity poses,
physics state) install \`window.__harness\` via \`installHarness\` from \`@dylanebert/shallot/harness\` — in a
manifest project like this one, from a plugin's \`initialize(state)\` hook (the engine's AGENTS.md has the
worked example).

## Engine reference

The engine is the documentation. Read \`node_modules/@dylanebert/shallot/AGENTS.md\` for the full
contract (ECS, plugins, scenes, GPU, UI, and the \`shallot verify\` harness), and every public export
carries JSDoc. The examples index lives at \`node_modules/@dylanebert/shallot/examples/AGENTS.md\` — grep
it for the problem you have, then read that recipe's source, before writing a pattern from scratch.

## Conventions

Data-oriented, ECS, declarative. Add components and systems, not methods — a \`Jump\` marker plus a
system, never \`player.jump()\`. Scenes declare; code transforms. One source of truth: every value has
one authoritative home; derive, don't duplicate.
`;

// The project manifest: `shallot dev` and a shipped `shallot build` both read it. `scene` is the scene
// to open; `plugins` is enablement — "Orbit": true turns on the orbit camera the scene uses, and
// "Spin": "./src/spin" declares our own plugin by its module path. The default plugins (render, lit
// surface) are on unless you set one false.
const MANIFEST = `{
  "$schema": "./node_modules/@dylanebert/shallot/shallot.schema.json",
  "scene": "scenes/scene.scene",
  "plugins": {
    "Orbit": true,
    "Spin": "./src/spin"
  }
}
`;

// Ambient types + the tsconfig anchor. `include: ["src"]` needs at least one matching file, so this
// keeps `bunx tsc --noEmit` green (no TS18003) even when spin.ts is deleted for a static scene. It's
// infrastructure, not demo content — don't delete it.
const ENV = `/// <reference types="@webgpu/types" />
`;

const SPIN = `import { type Plugin, type State, type System, Part, quat, Transform } from "@dylanebert/shallot";

// A plugin is plain data: components + systems the engine runs. This system spins every Part around Y.
// It runs in the "simulation" group, which plays when the project runs. Delete this file (and its
// \`shallot.json\` entry) for a static scene.
const SpinSystem: System = {
    group: "simulation",
    update(state: State) {
        // Derive the angle from elapsed time, not a module-level accumulator, so it stays
        // correct after a hot reload or State rebuild rather than carrying stale rotation.
        const q = quat(0, (state.time.elapsed * 45) % 360, 0);
        for (const eid of state.query([Part, Transform])) {
            Transform.rot.set(eid, q.x, q.y, q.z, q.w);
        }
    },
};

// The default export is the plugin — \`shallot.json\` references this file by path and imports
// its default. The name ("Spin") is how the manifest lists it.
const SpinPlugin: Plugin = { name: "Spin", systems: [SpinSystem] };
export default SpinPlugin;
`;

const SCENE = `<scene>
    <a ambient-light="color: 0xd0dcec; intensity: 0.5" />
    <a directional-light="direction: -0.4 -1 -0.55; color: 0xfff4e0; intensity: 1.1" />

    <!-- the camera auto-binds to the page's <canvas>; drag to orbit, scroll to zoom -->
    <a camera sear orbit="distance: 5; yaw: 0.6; pitch: 0.25" transform />

    <!-- a Part is the engine's drop-in renderable: a mesh (default "cube") wearing a surface (default "default", lit) -->
    <a part transform="pos: 0 0 0" color="rgba: 0.85 0.55 0.35 1" />
</scene>
`;

if (import.meta.main) {
    const args = process.argv.slice(2);
    const name = args.find((a) => !a.startsWith("--"));
    if (!name) {
        console.error("Usage: bun create shallot <project-name>");
        process.exit(1);
    }

    const dir = resolve(name);
    if (existsSync(dir)) {
        console.error(`Directory "${name}" already exists`);
        process.exit(1);
    }

    scaffold(dir, template(name));

    console.log(`Created ${name}/`);
    console.log();
    console.log("Next steps:");
    console.log(`  cd ${name}`);
    console.log("  bun install");
    console.log("  bunx shallot dev");
}
