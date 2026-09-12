import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { readCheckDeclarations } from "./surface";

// `bun run examples:index [--check]`: emit `examples/AGENTS.md` from each source-visible
// `examples/*/shallot.json` (`--root <dir>` is for isolated fixture tests). The index is never
// hand-written; `--check` reds when the committed file differs from what the declarations
// generate. Every source-visible example dir must declare both fields.

const args = Bun.argv.slice(2);
const rootIndex = args.indexOf("--root");
if (rootIndex !== -1 && args[rootIndex + 1] === undefined) {
    console.error("✗ --root requires a directory");
    process.exit(1);
}
const root = resolve(
    rootIndex === -1 ? resolve(import.meta.dir, "..") : (args[rootIndex + 1] as string),
);
const examples = resolve(root, "examples");
const out = resolve(examples, "AGENTS.md");
const KINDS = ["recipe", "showcase"] as const;
type Kind = (typeof KINDS)[number];

const COVERAGE_RECIPE_AREAS = {
    "Physics closeout": [],
    "The frame": [
        "animate-with-clips",
        "ascii",
        "custom-material",
        "day-night-sky",
        "import-gltf",
        "particles",
    ],
    "Input and gameplay": ["drive-a-vehicle", "first-person", "respond-to-input"],
    "Engine core, audio, project and CLI": ["play-sound", "save-and-restore", "svelte-ui"],
} as const;

const COVERAGE_PLAN = `## Coverage plan

The builder journey runs from a blank directory to a played URL: create a project and its first scene; define state and persistence; add actions, gameplay and physics; draw the frame and emit confirmed effects; add live UI; then check, build and publish the same project. Coverage is ordered by risk rather than by that journey: physics first, the frame second, then input with gameplay, then engine core with audio, project and CLI. A recipe carries demand; module checks carry reference or invariant claims.

### 1. Physics closeout

No physics recipe is kept or added. At the Box3D resync, regenerate from and pin Erin Catto's upstream Box3D; the current harness fork stops being the referent. Reproduce mechanisms in checks, not sample scenes.

| family | kept or added recipes | retired claims and checks owed | source |
| --- | --- | --- | --- |
| solver parity | — | world hashes and periodic body state at every step, single-thread and every thread count, plus restore/resimulation equality | reference |
| solver scheduling | — | graph coloring, overflow, sleeping, waking and parallel work partition preserve the upstream result | reference |
| contact generation | — | distance, time of impact, geometry, hull, manifold and contact persistence match upstream cases | reference |
| broad phase and queries | — | dynamic tree, pair table, ray casts, shape casts and mover queries retain exact selected results | reference |
| joints and limits | — | \`joints\`, \`breakable-joints\` and \`ragdoll\`: joint creation, limits, motors, break thresholds and events | reference |
| contacts and moving support | — | \`moving-platform\`, \`surface-friction\` and \`physics-playground\`: support velocity, friction/contact response, sleep and stacked mixed-shape behavior | reference |
| shapes and composition | — | compounds, meshes and heightfields compose through the public body/shape seam without changing their upstream physical result | reference |
| public physics seam | — | service-free plugin lifecycle, shared-pool ownership, generic observed poses, snapshot/restore/hash and interpolation | invariant |

### 2. The frame

| family | kept or added recipes | retired claims and checks owed | source |
| --- | --- | --- | --- |
| animation and GPU particles | keep \`animate-with-clips\`, \`particles\` | clips remain tweened playables on a loop; compute-to-draw particles remain observable without a per-frame CPU simulation | demand |
| materials, sky and ASCII | keep \`custom-material\`, \`day-night-sky\`, \`ascii\` | custom surface/backdrop, moving-sun sky and character-cell output; CPU structure plus the cheapest rendered witness for each visible claim | demand |
| glTF placement | keep \`import-gltf\` | load and place one model | demand |
| glTF conformance | — | accessor, sparse-data and node-transform conformance against Khronos assets and Three.js decode | reference |
| world-space UI | add \`world-space-ui\` from retired \`annotate-the-world\` and \`billboards-and-sprites\` | labels, billboards, sprites and meters stay attached and camera-facing through one demand recipe and module checks | demand |
| compute seam | — | retired \`compute-and-readback\`: dispatched data is readable through the public compute seam | invariant |
| frame pipeline | — | transforms, culling, compaction, lighting, fog, mirrors, post, atlases, skinning and drawing extras; decide T7 at the first rendered claim and persist human captures | invariant |

### 3. Input and gameplay

| family | kept or added recipes | retired claims and checks owed | source |
| --- | --- | --- | --- |
| input response | keep \`respond-to-input\` | held and edge actions move and trigger the requested thing | demand |
| mapped-action mechanism | — | actions inject above devices on a tick; one Chromium witness proves a real event reaches the same map, following the Godot and Unity precedent | reference |
| vehicle play | keep \`drive-a-vehicle\` | WASD drives the public vehicle seam; physics mechanisms stay in Physics closeout | demand |
| first-person play | keep \`first-person\` | walking, grounded movement and mapped mouse look; rebuild the scene after the person chooses or replaces the proposed Valve/Unreal movement-test reference | demand |
| gameplay mechanisms | — | character, player, BVH, orbit and profile run on the stepped clock; fixed gameplay renders through interpolation without refresh-rate judder | invariant |
| observation and replay | — | action injection, query, record and hash share one registered-data contract; confirm the record shape and prove the plugin absent from built games | invariant |

### 4. Engine core, audio, project and CLI

| family | kept or added recipes | retired claims and checks owed | source |
| --- | --- | --- | --- |
| first scene | add \`hello-world\` in place of retired \`game-loop\` | a rotating cube and one system establish the smallest blank-project-to-running-scene path | demand |
| state, save and live UI | keep \`save-and-restore\`, \`svelte-ui\` | save, change and restore; a HUD reads live game state; scene codec, isolation and snapshot mechanisms stay module checks | demand |
| spatial sound | keep \`play-sound\` | a moving world source pans and attenuates against its listener through CPU-readable voice state | demand |
| audio mechanism | — | DSP kernels retain their permissive external golds and the worklet gets one integration witness | reference |
| project and CLI | — | create reaches the first scene without a prompt; add, dev, check, packed build, native target and publish/deploy refusal operate from declarations with structured recovery | invariant |
| played URL | — | a packed consumer builds the same checked project and its published URL boots to an observable frame; hosted reports go to files and unsupported seats refuse | demand |
| sandbox showcase | add the sandbox showcase after the area recipes | compose the recipes on public package seams, run only user-project gates, and retain the rule-of-three boundary log | demand |
`;

const evidence = Bun.spawnSync(
    ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "examples"],
    { cwd: root, stdout: "pipe", stderr: "pipe" },
);
if (!evidence.success) {
    console.error("✗ `git ls-files` failed — examples index needs a Git source file list.");
    process.exit(1);
}

const names = new Set<string>();
for (const path of evidence.stdout.toString().split("\0")) {
    if (!path.startsWith("examples/")) continue;
    const relative = path.slice("examples/".length);
    const slash = relative.indexOf("/");
    if (slash > 0 && slash < relative.length - 1) names.add(relative.slice(0, slash));
}

const rows: { name: string; kind: Kind; description: string; checkSize: string }[] = [];
const errors: string[] = [];
for (const name of [...names].sort()) {
    const path = resolve(examples, name, "shallot.json");
    if (!existsSync(path)) {
        errors.push(`examples/${name}/ has no shallot.json`);
        continue;
    }
    const { kind, description, problem, check } = JSON.parse(readFileSync(path, "utf8"));
    if (!KINDS.includes(kind))
        errors.push(`examples/${name}/shallot.json: kind must be ${KINDS.join(" | ")}`);
    else if (typeof description !== "string" || description.trim() === "")
        errors.push(`examples/${name}/shallot.json: description is missing`);
    else {
        let checkSize = "-";
        if (check !== undefined) {
            if (
                Array.isArray(check) ||
                check === null ||
                typeof check !== "object" ||
                typeof check.file !== "string" ||
                Object.keys(check).some((field) => field !== "file")
            ) {
                errors.push(`examples/${name}/shallot.json: check must be { file }`);
            } else {
                const result = readCheckDeclarations(root, resolve(examples, name, check.file));
                errors.push(...result.errors);
                checkSize = [...new Set(result.rows.map((row) => row.size))].join("/") || "-";
            }
        }
        rows.push({
            name,
            kind,
            checkSize,
            description: (typeof problem === "string" && problem.trim() !== ""
                ? problem
                : description
            ).trim(),
        });
    }
}
if (errors.length > 0) {
    for (const e of errors) console.error(`✗ ${e}`);
    process.exit(1);
}

const cell = (s: string) => s.replaceAll("|", "\\|");
const lines = [
    "# Examples",
    "",
    "From `bun run examples:index` and `examples/*/shallot.json`; edit manifests. Use `bunx shallot dev examples/<name>`.",
];
for (const kind of KINDS) {
    const own = rows.filter((r) => r.kind === kind);
    if (own.length === 0) continue;
    lines.push("", kind === "recipe" ? "## Recipes" : "## Showcase", "");
    if (kind === "recipe") {
        lines.push("| name | description | add |", "| --- | --- | --- |");
        for (const r of own)
            lines.push(
                `| \`${r.name}\` | ${cell(r.description)}${r.checkSize === "-" ? "" : ` (check: ${cell(r.checkSize)})`} | \`bunx shallot add ${r.name}\` |`,
            );
    } else {
        lines.push("| name | description |", "| --- | --- |");
        for (const r of own) lines.push(`| \`${r.name}\` | ${cell(r.description)} |`);
    }
}
if (rootIndex === -1) {
    const mapped: string[] = Object.values(COVERAGE_RECIPE_AREAS).flatMap((names) => [...names]);
    const actual = rows.filter((row) => row.kind === "recipe").map((row) => row.name);
    const duplicate = mapped.find((name, index) => mapped.indexOf(name) !== index);
    const missing = actual.filter((name) => !mapped.includes(name));
    const stale = mapped.filter((name) => !actual.includes(name));
    if (duplicate || missing.length > 0 || stale.length > 0) {
        console.error(
            `✗ coverage-plan recipe map drift${duplicate ? `; duplicate: ${duplicate}` : ""}${missing.length > 0 ? `; missing: ${missing.join(", ")}` : ""}${stale.length > 0 ? `; stale: ${stale.join(", ")}` : ""}`,
        );
        process.exit(1);
    }
}
lines.push("", COVERAGE_PLAN.trimEnd());
const text = `${lines.join("\n")}\n`;

if (process.argv.includes("--check")) {
    const committed = existsSync(out) ? readFileSync(out, "utf8") : "";
    if (committed !== text) {
        console.error(
            "✗ examples/AGENTS.md differs from the manifests; run `bun run examples:index`.",
        );
        process.exit(1);
    }
} else {
    writeFileSync(out, text);
}
