import { resolve } from "path";

const root = resolve(import.meta.dir, "..");

// The doc set is what Git tracks, not what the filesystem holds. A checkout may contain generated
// or third-party Markdown that is not part of the repository's install instructions.
const tracked = Bun.spawnSync(["git", "ls-files", "-z", "*.md"], { cwd: root });
if (!tracked.success) {
    console.error(
        "✗ `git ls-files` failed — check-pins needs a git checkout to scope its doc set.",
    );
    process.exit(1);
}
const docs = tracked.stdout.toString().split("\0").filter(Boolean);
if (docs.length === 0) {
    console.error(
        "✗ `git ls-files '*.md'` matched nothing — the pin scan would be vacuously green.",
    );
    process.exit(1);
}

// These are the package ranges that installation commands and workspace manifests must agree with.
const PIN_SOURCES: Record<string, { manifest: string; field: string }> = {
    typegpu: { manifest: "package.json", field: "peerDependencies" },
    "unplugin-typegpu": { manifest: "package.json", field: "dependencies" },
    typescript: { manifest: "package.json", field: "devDependencies" },
};

const declared: Record<string, string> = {};
for (const [name, { manifest, field }] of Object.entries(PIN_SOURCES)) {
    const json = await Bun.file(resolve(root, manifest)).json();
    const range = json[field]?.[name];
    if (typeof range !== "string") {
        console.error(
            `✗ ${manifest} declares no ${field}.${name} — update check-pins.ts's PIN_SOURCES.`,
        );
        process.exit(1);
    }
    declared[name] = range;
}

// Match only tracked package names in fenced `bun add` commands. A bare package name has no range
// to disagree with, and a package name embedded in a longer name is not this pin site.
const PIN_RE = new RegExp(`(?<![-\\w])(${Object.keys(declared).join("|")})@(\\S+)`, "g");

type PinDrift = { file: string; line: number; name: string; found: string; want: string };
const drift: PinDrift[] = [];
let scanned = 0;

for (const file of docs) {
    scanned++;
    const lines = (await Bun.file(resolve(root, file)).text()).split("\n");
    let inFence = false;
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        if (line.trim().startsWith("```")) {
            inFence = !inFence;
            continue;
        }
        if (!inFence || !line.trim().startsWith("bun add")) continue;
        for (const [, name, found] of line.matchAll(PIN_RE)) {
            if (found !== declared[name]) {
                drift.push({ file, line: i + 1, name, found, want: declared[name] });
            }
        }
    }
}

// The manifests are pin sites too. Enumerate every tracked manifest so an example carrying an old
// minor cannot silently install a second branded TypeGPU identity.
const DEP_FIELDS = [
    "dependencies",
    "devDependencies",
    "peerDependencies",
    "optionalDependencies",
] as const;

const manifestTracked = Bun.spawnSync(
    ["git", "ls-files", "-z", "package.json", "**/package.json"],
    { cwd: root },
);
if (!manifestTracked.success) {
    console.error(
        "✗ `git ls-files` failed — check-pins needs a git checkout to scope its manifest set.",
    );
    process.exit(1);
}
const manifestFiles = manifestTracked.stdout.toString().split("\0").filter(Boolean);
if (manifestFiles.length === 0) {
    console.error(
        "✗ `git ls-files 'package.json' '**/package.json'` matched nothing — the manifest-pin arm would be vacuously green.",
    );
    process.exit(1);
}

let manifestPkgCount = 0;
for (const file of manifestFiles) {
    const json = await Bun.file(resolve(root, file)).json();
    let declaredInThis = false;
    for (const field of DEP_FIELDS) {
        const deps = json[field];
        if (!deps || typeof deps !== "object") continue;
        for (const [name, range] of Object.entries(deps)) {
            if (!(name in declared)) continue;
            declaredInThis = true;
            if (typeof range !== "string") continue;
            if (range !== declared[name]) {
                drift.push({ file, line: 0, name, found: range, want: declared[name] });
            }
        }
    }
    if (declaredInThis) manifestPkgCount++;
}

if (manifestPkgCount === 0) {
    console.error(
        "✗ no tracked `package.json` declares a tracked package — the manifest-pin arm would be vacuously green.",
    );
    process.exit(1);
}

if (drift.length > 0) {
    console.error(`✗ ${drift.length} pin(s) disagree with the manifests:\n`);
    for (const pin of drift) {
        console.error(`  ${pin.line ? `${pin.file}:${pin.line}` : pin.file}`);
        console.error(`    ${pin.name}@${pin.found} — the manifest declares ${pin.want}`);
    }
    console.error(
        "\nA documented install or a workspace manifest must resolve against the shipped manifests. Bump the pin with the manifest, in the same commit.",
    );
    process.exit(1);
}

console.log(
    `✓ install/manifest pins match the manifests (${scanned} doc(s), ${manifestPkgCount} manifest(s))`,
);
