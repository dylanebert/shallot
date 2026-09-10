import { resolve } from "path";

const root = resolve(import.meta.dir, "..");
const shallot = await Bun.file(resolve(root, "package.json")).json();
const create = await Bun.file(resolve(root, "packages/create-shallot/package.json")).json();
const release = process.argv.includes("--release");

const fail = (msg: string) => {
    console.error(msg);
    process.exit(1);
};

if (shallot.version !== create.version) {
    fail(
        `Version mismatch: @dylanebert/shallot@${shallot.version} vs create-shallot@${create.version}`,
    );
}

// Release-time only: nothing else catches a bump that never happened. Every other arm compares
// version sites to each other, so a whole cycle run against an unbumped tree is uniformly green
// until npm rejects the republish — with the tell (a pack named `…-0.9.0.tgz`) buried in the
// dogfood evidence. A release tags `v<version>` on `main`, so an existing tag means this version
// already shipped. Run before the pack: `bun run scripts/check-versions.ts --release`.
if (release) {
    const tag = `v${shallot.version}`;
    const found = Bun.spawnSync(["git", "tag", "--list", tag], { cwd: root });
    if (found.exitCode !== 0) {
        const why = found.exitCode === null ? "spawn failed" : `exit ${found.exitCode}`;
        fail(`git tag --list failed (${why}) — cannot verify ${tag} is untagged.`);
    }
    if (found.stdout.toString().trim() !== "") {
        fail(`${tag} is already tagged — ${shallot.version} shipped; bump before packing.`);
    }
}

const solver = await Bun.file(resolve(root, "packages/shallot-physics/package.json")).json();
if (solver.version !== shallot.version || solver.private !== true)
    fail("private solver/distribution version mismatch");
if (
    Object.keys(solver.dependencies ?? {}).length ||
    Object.keys(solver.peerDependencies ?? {}).length
)
    fail("solver must remain dependency-free");

// Runtime dependencies must resolve to a PUBLISHED version. A `link:` / `file:` / `workspace:`
// protocol (handy for local co-development) survives verbatim into the published tarball and is
// unresolvable for an npm consumer — it silently broke the default physics backend once.
// devDependencies are exempt (never shipped).
for (const [name, range] of Object.entries(shallot.dependencies ?? {})) {
    if (typeof range === "string" && /^(link|file|workspace):/.test(range)) {
        fail(
            `@dylanebert/shallot depends on ${name} via "${range}" — a local protocol can't publish; pin a released version.`,
        );
    }
}

// The Rust crates ship inside the shallot release (audio → bundled WASM, window
// → native host binary), so each tracks the shallot version it builds alongside.
// Their `Cargo.lock`s are build output, not version sites — cargo rewrites the
// own-package entry from the manifest on the next build (`rust/audio`'s is gitignored;
// `rust/window`'s is tracked for reproducible native builds). `rust/physics` is `publish = false`
// and versions independently of the release.
for (const crate of ["rust/audio/Cargo.toml", "rust/window/Cargo.toml"]) {
    const text = await Bun.file(resolve(root, crate)).text();
    const version = text.match(/^version = "(.+)"/m)?.[1];
    if (version !== shallot.version) {
        fail(`Version mismatch: ${crate}@${version} vs @dylanebert/shallot@${shallot.version}`);
    }
}

// `bun.lock` records each member package's version independently of its `package.json`, and a
// stale entry survives a release untouched: it read 0.9.0 through the whole 0.9.1 cycle, because
// nothing read it. The root entry carries no version. Bun writes the lockfile with trailing commas,
// which `JSON.parse` rejects — strip them at the boundary (no lockfile string value ends in a comma
// before a closing brace).
const lockText = await Bun.file(resolve(root, "bun.lock")).text();
const lock = JSON.parse(lockText.replace(/,(\s*[}\]])/g, "$1"));
for (const dir of ["packages/shallot-physics", "packages/create-shallot"]) {
    const version = lock.workspaces?.[dir]?.version;
    if (version !== shallot.version) {
        fail(
            `Version mismatch: bun.lock records ${dir}@${version} vs @dylanebert/shallot@${shallot.version} — re-run \`bun install\`.`,
        );
    }
}

// The docs half of the release checklist, which is the half that slipped in 0.9.1: it published
// and tagged with no changelog entry. Between releases the top section is `Unreleased`; a release
// renames it to the version, so `--release` refuses anything but the version on top.
const changelog = await Bun.file(resolve(root, "CHANGELOG.md")).text();
const top = changelog.match(/^## (\S+)/m)?.[1];
if (top !== shallot.version && (release || top !== "Unreleased")) {
    fail(
        `CHANGELOG.md's top section is ${top}, not ${release ? shallot.version : `${shallot.version} or Unreleased`} — every release earns its entry.`,
    );
}

// MIGRATION.md is not a version site: its title and install line name the minor the guide targets, and a micro
// never changes what the guide says. Prose dating a change to the release that shipped it
// ("ships compiled as of 0.9.1") is a historical fact and must not move either.

// `README.md`'s demo table links each showcase's source at `tree/v<version>` — a hand-written tag
// per row, correct today and drift-by-construction on every bump: derived from nothing, so no
// gate read them before this arm. The arm asserts every version-bearing README link agrees
// with the package version.
const readme = await Bun.file(resolve(root, "README.md")).text();
const tagLinkRe = /dylanebert\/shallot\/tree\/v(\d+\.\d+\.\d+)\b/g;
let readmeLinkCount = 0;
for (const m of readme.matchAll(tagLinkRe)) {
    readmeLinkCount++;
    if (m[1] !== shallot.version) {
        fail(
            `README.md carries a \`tree/v${m[1]}\` link, not \`tree/v${shallot.version}\` — retag the demo-table source links at the current release.`,
        );
    }
}
if (readmeLinkCount === 0) {
    fail(
        "README.md carries no `tree/v<version>` links — the README-link arm would be vacuously green.",
    );
}
