import { resolve } from "path";

const root = resolve(import.meta.dir, "..");
const shallot = await Bun.file(resolve(root, "packages/shallot/package.json")).json();
const create = await Bun.file(resolve(root, "packages/create-shallot/package.json")).json();

const fail = (msg: string) => {
    console.error(msg);
    process.exit(1);
};

if (shallot.version !== create.version) {
    fail(
        `Version mismatch: @dylanebert/shallot@${shallot.version} vs create-shallot@${create.version}`,
    );
}

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
// `rust/window`'s is tracked for reproducible native builds). `rust/tumble` is `publish = false`
// and versions independently of the release.
for (const crate of ["rust/audio/Cargo.toml", "rust/window/Cargo.toml"]) {
    const text = await Bun.file(resolve(root, "packages/shallot", crate)).text();
    const version = text.match(/^version = "(.+)"/m)?.[1];
    if (version !== shallot.version) {
        fail(`Version mismatch: ${crate}@${version} vs @dylanebert/shallot@${shallot.version}`);
    }
}

// `bun.lock` records each workspace package's version independently of its `package.json`, and a
// stale entry survives a release untouched: it read 0.9.0 through the whole 0.9.1 cycle, because
// nothing read it. Bun writes the lockfile with trailing commas, which `JSON.parse` rejects —
// strip them at the boundary (no lockfile string value ends in a comma before a closing brace).
const lockText = await Bun.file(resolve(root, "bun.lock")).text();
const lock = JSON.parse(lockText.replace(/,(\s*[}\]])/g, "$1"));
for (const dir of ["packages/shallot", "packages/create-shallot"]) {
    const version = lock.workspaces?.[dir]?.version;
    if (version !== shallot.version) {
        fail(
            `Version mismatch: bun.lock records ${dir}@${version} vs @dylanebert/shallot@${shallot.version} — re-run \`bun install\`.`,
        );
    }
}

// The docs half of the release checklist, which is the half that slipped in 0.9.1: it published
// and tagged with no changelog entry and a migration guide still targeting 0.9. Neither file is
// read by any other gate, and both are only ever wrong right after a bump — so tying them to the
// current version is what makes the bump-then-document order self-enforcing rather than
// remembered. A missing bump is `--release` below; a *partial* one is the arms above.
const changelog = await Bun.file(resolve(root, "CHANGELOG.md")).text();
const newest = changelog.match(/^## (\d+\.\d+\.\d+)/m)?.[1];
if (newest !== shallot.version) {
    fail(
        `CHANGELOG.md's newest entry is ${newest}, not ${shallot.version} — every release earns its entry.`,
    );
}

// Only the two spans that name the *recommended target* are checked. Prose dating a change to the
// release that shipped it ("ships compiled as of 0.9.1") is a historical fact and must not move,
// so a blanket stale-version sweep over this file would be wrong.
const migration = await Bun.file(resolve(root, "packages/shallot/MIGRATION.md")).text();
const targets = {
    title: migration.match(/^# Migrating from \S+ to (\S+)/)?.[1],
    install: migration.match(/bun add @dylanebert\/shallot@\^(\S+)/)?.[1],
};
for (const [where, version] of Object.entries(targets)) {
    if (version !== shallot.version) {
        fail(
            `MIGRATION.md's ${where} targets ${version}, not ${shallot.version} — retarget the guide at the release it recommends.`,
        );
    }
}

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

// Release-time only: nothing else catches a bump that never happened. Every other arm compares
// version sites to each other, so a whole cycle run against an unbumped tree is uniformly green
// until npm rejects the republish — with the tell (a pack named `…-0.9.0.tgz`) buried in the
// dogfood evidence. A release tags `v<version>` on `main`, so an existing tag means this version
// already shipped. Run before the pack: `bun run scripts/check-versions.ts --release`.
if (process.argv.includes("--release")) {
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
