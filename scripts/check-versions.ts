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
for (const crate of ["rust/audio/Cargo.toml", "rust/window/Cargo.toml"]) {
    const text = await Bun.file(resolve(root, "packages/shallot", crate)).text();
    const version = text.match(/^version = "(.+)"/m)?.[1];
    if (version !== shallot.version) {
        fail(`Version mismatch: ${crate}@${version} vs @dylanebert/shallot@${shallot.version}`);
    }
}
