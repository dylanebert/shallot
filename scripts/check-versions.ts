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

// The Rust crates ship inside the shallot release (audio → bundled WASM, window
// → native host binary), so each tracks the shallot version it builds alongside.
for (const crate of ["rust/audio/Cargo.toml", "rust/window/Cargo.toml"]) {
    const text = await Bun.file(resolve(root, "packages/shallot", crate)).text();
    const version = text.match(/^version = "(.+)"/m)?.[1];
    if (version !== shallot.version) {
        fail(`Version mismatch: ${crate}@${version} vs @dylanebert/shallot@${shallot.version}`);
    }
}
