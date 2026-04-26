import { resolve } from "path";

const root = resolve(import.meta.dir, "..");
const shallot = await Bun.file(resolve(root, "packages/shallot/package.json")).json();
const create = await Bun.file(resolve(root, "packages/create-shallot/package.json")).json();

if (shallot.version !== create.version) {
    console.error(
        `Version mismatch: @dylanebert/shallot@${shallot.version} vs create-shallot@${create.version}`,
    );
    process.exit(1);
}
