import { readFileSync } from "node:fs";
import { cargoRuntimeDir, requireRuntimeDir } from "../src/native/provenance.js";

const [messagesPath, marker] = process.argv.slice(2);
if (!messagesPath || !marker) {
    throw new Error("usage: node scripts/native-runtime.mjs <cargo-json> <runtime-marker>");
}
const path = cargoRuntimeDir(readFileSync(messagesPath, "utf8"), marker);
console.log(requireRuntimeDir(path, marker));
