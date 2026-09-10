import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

interface ReferencePin {
    commit?: unknown;
}

interface PinCheck {
    declared: string;
    recorded: string;
}

function readJson(path: string): ReferencePin {
    return JSON.parse(readFileSync(path, "utf8")) as ReferencePin;
}

/** Read the Box3D pin and the commit recorded beside the C-generated solver fixtures. */
export function readReferencePin(root: string): PinCheck {
    const declaredPath = resolve(root, "crates/physics/reference.json");
    const recordedPath = resolve(root, "src/standard/physics/solver/fixtures/reference-pin.json");
    if (!existsSync(declaredPath)) {
        throw new Error(`reference pin missing: ${declaredPath}`);
    }
    if (!existsSync(recordedPath)) {
        throw new Error(`fixture reference pin missing: ${recordedPath}`);
    }

    const declared = String(readJson(declaredPath).commit ?? "");
    const recorded = String(readJson(recordedPath).commit ?? "");
    if (!/^[0-9a-f]{40}$/.test(declared)) {
        throw new Error(`invalid reference commit in ${declaredPath}: ${declared || "<empty>"}`);
    }
    if (!/^[0-9a-f]{40}$/.test(recorded)) {
        throw new Error(
            `invalid fixture reference commit in ${recordedPath}: ${recorded || "<empty>"}`,
        );
    }
    if (declared !== recorded) {
        throw new Error(
            `reference pin mismatch: ${declaredPath} declares ${declared}, but ${recordedPath} records ${recorded}`,
        );
    }
    return { declared, recorded };
}

if (import.meta.main) {
    const args = Bun.argv.slice(2);
    const rootIndex = args.indexOf("--root");
    const root = resolve(
        rootIndex === -1 ? resolve(import.meta.dir, "..") : (args[rootIndex + 1] ?? "."),
    );
    try {
        const { declared } = readReferencePin(root);
        console.log(`reference pin ${declared} matches solver fixtures`);
    } catch (error) {
        console.error((error as Error).message);
        process.exit(1);
    }
}
