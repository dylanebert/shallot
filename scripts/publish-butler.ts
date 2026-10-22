import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { butlerPath, ensureButler } from "../src/cli/publish-itch";

/** Tests consume the butler `bun scripts/publish-butler.ts` fetched, never fetch it. */
export function publishTestButler(): string {
    const path = butlerPath();
    if (!existsSync(join(dirname(path), ".complete")))
        throw new Error(`Missing pinned butler: ${path}. Run bun scripts/publish-butler.ts.`);
    return path;
}

if (import.meta.main) console.log(await ensureButler());
