import { readdirSync, renameSync, statSync } from "node:fs";
import { join } from "node:path";

// Fixture sources carry a trailing `.fixture`; materializing strips it so the production reader sees ordinary files.
export function unfixture(dir: string): void {
    for (const entry of readdirSync(dir)) {
        const path = join(dir, entry);
        if (statSync(path).isDirectory()) unfixture(path);
        else if (entry.endsWith(".fixture")) renameSync(path, path.slice(0, -".fixture".length));
    }
}
