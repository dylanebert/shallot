import { readdirSync, renameSync, statSync } from "node:fs";
import { join } from "node:path";

// Fixture check files are stored with a trailing `.fixture` so the real discovery and the real
// runner never see them; materializing strips it, giving the production readers a real tree.
export function unfixture(dir: string): void {
    for (const entry of readdirSync(dir)) {
        const path = join(dir, entry);
        if (statSync(path).isDirectory()) unfixture(path);
        else if (entry.endsWith(".fixture")) renameSync(path, path.slice(0, -".fixture".length));
    }
}
