import { rmSync } from "node:fs";

/** Remove a test directory after child processes using it have exited. */
export function removeTestDirectory(path: string): void {
    // Windows can retain a just-exited child's file locks until after its exit is observed.
    rmSync(path, { recursive: true, force: true, maxRetries: 10 });
}
