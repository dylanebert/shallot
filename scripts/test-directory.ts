import { rmSync } from "node:fs";

// Node's rm retries these with a 100 ms linear backoff; Bun accepts maxRetries but ignores it.
const TRANSIENT = new Set(["EBUSY", "EMFILE", "ENFILE", "ENOTEMPTY", "EPERM"]);
const RETRIES = 10;

/** Remove a test directory after child processes using it have exited. */
export function removeTestDirectory(
    path: string,
    remove: (path: string) => void = (target) => rmSync(target, { recursive: true, force: true }),
): void {
    // Windows can retain a just-exited child's file locks until after its exit is observed.
    for (let attempt = 1; ; attempt++) {
        try {
            remove(path);
            return;
        } catch (error) {
            const code = (error as NodeJS.ErrnoException).code;
            if (attempt > RETRIES || !code || !TRANSIENT.has(code)) throw error;
            Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, attempt * 100);
        }
    }
}
