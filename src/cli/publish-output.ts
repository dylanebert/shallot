import { format } from "node:util";

/** Publish reserves stdout for the play link; progress and provider reports go to stderr. */
export function publishProgress(...values: unknown[]): void {
    process.stderr.write(`${format(...values)}\n`);
}
