export interface OracleDeleteAnswer {
    status: number | null;
    stderr: string;
}

/** Failed trials retain their Workers for inspection; successful trials delete only their own names. */
export function cleanupOracleWorkers(
    names: Iterable<string>,
    passed: boolean,
    remove: (name: string) => OracleDeleteAnswer,
    report: (message: string) => void,
): void {
    const failures: string[] = [];
    for (const name of new Set(names)) {
        const command = `wrangler delete --name ${name}`;
        if (!passed) {
            report(`Retained Worker for inspection: ${command}`);
            continue;
        }
        const answer = remove(name);
        if (answer.status === 0) report(`Deleted Worker ${name}.`);
        else {
            report(`Could not delete Worker: ${command}`);
            failures.push(`${name}: ${answer.stderr || `exit ${answer.status}`}`);
        }
    }
    if (failures.length) throw new Error(`Oracle Worker cleanup failed:\n${failures.join("\n")}`);
}
