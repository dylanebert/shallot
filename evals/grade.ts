// Grade one eval task's project. Grading drives the task's withheld gate against the running app, and
// that driver is `shallot check`, which this version does not ship yet; until it does this entry
// validates its arguments and the task, reports that, and exits 2.
//
// Run: `bun run evals/grade.ts <task> <projectDir> [--json] [--port <n>]`

import { existsSync } from "node:fs";
import { join, resolve } from "node:path";

const EVALS = import.meta.dir;

const args = process.argv.slice(2);
const portIdx = args.indexOf("--port");
const positional = args.filter((a, i) => !a.startsWith("--") && args[i - 1] !== "--port");
const [task, projectArg] = positional;
if (!task || !projectArg || (portIdx !== -1 && !args[portIdx + 1])) {
    console.error("Usage: bun run evals/grade.ts <task> <projectDir> [--json] [--port <n>]");
    process.exit(1);
}
const project = resolve(projectArg);
const taskGate = join(EVALS, "tasks", task, "gate.ts");
if (!existsSync(taskGate)) throw new Error(`no gate for task ${task} at ${taskGate}`);
if (!existsSync(join(project, "package.json"))) throw new Error(`no project at ${project}`);

console.error("grading needs shallot check, not available in this version");
process.exit(2);
