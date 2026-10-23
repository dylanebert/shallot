import { expect, test } from "bun:test";
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { runCreate } from "./create";

const examples = resolve(import.meta.dir, "../../examples");
const templates = readdirSync(examples, { withFileTypes: true }).filter(
    (entry) => entry.isDirectory() && existsSync(join(examples, entry.name, "index.html")),
);

for (const { name } of templates) {
    test(`${name}'s generated workflow parses and preserves every job's steps`, () => {
        const scratch = mkdtempSync(join(tmpdir(), "shallot-workflow-"));
        try {
            const destination = join(scratch, name);
            expect(runCreate([destination, "--template", name])).toBe(0);
            const source = readFileSync(join(destination, ".github/workflows/test.yml"), "utf8");
            const workflow = Bun.YAML.parse(source) as {
                jobs: Record<string, { steps: { run?: string; uses?: string }[] }>;
            };
            const components = JSON.parse(readFileSync(join(destination, "package.json"), "utf8"))
                .scripts.typecheck;
            expect(workflow.jobs.cheap!.steps.map((step) => step.run?.trim())).toContain(
                components ? "bun run typecheck" : "bun check",
            );
            const jobs = [
                ...source
                    .split("\njobs:\n")[1]!
                    .matchAll(/^ {2}(\w+):\n([\s\S]*?)(?=^ {2}\w+:|(?![\s\S]))/gm),
            ];
            expect(Object.keys(workflow.jobs)).toEqual(jobs.map((job) => job[1]));
            for (const job of jobs) {
                const steps = workflow.jobs[job[1]!]!.steps;
                expect(steps).toHaveLength([...job[2]!.matchAll(/^ {6}- /gm)].length);
                for (const step of steps) {
                    expect(typeof (step.run ?? step.uses)).toBe("string");
                    if (step.run !== undefined) expect(step.run.trim().length).toBeGreaterThan(0);
                }
            }
        } finally {
            rmSync(scratch, { recursive: true, force: true });
        }
    });
}
