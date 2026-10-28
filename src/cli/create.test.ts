import { expect, test } from "bun:test";
import { existsSync, readdirSync } from "node:fs";
import { join, resolve } from "node:path";
import { packageManifest, testWorkflow } from "./create";

const repo = resolve(import.meta.dir, "../..");
const examples = join(repo, "examples");
const templates = readdirSync(examples, { withFileTypes: true }).filter(
    (entry) => entry.isDirectory() && existsSync(join(examples, entry.name, "index.html")),
);

for (const { name } of templates) {
    test(`${name}'s generated workflow parses and preserves every job's steps`, () => {
        const template = join(examples, name);
        const source = testWorkflow(template);
        const workflow = Bun.YAML.parse(source) as {
            jobs: Record<string, { steps: { run?: string; uses?: string }[] }>;
        };
        const components = JSON.parse(packageManifest(name, template, repo)).scripts.typecheck;
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
    });
}
