import { test } from "bun:test";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { compileSubjects, measureCompile } from "./compile.fixture";

await setupGlobals();
for (const subject of compileSubjects) {
    test(`report ${subject.name} compile duration`, async () => {
        console.log(
            JSON.stringify({
                composition: subject.name,
                ...(await measureCompile(subject.config)),
            }),
        );
    });
}
