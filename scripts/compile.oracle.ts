import { test } from "bun:test";
import { compileSubjects, measureCompile } from "./compile.fixture";

const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();
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
