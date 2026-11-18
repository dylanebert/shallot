import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";

const source = readFileSync(new URL("../src/cli/publish.oracle.ts", import.meta.url), "utf8");

test("personal Fly idle confirms closed game tabs before either idle clock starts", () => {
    const idle = source.slice(
        source.indexOf('clients("hot");'),
        source.indexOf("const stoppedAt ="),
    );
    expect(idle).toMatch(
        /if \(personal\) \{\s+await confirmTrial\(\s+null,\s+"Close every tab of the published game\.[^"]*WebSocket[^"]*",\s+"Are all published game tabs closed\? \[y\/N\] ",\s*\);/,
    );
    const confirmed = idle.indexOf("Are all published game tabs closed?");
    expect(idle.indexOf("const idleStartedAt = Date.now()")).toBeGreaterThan(confirmed);
    expect(idle.indexOf("const idle = performance.now()")).toBeGreaterThan(confirmed);
    expect(idle).toContain("assertFlySocketsClosed(socketObservations)");
});

test("Fly idle announces the several-minute wait once and silences only personal poll commands", () => {
    const idle = source.slice(
        source.indexOf('clients("hot");'),
        source.indexOf("const stoppedAt ="),
    );
    expect(idle.match(/Waiting for Fly[^"\n]*several minutes/g)).toHaveLength(1);
    expect(idle.indexOf("Waiting for Fly")).toBeLessThan(idle.indexOf("while (idleMachine"));
    expect(idle.match(/idleMachine = machine\(personal\)/g)).toHaveLength(2);
    expect(source).toContain("const machine = (quiet = false) =>");
    expect(source).toContain('fly(["machine", "list", "--app", name, "--json"], quiet)');
    expect(source).toContain("const fly = (args: string[], quiet = false) =>");
    expect(source).toContain("run([cli, ...args], root, false, quiet)");
    expect(source).toContain("allowFailure = false, quiet = false");
    expect(source).toMatch(/if \(!quiet\) console\.info\(`\$ \$\{cmd\.join\(" "\)\}`\)/);
});
