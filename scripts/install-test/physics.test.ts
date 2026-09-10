import { afterEach, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { archiveMembers, checkMembers } from "./physics";

const owned: string[] = [];
afterEach(() => {
    for (const dir of owned.splice(0)) rmSync(dir, { recursive: true });
});

function fixture(extra?: string): { dir: string; tar: string } {
    const dir = mkdtempSync(join(tmpdir(), "physics-archive-"));
    owned.push(dir);
    mkdirSync(join(dir, "package/src"), { recursive: true });
    writeFileSync(join(dir, "package/package.json"), '{"private":true}');
    writeFileSync(join(dir, "package/src/index.ts"), "export const value = 1;");
    if (extra) writeFileSync(join(dir, extra), "unintended archive member");
    return { dir, tar: join(dir, "candidate.tgz") };
}

function pack(dir: string, tar: string, paths = ["package"]): void {
    const result = Bun.spawnSync(["env", "COPYFILE_DISABLE=1", "tar", "-czf", tar, ...paths], {
        cwd: dir,
        stdout: "pipe",
        stderr: "pipe",
    });
    expect(result.exitCode).toBe(0);
}

const expected = ["package/package.json", "package/src/index.ts"];

test("direct archive inventory admits the complete regular files and their directories", () => {
    const { dir, tar } = fixture();
    pack(dir, tar);
    const members = archiveMembers(tar);
    expect(members.length).toBe(4);
    expect(members.filter(({ type }) => type === "-").length).toBe(2);
    expect(() => checkMembers(members, expected)).not.toThrow();
});

for (const extra of ["package/._package.json", "package/src/._index.ts", "._package"]) {
    test(`direct archive inventory exposes and refuses ${extra}`, () => {
        const { dir, tar } = fixture(extra);
        pack(dir, tar, extra === "._package" ? ["package", extra] : ["package"]);
        const members = archiveMembers(tar);
        expect(members.find(({ path }) => path === extra)?.type).toBe("-");
        expect(() => checkMembers(members, expected)).toThrow("platform metadata");
    });
}

test("an ordinary extra member is not hidden by a metadata-only rule", () => {
    const { dir, tar } = fixture("package/unintended.txt");
    pack(dir, tar);
    expect(() => checkMembers(archiveMembers(tar), expected)).toThrow("regular member population");
});

test("a missing regular member cannot shrink acceptance", () => {
    const { dir, tar } = fixture();
    rmSync(join(dir, "package/src/index.ts"));
    pack(dir, tar);
    expect(() => checkMembers(archiveMembers(tar), expected)).toThrow("regular member population");
});

test("duplicate members survive inventory and refuse", () => {
    const { dir, tar } = fixture();
    pack(dir, tar, ["package", "package/src/index.ts"]);
    const members = archiveMembers(tar);
    expect(members.filter(({ path }) => path === "package/src/index.ts").length).toBe(2);
    expect(() => checkMembers(members, expected)).toThrow("duplicate member");
});

test("a link cannot masquerade as the required regular member", () => {
    const { dir, tar } = fixture();
    rmSync(join(dir, "package/src/index.ts"));
    symlinkSync("../package.json", join(dir, "package/src/index.ts"));
    pack(dir, tar);
    expect(() => checkMembers(archiveMembers(tar), expected)).toThrow("unexpected type");
});

test("an extra empty directory is still an unintended archive member", () => {
    const { dir, tar } = fixture();
    mkdirSync(join(dir, "package/extra"));
    pack(dir, tar);
    expect(() => checkMembers(archiveMembers(tar), expected)).toThrow("unexpected directory");
});
