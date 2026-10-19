import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import semver from "semver";

export function changelogReleaseAllowed(
    baseVersion: string,
    headVersion: string,
    changelogChanged: boolean,
): boolean {
    return (
        !changelogChanged ||
        (baseVersion !== headVersion &&
            semver.valid(headVersion) !== null &&
            semver.prerelease(headVersion) === null)
    );
}

if (import.meta.main) {
    const root = resolve(import.meta.dir, "..");
    const git = (...args: string[]) => execFileSync("git", args, { cwd: root, encoding: "utf8" });
    let baseCommit: string;
    try {
        baseCommit = git("merge-base", "HEAD", "origin/main").trim();
    } catch {
        console.error(
            "CHANGELOG.md gate: no common base with origin/main; run `git fetch origin main` (and `git fetch --unshallow` in a shallow clone), then rerun.",
        );
        process.exit(1);
    }
    const base = JSON.parse(git("show", `${baseCommit}:package.json`));
    const head = await Bun.file(resolve(root, "package.json")).json();
    const changed = git("diff", "--name-only", baseCommit, "--", "CHANGELOG.md").trim() !== "";
    if (!changelogReleaseAllowed(base.version, head.version, changed)) {
        console.error(
            "CHANGELOG.md: the release author writes one entry per published release, at the release. A change since this branch's base on origin/main requires package.json's version to change to a release without a prerelease tag. See CONTRIBUTING.md (Dependencies and releases).",
        );
        process.exitCode = 1;
    }
}
