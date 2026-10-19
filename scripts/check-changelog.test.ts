import { expect, test } from "bun:test";
import { changelogReleaseAllowed } from "./check-changelog";

test("a changelog edit requires a version change", () => {
    expect(changelogReleaseAllowed("0.10.0-next.7", "0.10.0-next.7", true)).toBe(false);
    expect(changelogReleaseAllowed("0.10.0", "0.10.0", true)).toBe(false);
});

test("a prerelease bump cannot carry a changelog edit", () => {
    expect(changelogReleaseAllowed("0.10.0-next.7", "0.10.0-next.8", true)).toBe(false);
});

test("an invalid version cannot carry a changelog edit", () => {
    expect(changelogReleaseAllowed("0.10.0-next.7", "not-a-version", true)).toBe(false);
});

test("a release bump can carry a changelog edit", () => {
    expect(changelogReleaseAllowed("0.10.0-next.7", "0.10.0", true)).toBe(true);
    expect(changelogReleaseAllowed("0.10.0", "0.10.1", true)).toBe(true);
});

test("no changelog edit requires no release bump", () => {
    expect(changelogReleaseAllowed("0.10.0-next.7", "0.10.0-next.7", false)).toBe(true);
    expect(changelogReleaseAllowed("0.10.0-next.7", "0.10.0-next.8", false)).toBe(true);
});
