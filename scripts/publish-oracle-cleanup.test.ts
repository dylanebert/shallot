import { expect, test } from "bun:test";
import { cleanupOracleWorkers } from "./publish-oracle-cleanup";

for (const passed of [true, false]) {
    test(`a ${passed ? "successful" : "failed"} trial ${passed ? "deletes" : "retains"} its Workers`, () => {
        const deleted: string[] = [];
        const reports: string[] = [];
        cleanupOracleWorkers(
            ["trial", "collision", "trial"],
            passed,
            "account-a",
            (name) => {
                deleted.push(name);
                return { status: 0, stderr: "" };
            },
            () => false,
            (message) => reports.push(message),
        );
        expect(deleted).toEqual(passed ? ["trial", "collision"] : []);
        expect(reports).toEqual(
            passed
                ? ["Deleted Worker trial.", "Deleted Worker collision."]
                : [
                      "Retained Worker for inspection: CLOUDFLARE_ACCOUNT_ID=account-a wrangler delete --name trial",
                      "Retained Worker for inspection: CLOUDFLARE_ACCOUNT_ID=account-a wrangler delete --name collision",
                  ],
        );
    });
}

test("an unresolved account never prints or executes an unbound delete", () => {
    const reports: string[] = [];
    cleanupOracleWorkers(
        ["trial"],
        false,
        undefined,
        () => {
            throw new Error("must not delete");
        },
        () => {
            throw new Error("must not check another account");
        },
        (message) => reports.push(message),
    );
    expect(reports).toEqual([
        "Retained Worker trial: account unresolved; resolve its account before deleting.",
    ]);
});

test("a failed deletion reports the command, attempts remaining names and fails the trial", () => {
    const deleted: string[] = [];
    const reports: string[] = [];
    expect(() =>
        cleanupOracleWorkers(
            ["trial", "collision"],
            true,
            "account-a",
            (name) => {
                deleted.push(name);
                return name === "trial"
                    ? { status: 1, stderr: "permission denied" }
                    : { status: 0, stderr: "" };
            },
            (name) => name === "trial",
            (message) => reports.push(message),
        ),
    ).toThrow("trial: Worker still exists; permission denied");
    expect(deleted).toEqual(["trial", "collision"]);
    expect(reports).toEqual([
        "Could not delete Worker: CLOUDFLARE_ACCOUNT_ID=account-a wrangler delete --name trial",
        "Deleted Worker collision.",
    ]);
});

test("an absent Worker is deleted even when Wrangler's legacy KV cleanup fails", () => {
    const reports: string[] = [];
    const checked: string[] = [];
    const error = "storage/kv/namespaces: Authentication error [code: 10000]";
    cleanupOracleWorkers(
        ["trial"],
        true,
        "account-a",
        () => ({ status: 1, stderr: error }),
        (name) => {
            checked.push(name);
            return false;
        },
        (message) => reports.push(message),
    );
    expect(checked).toEqual(["trial"]);
    expect(reports).toEqual([
        "Deleted Worker trial.",
        `Wrangler reported an error after Worker trial was deleted: ${error}`,
    ]);
});

test("a successful delete exit does not prove absence", () => {
    expect(() =>
        cleanupOracleWorkers(
            ["trial"],
            true,
            "account-a",
            () => ({ status: 0, stderr: "" }),
            () => true,
            () => {},
        ),
    ).toThrow("trial: Worker still exists");
});

test("an inconclusive absence check fails and still attempts the remaining Workers", () => {
    const deleted: string[] = [];
    const reports: string[] = [];
    expect(() =>
        cleanupOracleWorkers(
            ["trial", "collision"],
            true,
            "account-a",
            (name) => {
                deleted.push(name);
                return { status: 0, stderr: "" };
            },
            (name) => {
                if (name === "trial") throw new Error("versions list: authentication error");
                return false;
            },
            (message) => reports.push(message),
        ),
    ).toThrow("trial: versions list: authentication error");
    expect(deleted).toEqual(["trial", "collision"]);
    expect(reports).toEqual([
        "Could not verify Worker deletion: CLOUDFLARE_ACCOUNT_ID=account-a wrangler delete --name trial",
        "Deleted Worker collision.",
    ]);
});
