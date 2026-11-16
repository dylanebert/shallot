import { test } from "bun:test";
import { randomBytes } from "node:crypto";
import { resolve } from "node:path";
import { bunChild } from "./bun-child";

const root = resolve(import.meta.dir, "..");
const git = (...args: string[]) => {
    const result = Bun.spawnSync(["git", ...args], { cwd: root });
    if (result.exitCode !== 0) throw new Error(result.stderr.toString());
    return result.stdout.toString().trim();
};
const u32 = (text: string) => {
    const value = Number(text);
    if (!/^\d+$/.test(text) || !Number.isSafeInteger(value) || value < 0 || value > 0xffffffff)
        throw new Error("Seed must be a decimal u32");
    return value;
};

test("search existing multiplayer invariants over seeded simulated links", async () => {
    const commit = git("rev-parse", "HEAD");
    if (git("status", "--porcelain"))
        throw new Error("Search requires a clean worktree for seed + commit replay");
    if (process.env.SHALLOT_SEARCH_COMMIT && process.env.SHALLOT_SEARCH_COMMIT !== commit)
        throw new Error(
            `Replay requires commit=${process.env.SHALLOT_SEARCH_COMMIT}; current=${commit}`,
        );
    const replay = process.env.SHALLOT_SEARCH_SEED;
    const budget = replay === undefined ? Number(process.env.SHALLOT_SEARCH_BUDGET ?? 20) : 1;
    if (!Number.isSafeInteger(budget) || budget < 1)
        throw new Error("Budget must be a positive integer");
    console.info(
        "Manual oracle; gates nothing. Simulated links, not socket impairment or coverage proof.",
    );
    console.info(
        "Invariants: replication confirmed rows/order, eid references/reuse, disclosure, late join (loss/duplication disabled after tick 79, drain through tick 349); example predicted lift rider (100 ms / 2% loss) motion, authoritative gap error < 0.08 and final grounded state (existing replication.test.ts and lift.node.ts assertions).",
    );
    console.info(
        `budget=${budget} seeds; one seed controls configuration and link schedules per child run`,
    );
    let ran = 0;
    for (; ran < budget; ) {
        const seed = replay === undefined ? randomBytes(4).readUInt32LE() : u32(replay);
        let state = seed;
        const random = () => {
            state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
            return state / 0x100000000;
        };
        const configuration = {
            seed,
            replication: {
                minDelay: 1,
                maxDelay: 4 + Math.floor(random() * 9),
                loss: 0.1 + random() * 0.1,
                duplication: 0.25 + random() * 0.2,
            },
        };
        console.info(
            `seed=${seed} commit=${commit} run=${ran + 1}/${budget} configuration=${JSON.stringify(configuration)}`,
        );
        const child = Bun.spawn({
            ...bunChild([
                "test",
                "./src/extras/multiplayer/replication.test.ts",
                "./examples/multiplayer/src/lift.node.ts",
            ]),
            cwd: root,
            env: {
                ...process.env,
                SHALLOT_MULTIPLAYER_SEED: String(seed),
                SHALLOT_MULTIPLAYER_SEARCH: JSON.stringify(configuration),
            },
            stdout: "inherit",
            stderr: "inherit",
        });
        const code = await child.exited;
        ran++;
        if (code !== 0) {
            console.info(
                `RED seed=${seed} commit=${commit}; seeds ran=${ran}/${budget}; child exit=${code}`,
            );
            console.info(
                `Replay: git checkout ${commit} && SHALLOT_SEARCH_COMMIT=${commit} SHALLOT_SEARCH_SEED=${seed} bun test ./scripts/multiplayer-search.oracle.ts`,
            );
            throw new Error(`Multiplayer search red seed=${seed} commit=${commit}`);
        }
    }
    console.info(
        `GREEN commit=${commit}; seeds ran=${ran}/${budget}; all listed invariants passed`,
    );
}, 0);
