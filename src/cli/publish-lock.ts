import {
    lstatSync,
    mkdtempSync,
    readdirSync,
    renameSync,
    rmdirSync,
    rmSync,
    unlinkSync,
    writeFileSync,
} from "node:fs";
import { basename, join } from "node:path";

function code(error: unknown): string | undefined {
    return (error as NodeJS.ErrnoException).code;
}

function removeHolder(lock: string, holder?: string): void {
    if (holder) {
        try {
            unlinkSync(join(lock, holder));
        } catch (error) {
            if (code(error) !== "ENOENT") throw error;
        }
    }
    try {
        rmdirSync(lock);
    } catch (error) {
        if (!["ENOENT", "ENOTEMPTY", "EEXIST"].includes(code(error) ?? "")) throw error;
    }
}

export function acquireFlyPublishLock(project: string): () => void {
    const lock = join(project, ".shallot-fly-publish.lock");
    const staging = mkdtempSync(join(project, ".shallot-fly-publish-"));
    const holder = `${process.pid}-${basename(staging).slice(-6)}`;
    try {
        writeFileSync(join(staging, holder), "");
        for (;;) {
            try {
                // A directory rename cannot replace a nonempty directory. Pass ownership together
                // with the lock, so neither a contender nor a stale cleanup sees an ownerless claim.
                renameSync(staging, lock);
                return () => removeHolder(lock, holder);
            } catch (error) {
                if (!["EEXIST", "ENOTEMPTY", "EPERM"].includes(code(error) ?? "")) throw error;
            }
            let entries: string[];
            try {
                if (!lstatSync(lock).isDirectory())
                    throw new Error(`Invalid Fly publish lock: ${lock}`);
                entries = readdirSync(lock);
            } catch (error) {
                if (code(error) === "ENOENT") continue;
                throw error;
            }
            if (entries.length === 0) {
                removeHolder(lock);
                continue;
            }
            const owner = entries[0]!;
            const pid = Number(/^([1-9]\d*)-[a-zA-Z0-9]{6}$/.exec(owner)?.[1]);
            if (entries.length !== 1 || !Number.isSafeInteger(pid) || pid > 0x7fffffff)
                throw new Error(`Invalid Fly publish lock: ${lock}`);
            try {
                process.kill(pid, 0);
            } catch (error) {
                if (code(error) === "ESRCH") {
                    removeHolder(lock, owner);
                    continue;
                }
                if (code(error) !== "EPERM") throw error;
            }
            throw new Error(
                `Fly publish is already running for this project (PID ${pid}; ${lock}).`,
            );
        }
    } finally {
        rmSync(staging, { recursive: true, force: true });
    }
}
