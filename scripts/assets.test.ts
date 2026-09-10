import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { existsSync, lstatSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type Asset, fetchAsset, missing, type Paths, place, remedy } from "./assets";

const body = new TextEncoder().encode("pinned bytes");
const hash = createHash("sha256").update(body).digest("hex");
let hits = 0;
const server = Bun.serve({
    port: 0,
    fetch(req) {
        hits++;
        const path = new URL(req.url).pathname;
        if (path === "/good.bin" || path === "/dir/a.bin") return new Response(body);
        if (path === "/bad.bin") return new Response("tampered");
        return new Response("", { status: 404 });
    },
});
const url = `http://localhost:${server.port}`;
let root = "";
const fresh = (): Paths => {
    const dir = mkdtempSync(join(root, "case-"));
    return { cache: join(dir, "cache"), publicDir: join(dir, "public") };
};
const asset = (name: string, file: string): Asset => ({
    name,
    url: `${url}/${file}`,
    sha256: hash,
    bytes: body.byteLength,
    dest: `${name}/${file}`,
});

beforeAll(() => {
    root = mkdtempSync(join(tmpdir(), "shallot-assets-"));
});
afterAll(() => {
    server.stop(true);
    rmSync(root, { recursive: true, force: true });
});

describe("assets", () => {
    test("a download that fails its hash throws and caches nothing", async () => {
        const paths = fresh();
        const bad = asset("bad", "bad.bin");
        await expect(fetchAsset(bad, paths)).rejects.toThrow(/expected sha256/);
        expect(existsSync(join(paths.cache, hash))).toBe(false);
        expect(existsSync(join(paths.publicDir, bad.dest))).toBe(false);
    });

    test("a second fetch does nothing", async () => {
        const paths = fresh();
        const good = asset("good", "good.bin");
        expect(await fetchAsset(good, paths)).toBe(body.byteLength);
        expect(lstatSync(join(paths.publicDir, good.dest)).isSymbolicLink()).toBe(true);
        const before = hits;
        expect(await fetchAsset(good, paths)).toBe(0);
        expect(hits).toBe(before);
    });

    test("a corrupt cache entry is refetched", async () => {
        const paths = fresh();
        const good = asset("good", "good.bin");
        await fetchAsset(good, paths);
        rmSync(join(paths.publicDir, good.dest));
        writeFileSync(join(paths.cache, hash), "rot");
        expect(await fetchAsset(good, paths)).toBe(body.byteLength);
        expect(readFileSync(join(paths.publicDir, good.dest))).toEqual(Buffer.from(body));
    });

    test("directory entries place each file under dest", async () => {
        const paths = fresh();
        const dir: Asset = {
            name: "dir",
            url: `${url}/dir`,
            dest: "d",
            files: [{ path: "a.bin", sha256: hash, bytes: body.byteLength }],
        };
        await fetchAsset(dir, paths);
        expect(missing([dir], paths)).toEqual([]);
    });

    test("check names the fetch command for anything missing", async () => {
        const paths = fresh();
        const good = asset("good", "good.bin");
        expect(missing([good], paths)).toEqual(["good"]);
        expect(remedy("good")).toBe("missing asset good: run bun run assets good");
        await fetchAsset(good, paths);
        expect(missing([good], paths)).toEqual([]);
        rmSync(join(paths.publicDir, good.dest));
        expect(missing([good], paths)).toEqual(["good"]);
    });

    test("placement copies when symlinks are refused", () => {
        const paths = fresh();
        const src = join(root, "src.bin");
        writeFileSync(src, body);
        const dest = join(paths.publicDir, "x/copy.bin");
        const refuse = () => {
            throw new Error("EPERM");
        };
        expect(place(src, dest, refuse)).toBe("copy");
        expect(lstatSync(dest).isSymbolicLink()).toBe(false);
        expect(readFileSync(dest)).toEqual(Buffer.from(body));
        expect(place(src, dest)).toBe("link");
    });
});
