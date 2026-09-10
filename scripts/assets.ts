import { createHash } from "node:crypto";
import {
    copyFileSync,
    existsSync,
    lstatSync,
    mkdirSync,
    readdirSync,
    readFileSync,
    renameSync,
    rmSync,
    symlinkSync,
    writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { normalize } from "../src/project/manifest";
import { compose, DARK, fromBlocks, MARK, toSvg } from "../src/standard/loading/mark";
import { toPng } from "./png";

// Each asset links under the `public/` of every example whose `shallot.json` names it in `assets`, so
// a recipe copied out by `shallot add` carries both its files and the declaration that fetches them.
// An asset no example declares is fetched into the cache only, for the generators.

/** one pinned asset in `assets.json`: a single file (`sha256`/`bytes`, `url` and `dest` name the file)
 *  or a directory (`files`, `url` and `dest` name the directory each `path` joins). `dest` is relative to
 *  the consumer's public directory. */
export interface Asset {
    name: string;
    url: string;
    dest: string;
    sha256?: string;
    bytes?: number;
    files?: { path: string; sha256: string; bytes: number }[];
    /** SPDX identifier, for a third-party asset whose license travels with its pin. */
    license?: string;
}

export interface Pin {
    url: string;
    dest: string;
    sha256: string;
    bytes: number;
}

export interface Paths {
    cache: string;
    /** one subdirectory per example, each declaring its assets in `shallot.json`. */
    examples: string;
}

/** one example that loads fetched assets: its `public/` and the `assets.json` names it declares. */
export interface Consumer {
    name: string;
    publicDir: string;
    assets: string[];
}

const ROOT = resolve(import.meta.dir, "..");

export const MANIFEST = join(ROOT, "assets.json");

export const PATHS: Paths = {
    cache: join(process.env.XDG_CACHE_HOME || join(homedir(), ".cache"), "shallot", "assets"),
    examples: join(ROOT, "examples"),
};

export function load(path = MANIFEST): Asset[] {
    return JSON.parse(readFileSync(path, "utf8"));
}

/** every example that declares at least one asset, by directory name. */
export function consumers(paths: Paths = PATHS): Consumer[] {
    if (!existsSync(paths.examples)) return [];
    return readdirSync(paths.examples)
        .sort()
        .flatMap((name) => {
            const manifest = join(paths.examples, name, "shallot.json");
            if (!existsSync(manifest)) return [];
            const assets = normalize(readFileSync(manifest, "utf8")).assets ?? [];
            if (assets.length === 0) return [];
            return [{ name, publicDir: join(paths.examples, name, "public"), assets }];
        });
}

/** the files an asset pins, flattened to one shape. */
export function pins(asset: Asset): Pin[] {
    if (!asset.files) {
        return [
            {
                url: asset.url,
                dest: asset.dest,
                sha256: asset.sha256 ?? "",
                bytes: asset.bytes ?? 0,
            },
        ];
    }
    return asset.files.map((f) => ({
        url: `${asset.url}/${f.path.split("/").map(encodeURIComponent).join("/")}`,
        dest: `${asset.dest}/${f.path}`,
        sha256: f.sha256,
        bytes: f.bytes,
    }));
}

export function remedy(name: string): string {
    return `missing asset ${name}: run bun run assets ${name}`;
}

/** the verified cache path of single-file asset `name`, for generators that read an asset directly.
 *  Throws the fetch remedy when the entry is absent or fails its hash. */
export function cached(name: string, paths: Paths = PATHS): string {
    const asset = load().find((a) => a.name === name);
    if (!asset || asset.files) throw new Error(`no single-file asset ${name} in assets.json`);
    const path = join(paths.cache, asset.sha256 ?? "");
    if (!existsSync(path) || sha256(path) !== asset.sha256) throw new Error(remedy(name));
    return path;
}

/** the verified cache path of each file of directory asset `name`, keyed by its path within the
 *  asset. Throws the fetch remedy when any file is absent or fails its hash. */
export function cachedFiles(name: string, paths: Paths = PATHS): Map<string, string> {
    const asset = load().find((a) => a.name === name);
    if (!asset?.files) throw new Error(`no directory asset ${name} in assets.json`);
    const out = new Map<string, string>();
    for (const file of asset.files) {
        const path = join(paths.cache, file.sha256);
        if (!existsSync(path) || sha256(path) !== file.sha256) throw new Error(remedy(name));
        out.set(file.path, path);
    }
    return out;
}

function sha256(path: string): string {
    return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function present(pin: Pin, publicDir: string): boolean {
    const dest = join(publicDir, pin.dest);
    return existsSync(dest) && sha256(dest) === pin.sha256;
}

/** each declared placement of `assets` whose files are absent or fail their hash. */
export function missing(
    assets: Asset[],
    paths: Paths = PATHS,
): { consumer: string; name: string }[] {
    return consumers(paths).flatMap((c) =>
        assets
            .filter((a) => c.assets.includes(a.name))
            .filter((a) => !pins(a).every((p) => present(p, c.publicDir)))
            .map((a) => ({ consumer: c.name, name: a.name })),
    );
}

/** symlink `dest` to `src`, copying when the filesystem refuses links. Returns how it placed it. */
export function place(
    src: string,
    dest: string,
    link: typeof symlinkSync = symlinkSync,
): "link" | "copy" {
    mkdirSync(dirname(dest), { recursive: true });
    if (existsSync(dest) || isLink(dest)) rmSync(dest);
    try {
        link(src, dest);
        return "link";
    } catch {
        copyFileSync(src, dest);
        return "copy";
    }
}

function isLink(path: string): boolean {
    try {
        return lstatSync(path).isSymbolicLink();
    } catch {
        return false;
    }
}

async function download(pin: Pin, cached: string): Promise<void> {
    const res = await fetch(pin.url);
    if (!res.ok) throw new Error(`${pin.url}: HTTP ${res.status}`);
    const body = new Uint8Array(await res.arrayBuffer());
    const hash = createHash("sha256").update(body).digest("hex");
    if (hash !== pin.sha256 || body.byteLength !== pin.bytes) {
        throw new Error(
            `${pin.url}: expected sha256 ${pin.sha256} (${pin.bytes} bytes), got ${hash} (${body.byteLength} bytes)`,
        );
    }
    mkdirSync(dirname(cached), { recursive: true });
    const tmp = `${cached}.${process.pid}.tmp`;
    writeFileSync(tmp, body);
    renameSync(tmp, cached);
}

/** fetch each file of `asset` into the content-addressed cache, verify it, and place it under each of
 *  `publicDirs`. Returns the bytes fetched; 0 means the cache already held it. A cache entry that fails
 *  its hash is deleted and refetched; a download that fails its hash throws. */
export async function fetchAsset(
    asset: Asset,
    publicDirs: string[],
    paths: Paths = PATHS,
    link: typeof symlinkSync = symlinkSync,
): Promise<number> {
    let fetched = 0;
    for (const pin of pins(asset)) {
        const cached = join(paths.cache, pin.sha256);
        if (existsSync(cached) && sha256(cached) !== pin.sha256) rmSync(cached);
        if (!existsSync(cached)) {
            await download(pin, cached);
            fetched += pin.bytes;
        }
        for (const dir of publicDirs) {
            if (!present(pin, dir)) place(cached, join(dir, pin.dest), link);
        }
    }
    return fetched;
}

function select(all: Asset[], names: string[]): Asset[] {
    const unknown = names.filter((n) => !all.some((a) => a.name === n));
    if (unknown.length > 0) {
        throw new Error(
            `unknown asset ${unknown.join(", ")}; known: ${all.map((a) => a.name).join(", ")}`,
        );
    }
    return names.length === 0 ? all : all.filter((a) => names.includes(a.name));
}

// Every default icon a shallot project ships is a render of the one bitmap mark, so the shape
// can't drift between the boot splash, a scaffolded project's favicon and the native window.
const NATIVE_ICON = "assets/icon-1024.png";
// The native icon is opaque and square: the 12×14 mark centred in a 16×16 field on the dark ground,
// 64 device pixels a cell. A window manager scales it down; the frame keeps the mark off the edge.
const FRAME = 16;
const NATIVE_SCALE = 64;

function icons(): number {
    const mark = fromBlocks(MARK.m);
    const svg = `${toSvg(mark, DARK, 1)}\n`;
    const tracked = Bun.spawnSync(["git", "ls-files", "-z", "examples"], { cwd: ROOT });
    if (!tracked.success) throw new Error("`git ls-files` failed");
    const targets = tracked.stdout
        .toString()
        .split("\0")
        .filter((file) => file.endsWith("/public/icon.svg"));
    if (targets.length === 0) throw new Error("no example icons; the write is empty");
    for (const file of targets) writeFileSync(resolve(ROOT, file), svg);
    const width = mark[0]?.length ?? 0;
    const framed = compose(FRAME, FRAME, [
        {
            grid: mark,
            x: Math.floor((FRAME - width) / 2),
            y: Math.floor((FRAME - mark.length) / 2),
        },
    ]);
    writeFileSync(resolve(ROOT, NATIVE_ICON), toPng(framed, DARK, NATIVE_SCALE, DARK.bg));
    return targets.length;
}

const mb = (bytes: number) => `${(bytes / 1e6).toFixed(1)} MB`;

async function main(argv: string[]): Promise<number> {
    if (argv.includes("--icons")) {
        console.log(`assets: wrote ${icons()} example icons and ${NATIVE_ICON}`);
        return 0;
    }
    const check = argv.includes("--check");
    const all = load();
    const assets = select(
        all,
        argv.filter((a) => a !== "--check"),
    );
    const users = consumers();
    const unknown = users.flatMap((c) =>
        c.assets
            .filter((n) => !all.some((a) => a.name === n))
            .map((n) => `examples/${c.name}/shallot.json: unknown asset ${n}`),
    );
    for (const line of unknown) console.error(line);
    if (unknown.length > 0) return 1;
    if (check) {
        const placements = users.reduce(
            (n, c) => n + assets.filter((a) => c.assets.includes(a.name)).length,
            0,
        );
        const absent = missing(assets);
        for (const { consumer, name } of absent)
            console.error(`examples/${consumer}: ${remedy(name)}`);
        if (absent.length === 0)
            console.log(
                `assets: ${placements} placement(s) across ${users.length} example(s) present and verified`,
            );
        return absent.length > 0 ? 1 : 0;
    }
    let total = 0;
    for (const asset of assets) {
        const dirs = users.filter((c) => c.assets.includes(asset.name)).map((c) => c.publicDir);
        const bytes = await fetchAsset(asset, dirs);
        total += bytes;
        if (bytes > 0) console.log(`fetched ${asset.name} (${mb(bytes)})`);
    }
    console.log(
        total === 0
            ? `assets: nothing to fetch, ${assets.length} cached and placed`
            : `assets: fetched ${mb(total)}`,
    );
    return 0;
}

if (import.meta.main) {
    try {
        process.exit(await main(process.argv.slice(2)));
    } catch (e) {
        console.error(`assets: ${(e as Error).message}`);
        process.exit(1);
    }
}
