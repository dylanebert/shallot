import { createHash } from "node:crypto";
import {
    copyFileSync,
    existsSync,
    lstatSync,
    mkdirSync,
    readFileSync,
    renameSync,
    rmSync,
    symlinkSync,
    writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";

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
}

export interface Pin {
    url: string;
    dest: string;
    sha256: string;
    bytes: number;
}

export interface Paths {
    cache: string;
    publicDir: string;
}

const ROOT = resolve(import.meta.dir, "..");

export const MANIFEST = join(ROOT, "assets.json");

export const PATHS: Paths = {
    cache: join(process.env.XDG_CACHE_HOME || join(homedir(), ".cache"), "shallot", "assets"),
    publicDir: join(ROOT, "examples", "gym", "public"),
};

export function load(path = MANIFEST): Asset[] {
    return JSON.parse(readFileSync(path, "utf8"));
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

function sha256(path: string): string {
    return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function present(pin: Pin, paths: Paths): boolean {
    const dest = join(paths.publicDir, pin.dest);
    return existsSync(dest) && sha256(dest) === pin.sha256;
}

/** the names of `assets` whose files are absent or fail their hash under `paths.publicDir`. */
export function missing(assets: Asset[], paths: Paths = PATHS): string[] {
    return assets.filter((a) => !pins(a).every((p) => present(p, paths))).map((a) => a.name);
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

/** fetch each absent file of `asset` into the content-addressed cache, verify it, and place it under
 *  `paths.publicDir`. Returns the bytes fetched; 0 means the asset was already present. A cache entry
 *  that fails its hash is deleted and refetched; a download that fails its hash throws. */
export async function fetchAsset(
    asset: Asset,
    paths: Paths = PATHS,
    link: typeof symlinkSync = symlinkSync,
): Promise<number> {
    let fetched = 0;
    for (const pin of pins(asset)) {
        if (present(pin, paths)) continue;
        const cached = join(paths.cache, pin.sha256);
        if (existsSync(cached) && sha256(cached) !== pin.sha256) rmSync(cached);
        if (!existsSync(cached)) {
            await download(pin, cached);
            fetched += pin.bytes;
        }
        place(cached, join(paths.publicDir, pin.dest), link);
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

const mb = (bytes: number) => `${(bytes / 1e6).toFixed(1)} MB`;

async function main(argv: string[]): Promise<number> {
    const check = argv.includes("--check");
    const assets = select(
        load(),
        argv.filter((a) => a !== "--check"),
    );
    if (check) {
        const absent = missing(assets);
        for (const name of absent) console.error(remedy(name));
        if (absent.length === 0) console.log(`assets: ${assets.length} present and verified`);
        return absent.length > 0 ? 1 : 0;
    }
    let total = 0;
    for (const asset of assets) {
        const bytes = await fetchAsset(asset);
        total += bytes;
        if (bytes > 0) console.log(`fetched ${asset.name} (${mb(bytes)})`);
    }
    console.log(
        total === 0
            ? `assets: nothing to fetch, ${assets.length} present and verified`
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
