import { resolve, basename, join } from "node:path";
import {
    existsSync,
    mkdirSync,
    cpSync,
    rmSync,
    statSync,
    readdirSync,
    readFileSync,
    writeFileSync,
    chmodSync,
} from "node:fs";
import { execSync } from "node:child_process";
import { buildWeb } from "./build";

const RUST_CRATE = resolve(import.meta.dir, "../rust/window");
const DEFAULT_ICON = resolve(import.meta.dir, "../../../assets/icon-1024.png");
const WIN_TARGET = "x86_64-pc-windows-msvc";
const MAC_TARGET = "aarch64-apple-darwin";
const LINUX_TARGET = "x86_64-unknown-linux-gnu";

function cargoTarget(target: string, release: boolean): string {
    const profile = release ? "release" : "debug";
    return resolve(
        RUST_CRATE,
        `target/${target}/${profile}/shallot-window${target.includes("windows") ? ".exe" : ""}`,
    );
}

function cargoBuild(target: string, release: boolean): void {
    const cmd = target.includes("msvc") ? "cargo xwin build" : "cargo build";
    const args = [cmd, "--target", target];
    if (release) args.push("--release");
    execSync(args.join(" "), { cwd: RUST_CRATE, stdio: "inherit" });
}

function tarHeader(name: string, size: number, type: string): Buffer {
    const buf = Buffer.alloc(512);
    buf.write(name, 0, 100);
    buf.write("0000755\0", 100, 8);
    buf.write("0000000\0", 108, 8);
    buf.write("0000000\0", 116, 8);
    buf.write(size.toString(8).padStart(11, "0") + "\0", 124, 12);
    buf.write("00000000000\0", 136, 12);
    buf.write(type, 156, 1);
    buf.write("ustar\0", 257, 6);
    buf.write("00", 263, 2);
    buf.write("        ", 148, 8);
    let sum = 0;
    for (let i = 0; i < 512; i++) sum += buf[i];
    buf.write(sum.toString(8).padStart(6, "0") + "\0 ", 148, 8);
    return buf;
}

function createTar(dir: string, prefix: string): Buffer {
    const bufs: Buffer[] = [];
    function walk(d: string, p: string) {
        for (const name of readdirSync(d)) {
            const full = join(d, name);
            const rel = p ? p + "/" + name : name;
            if (statSync(full).isDirectory()) {
                bufs.push(tarHeader(rel + "/", 0, "5"));
                walk(full, rel);
            } else {
                const data = readFileSync(full);
                bufs.push(tarHeader(rel, data.length, "0"));
                bufs.push(data);
                const pad = 512 - (data.length % 512);
                if (pad < 512) bufs.push(Buffer.alloc(pad));
            }
        }
    }
    walk(dir, prefix);
    bufs.push(Buffer.alloc(1024));
    return Buffer.concat(bufs);
}

async function packRelease(path: string, distDir: string): Promise<void> {
    console.log("  compressing...");
    const tar = createTar(distDir, "dist");
    const compressed = Bun.zstdCompressSync(tar, { level: 19 });

    const exeData = await Bun.file(path).arrayBuffer();

    const footer = new ArrayBuffer(8);
    const view = new DataView(footer);
    view.setUint32(0, compressed.byteLength, true);
    view.setUint32(4, 0x544c4853, true); // SHLT

    await Bun.write(
        path,
        Buffer.concat([Buffer.from(exeData), Buffer.from(compressed), Buffer.from(footer)]),
    );

    const sizeMB = (statSync(path).size / 1024 / 1024).toFixed(1);
    console.log(`  ${basename(path)}: ${sizeMB} MB`);
}

async function packMacRelease(distDir: string, payloadPath: string): Promise<void> {
    console.log("  compressing...");
    const tar = createTar(distDir, "dist");
    const compressed = Bun.zstdCompressSync(tar, { level: 19 });
    await Bun.write(payloadPath, compressed);
    const sizeMB = (statSync(payloadPath).size / 1024 / 1024).toFixed(1);
    console.log(`  payload.bin: ${sizeMB} MB`);
}

export async function bundleNative(
    projectDir: string,
    outputDir: string,
    opts: { release?: boolean },
): Promise<void> {
    const release = opts.release ?? false;
    const name = basename(projectDir);

    await buildWeb(projectDir);

    const distDir = resolve(projectDir, "dist");
    if (!existsSync(distDir)) {
        console.error("  vite build produced no dist/ directory");
        process.exit(1);
    }

    console.log(`  compiling webview shell (${release ? "release" : "debug"})...`);
    cargoBuild(WIN_TARGET, release);

    const exe = cargoTarget(WIN_TARGET, release);
    if (!existsSync(exe)) {
        console.error(`  cargo build produced no exe at ${exe}`);
        process.exit(1);
    }

    const distIcon = resolve(distDir, "icon.png");
    if (!existsSync(distIcon) && existsSync(DEFAULT_ICON)) {
        cpSync(DEFAULT_ICON, distIcon);
    }

    if (existsSync(outputDir)) rmSync(outputDir, { recursive: true });
    mkdirSync(outputDir, { recursive: true });

    const outExe = resolve(outputDir, `${name}.exe`);
    cpSync(exe, outExe);

    if (release) {
        await packRelease(outExe, distDir);
    } else {
        cpSync(distDir, resolve(outputDir, "dist"), { recursive: true });
    }
}

function prepareMacIcon(pngPath: string): string {
    const tmp = resolve(pngPath + ".mac.png");
    const inset = Math.round(1024 * 0.8);
    cpSync(pngPath, tmp);
    execSync(`sips -z ${inset} ${inset} "${tmp}"`, { stdio: "pipe" });
    execSync(`sips --padToHeightWidth 1024 1024 --padColor 1F1E1D "${tmp}"`, { stdio: "pipe" });
    return tmp;
}

function convertIconToIcns(pngPath: string, icnsPath: string): void {
    const macIcon = prepareMacIcon(pngPath);
    const iconsetDir = resolve(icnsPath + ".iconset");
    mkdirSync(iconsetDir, { recursive: true });

    const sizes = [16, 32, 128, 256, 512];
    for (const size of sizes) {
        execSync(
            `sips -z ${size} ${size} "${macIcon}" --out "${iconsetDir}/icon_${size}x${size}.png"`,
            {
                stdio: "pipe",
            },
        );
        const double = size * 2;
        execSync(
            `sips -z ${double} ${double} "${macIcon}" --out "${iconsetDir}/icon_${size}x${size}@2x.png"`,
            {
                stdio: "pipe",
            },
        );
    }

    execSync(`iconutil -c icns "${iconsetDir}" -o "${icnsPath}"`, { stdio: "pipe" });
    rmSync(iconsetDir, { recursive: true });
    rmSync(macIcon);
}

export async function bundleNativeMac(
    projectDir: string,
    outputDir: string,
    opts: { release?: boolean },
): Promise<void> {
    const release = opts.release ?? false;
    const name = basename(projectDir);

    await buildWeb(projectDir);

    const distDir = resolve(projectDir, "dist");
    if (!existsSync(distDir)) {
        console.error("  vite build produced no dist/ directory");
        process.exit(1);
    }

    console.log(`  compiling webview shell (${release ? "release" : "debug"})...`);
    cargoBuild(MAC_TARGET, release);

    const bin = cargoTarget(MAC_TARGET, release);
    if (!existsSync(bin)) {
        console.error(`  cargo build produced no binary at ${bin}`);
        process.exit(1);
    }

    const distIcon = resolve(distDir, "icon.png");
    if (!existsSync(distIcon) && existsSync(DEFAULT_ICON)) {
        cpSync(DEFAULT_ICON, distIcon);
    }

    if (existsSync(outputDir)) rmSync(outputDir, { recursive: true });
    mkdirSync(outputDir, { recursive: true });

    const appDir = resolve(outputDir, `${name}.app`);
    const contentsDir = resolve(appDir, "Contents");
    const macosDir = resolve(contentsDir, "MacOS");
    const resourcesDir = resolve(contentsDir, "Resources");
    mkdirSync(macosDir, { recursive: true });
    mkdirSync(resourcesDir, { recursive: true });

    const plist = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>${name}</string>
    <key>CFBundleIdentifier</key>
    <string>com.multiplekex.${name}</string>
    <key>CFBundleName</key>
    <string>${name}</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleIconFile</key>
    <string>app</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>`;
    writeFileSync(resolve(contentsDir, "Info.plist"), plist);

    const outBin = resolve(macosDir, name);
    cpSync(bin, outBin);
    chmodSync(outBin, 0o755);

    if (existsSync(distIcon)) {
        const icnsPath = resolve(resourcesDir, "app.icns");
        convertIconToIcns(distIcon, icnsPath);
    }

    if (release) {
        await packMacRelease(distDir, resolve(resourcesDir, "payload.bin"));
        execSync(`codesign --force --sign - "${appDir}"`, { stdio: "pipe" });
    } else {
        cpSync(distDir, resolve(macosDir, "dist"), { recursive: true });
    }

    const appSize = (statSync(outBin).size / 1024 / 1024).toFixed(1);
    console.log(`  ${name}.app (${appSize} MB binary)`);
}

function findCefLibs(): string | null {
    const cefPath = process.env.CEF_PATH;
    if (cefPath && existsSync(cefPath)) return cefPath;

    const profile = resolve(RUST_CRATE, `target/${LINUX_TARGET}`);
    for (const dir of ["release", "debug"]) {
        const buildDir = resolve(profile, dir, "build");
        if (!existsSync(buildDir)) continue;
        for (const entry of readdirSync(buildDir)) {
            if (!entry.startsWith("cef-dll-sys-")) continue;
            const out = resolve(buildDir, entry, "out");
            if (!existsSync(out)) continue;
            for (const sub of readdirSync(out)) {
                const candidate = resolve(out, sub);
                if (existsSync(resolve(candidate, "libcef.so"))) return candidate;
            }
        }
    }
    return null;
}

function copyCefLibs(outputDir: string): void {
    const cefSrc = findCefLibs();
    if (!cefSrc) {
        console.error("  CEF libs not found. Set CEF_PATH or build first.");
        process.exit(1);
    }

    const cefOut = resolve(outputDir, "cef");
    mkdirSync(cefOut, { recursive: true });

    const files = [
        "libcef.so",
        "libEGL.so",
        "libGLESv2.so",
        "libvk_swiftshader.so",
        "libvulkan.so.1",
        "icudtl.dat",
        "v8_context_snapshot.bin",
        "chrome-sandbox",
        "chrome_100_percent.pak",
        "chrome_200_percent.pak",
        "resources.pak",
    ];

    for (const file of files) {
        const src = resolve(cefSrc, file);
        if (existsSync(src)) cpSync(src, resolve(cefOut, file));
    }

    const localesSrc = resolve(cefSrc, "locales");
    if (existsSync(localesSrc)) {
        cpSync(localesSrc, resolve(cefOut, "locales"), { recursive: true });
    }

    const swiftshaderSrc = resolve(cefSrc, "swiftshader");
    if (existsSync(swiftshaderSrc)) {
        cpSync(swiftshaderSrc, resolve(cefOut, "swiftshader"), { recursive: true });
    }

    console.log(`  copied CEF libs from ${cefSrc}`);
}

export async function bundleNativeLinux(
    projectDir: string,
    outputDir: string,
    opts: { release?: boolean },
): Promise<void> {
    const release = opts.release ?? false;
    const name = basename(projectDir);

    await buildWeb(projectDir);

    const distDir = resolve(projectDir, "dist");
    if (!existsSync(distDir)) {
        console.error("  vite build produced no dist/ directory");
        process.exit(1);
    }

    console.log(`  compiling webview shell (${release ? "release" : "debug"})...`);
    cargoBuild(LINUX_TARGET, release);

    const bin = cargoTarget(LINUX_TARGET, release);
    if (!existsSync(bin)) {
        console.error(`  cargo build produced no binary at ${bin}`);
        process.exit(1);
    }

    const distIcon = resolve(distDir, "icon.png");
    if (!existsSync(distIcon) && existsSync(DEFAULT_ICON)) {
        cpSync(DEFAULT_ICON, distIcon);
    }

    if (existsSync(outputDir)) rmSync(outputDir, { recursive: true });
    mkdirSync(outputDir, { recursive: true });

    const outBin = resolve(outputDir, name);
    cpSync(bin, outBin);
    chmodSync(outBin, 0o755);

    copyCefLibs(outputDir);

    if (release) {
        await packRelease(outBin, distDir);
    } else {
        cpSync(distDir, resolve(outputDir, "dist"), { recursive: true });
    }
}
