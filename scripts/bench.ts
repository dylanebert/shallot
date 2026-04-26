import { resolve, join } from "path";
import { existsSync, mkdirSync, cpSync, rmSync, readdirSync } from "fs";
import { Database } from "bun:sqlite";

const projectDir = resolve(import.meta.dir, "..");
const testDir = resolve(projectDir, "packages/shallot/tests/gpu");
const isWSL = process.platform === "linux" && existsSync("/proc/sys/fs/binfmt_misc/WSLInterop");

function detectDisplay(): boolean {
    if (isWSL) return true;
    if (process.platform !== "linux") return true;
    return !!(process.env.DISPLAY || process.env.WAYLAND_DISPLAY);
}

if (!detectDisplay()) {
    console.log("No display available. Skipping GPU tests.");
    process.exit(0);
}

const gpuEnv: Record<string, string> = {};
const passthroughArgs: string[] = [];

const args = process.argv.slice(2);
if (args.includes("--help") || args.includes("-h")) {
    console.log(`Usage: bun bench [options]

Options:
  --scenario <name>    Test scenario (default: benchmark)
  --count <n>          Body count for scenario (default: 100)
  --pipeline <name>    raster or raytracing (default: raster)
  --frames <n>         Measurement frames (default: 500)
  --warmup <n>         Warmup frames (default: 60)
  --effects <names>    Comma-separated effect names, or "all"/"none"
  --camera <mode>      static, pan (default: static)
  --layout <mode>      lorenz, grid (default: lorenz)
  --test <name>        Physics test variant (e.g. "box")

Scenarios:
  pile               Physics body stress test. Scales with --count (default 100)
  physics --test X   Benchmark: runs single physics variant
  audio              Audio scenario. Use --room to select room
  (none)             Default: benchmark with current settings

Effects:
  tonemap, fxaa, vignette, bloom, lensflare, godrays, posterize, dither,
  shadows, skylab, sky, sun, stars, moon, haze, clouds, nosun,
  pl1, pl2, pl3, pl4

Extra args are passed through to Playwright (e.g. --headed, --debug).`);
    process.exit(0);
}
for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    const next = () => args[++i];
    if (arg === "--scenario" && i + 1 < args.length) gpuEnv.GPU_SCENARIO = next();
    else if (arg === "--pipeline" && i + 1 < args.length) gpuEnv.GPU_PIPELINE = next();
    else if (arg === "--warmup" && i + 1 < args.length) gpuEnv.GPU_WARMUP = next();
    else if (arg === "--frames" && i + 1 < args.length) gpuEnv.GPU_FRAMES = next();
    else if (arg === "--count" && i + 1 < args.length) gpuEnv.GPU_COUNT = next();
    else if (arg === "--effects" && i + 1 < args.length) gpuEnv.GPU_EFFECTS = next();
    else if (arg === "--camera" && i + 1 < args.length) gpuEnv.GPU_CAMERA = next();
    else if (arg === "--layout" && i + 1 < args.length) gpuEnv.GPU_LAYOUT = next();
    else if (arg === "--room" && i + 1 < args.length) gpuEnv.GPU_ROOM = next();
    else if (arg === "--test" && i + 1 < args.length) gpuEnv.GPU_TEST = next();
    else if (arg === "--shapes" && i + 1 < args.length) gpuEnv.GPU_SHAPES = next();
    else passthroughArgs.push(arg);
}

function getCommit(): string {
    const result = Bun.spawnSync(["git", "rev-parse", "--short", "HEAD"], {
        cwd: projectDir,
        stdout: "pipe",
    });
    return new TextDecoder().decode(result.stdout).trim() || "unknown";
}

function extractBenchJson(output: string): string | null {
    const match = output.match(/__BENCH_JSON__(.+?)__BENCH_JSON__/);
    return match ? match[1] : null;
}

function ensureSchema(db: Database) {
    db.run("PRAGMA journal_mode = WAL");

    const hasTables = db
        .prepare(
            "SELECT count(*) as n FROM sqlite_master WHERE type='table' AND name IN ('runs','startup')",
        )
        .get() as { n: number };

    if (hasTables.n < 2) {
        db.run("DROP TABLE IF EXISTS runs");
        db.run("DROP TABLE IF EXISTS startup");

        db.run(`CREATE TABLE runs (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			timestamp TEXT NOT NULL,
			commit_hash TEXT NOT NULL,
			hardware TEXT NOT NULL,
			pipeline TEXT NOT NULL,
			scenario TEXT NOT NULL DEFAULT 'default',
			effects TEXT NOT NULL DEFAULT 'none',
			camera TEXT NOT NULL DEFAULT 'static',
			layout TEXT NOT NULL DEFAULT 'lorenz',
			object_count INTEGER,
			warmup_frames INTEGER NOT NULL,
			measure_frames INTEGER NOT NULL,
			gpu_avg REAL, gpu_median REAL, gpu_p5 REAL, gpu_p95 REAL, gpu_min REAL, gpu_max REAL,
			gpu_samples INTEGER, passes_json TEXT,
			cpu_total REAL, cpu_systems_json TEXT,
			memory_start INTEGER, memory_end INTEGER,
			memory_growth_per_frame REAL, memory_leak INTEGER DEFAULT 0,
			gc_count INTEGER, gc_pause_ms REAL,
			ramp INTEGER DEFAULT 0
		)`);

        db.run(`CREATE TABLE startup (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			timestamp TEXT NOT NULL,
			commit_hash TEXT NOT NULL,
			hardware TEXT NOT NULL,
			scenario TEXT NOT NULL DEFAULT 'default',
			total_ms REAL,
			plugin_init_json TEXT,
			scene_load_ms REAL,
			warm_json TEXT
		)`);
    }

    const startupCols = db.prepare("PRAGMA table_info(startup)").all() as { name: string }[];
    const startupColNames = new Set(startupCols.map((c) => c.name));
    if (!startupColNames.has("compile_total_ms")) {
        db.run("ALTER TABLE startup ADD COLUMN compile_total_ms REAL");
        db.run("ALTER TABLE startup ADD COLUMN compile_pipelines_json TEXT");
    }

    const cols = db.prepare("PRAGMA table_info(runs)").all() as { name: string }[];
    const colNames = new Set(cols.map((c) => c.name));
    if (!colNames.has("gc_count")) {
        db.run("ALTER TABLE runs ADD COLUMN gc_count INTEGER");
        db.run("ALTER TABLE runs ADD COLUMN gc_pause_ms REAL");
    }
    if (!colNames.has("ramp")) {
        db.run("ALTER TABLE runs ADD COLUMN ramp INTEGER DEFAULT 0");
    }
    if (!colNames.has("frame_avg")) {
        db.run("ALTER TABLE runs ADD COLUMN frame_avg REAL");
        db.run("ALTER TABLE runs ADD COLUMN frame_median REAL");
        db.run("ALTER TABLE runs ADD COLUMN frame_p5 REAL");
        db.run("ALTER TABLE runs ADD COLUMN frame_p95 REAL");
        db.run("ALTER TABLE runs ADD COLUMN frame_min REAL");
        db.run("ALTER TABLE runs ADD COLUMN frame_max REAL");
        db.run("ALTER TABLE runs ADD COLUMN frame_samples INTEGER");
        db.run("ALTER TABLE runs ADD COLUMN frame_clamped INTEGER");
        db.run("ALTER TABLE runs ADD COLUMN frame_avg_fixed_steps REAL");
    }
}

function writeResults(output: string) {
    const json = extractBenchJson(output);
    if (!json) return;

    const data = JSON.parse(json);
    const dbPath = join(testDir, "results.db");
    const db = new Database(dbPath);
    ensureSchema(db);

    const commit = getCommit();

    const insertRun = db.prepare(`INSERT INTO runs
		(timestamp, commit_hash, hardware, pipeline, scenario, effects, camera, layout,
		 object_count, warmup_frames, measure_frames,
		 gpu_avg, gpu_median, gpu_p5, gpu_p95, gpu_min, gpu_max, gpu_samples, passes_json,
		 cpu_total, cpu_systems_json,
		 memory_start, memory_end, memory_growth_per_frame, memory_leak,
		 gc_count, gc_pause_ms,
		 frame_avg, frame_median, frame_p5, frame_p95, frame_min, frame_max,
		 frame_samples, frame_clamped, frame_avg_fixed_steps,
		 ramp)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`);

    for (const run of data.runs) {
        const effectsKey = [...run.effects].sort().join(",") || "none";
        insertRun.run(
            data.timestamp,
            commit,
            data.hardware,
            run.pipeline,
            run.scenario ?? "default",
            effectsKey,
            run.camera ?? "static",
            run.layout ?? "lorenz",
            run.count ?? null,
            run.warmup,
            run.frames,
            run.gpu?.avg ?? null,
            run.gpu?.median ?? null,
            run.gpu?.p5 ?? null,
            run.gpu?.p95 ?? null,
            run.gpu?.min ?? null,
            run.gpu?.max ?? null,
            run.gpu?.samples ?? null,
            run.passes ? JSON.stringify(run.passes) : null,
            run.cpu?.total ?? null,
            run.cpu?.systems ? JSON.stringify(run.cpu.systems) : null,
            run.memory?.start ?? null,
            run.memory?.end ?? null,
            run.memory?.growthPerFrame ?? null,
            run.memory?.leak ? 1 : 0,
            run.memory?.gcCount ?? null,
            run.memory?.gcPauseMs ?? null,
            run.frame?.avg ?? null,
            run.frame?.median ?? null,
            run.frame?.p5 ?? null,
            run.frame?.p95 ?? null,
            run.frame?.min ?? null,
            run.frame?.max ?? null,
            run.frame?.samples ?? null,
            run.frame?.clampedFrames ?? null,
            run.frame?.avgFixedSteps ?? null,
            run.ramp ? 1 : 0,
        );
    }

    if (data.startup) {
        const insertStartup = db.prepare(`INSERT INTO startup
			(timestamp, commit_hash, hardware, scenario, total_ms,
			 plugin_init_json, scene_load_ms, warm_json)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?)`);
        const s = data.startup;
        insertStartup.run(
            data.timestamp,
            commit,
            data.hardware,
            s.scenario ?? "default",
            s.total ?? null,
            s.pluginInit ? JSON.stringify(s.pluginInit) : null,
            s.sceneLoad ?? null,
            s.warm ? JSON.stringify(s.warm) : null,
        );
    }

    if (data.compile) {
        const scenario = data.runs?.[0]?.scenario ?? "default";
        db.prepare(
            `INSERT INTO startup (timestamp, commit_hash, hardware, scenario, compile_total_ms, compile_pipelines_json)
			 VALUES (?, ?, ?, ?, ?, ?)`,
        ).run(
            data.timestamp,
            commit,
            data.hardware,
            scenario,
            data.compile.totalMs ?? null,
            data.compile.pipelines ? JSON.stringify(data.compile.pipelines) : null,
        );
    }

    db.close();
    console.log(`Results written to ${dbPath}`);
}

interface PlaywrightResult {
    exitCode: number;
    stdout: string;
}

function computeTimeout(): number {
    const warmup = parseInt(gpuEnv.GPU_WARMUP || "60", 10);
    const frames = parseInt(gpuEnv.GPU_FRAMES || "500", 10);
    return 30000 + (warmup + frames) * 50 + 30000;
}

async function runPlaywright(): Promise<PlaywrightResult> {
    if (isWSL) return runPlaywrightWSL();

    const playwrightArgs = [
        "bunx",
        "playwright",
        "test",
        "--config",
        "gym.config.ts",
        "gpu.pw.ts",
        ...passthroughArgs,
    ];

    const result = Bun.spawnSync(playwrightArgs, {
        cwd: testDir,
        stdout: "pipe",
        stderr: "inherit",
        env: { ...process.env, ...gpuEnv },
        timeout: computeTimeout(),
    });

    const stdout = new TextDecoder().decode(result.stdout);
    process.stdout.write(stdout);
    return { exitCode: result.exitCode, stdout };
}

async function startGymServer(): Promise<ReturnType<typeof Bun.spawn>> {
    try {
        await fetch("http://localhost:3002");
        Bun.spawnSync(["fuser", "-k", "3002/tcp"], { stdout: "ignore", stderr: "ignore" });
        await Bun.sleep(500);
    } catch {}

    const proc = Bun.spawn(["bun", "run", "dev", "--port", "3002", "--strictPort"], {
        cwd: resolve(projectDir, "examples/gym"),
        stdout: "ignore",
        stderr: "pipe",
    });

    for (let i = 0; i < 60; i++) {
        if (proc.exitCode !== null) {
            const stderr = await new Response(proc.stderr).text();
            throw new Error(`Gym server exited early (code ${proc.exitCode}): ${stderr}`);
        }
        try {
            await fetch("http://localhost:3002");
            console.log("Gym server ready on port 3002");
            return proc;
        } catch {
            if (i === 59) throw new Error("Gym server failed to start on port 3002");
            await Bun.sleep(500);
        }
    }
    throw new Error("unreachable");
}

async function runPlaywrightWSL(): Promise<PlaywrightResult> {
    const gymProc = await startGymServer();

    const winTempProc = Bun.spawnSync(
        ["powershell.exe", "-Command", "Write-Host -NoNewline $env:TEMP"],
        { stdout: "pipe" },
    );
    const winTempPath = new TextDecoder().decode(winTempProc.stdout).trim().replace(/\r/g, "");
    const wslTempProc = Bun.spawnSync(["wslpath", winTempPath], { stdout: "pipe" });
    const wslTemp = new TextDecoder().decode(wslTempProc.stdout).trim();
    const testTemp = join(wslTemp, "shallot-gpu-tests");
    const winTestTemp = winTempPath + "\\shallot-gpu-tests";

    rmSync(testTemp, { recursive: true, force: true });
    mkdirSync(testTemp, { recursive: true });

    const filesToCopy = new Set(["package.json", "gym.config.ts", "gpu.pw.ts"]);
    for (const file of readdirSync(testDir)) {
        if (filesToCopy.has(file)) {
            cpSync(join(testDir, file), join(testTemp, file));
        }
    }

    const fixturesDir = resolve(testDir, "../fixtures/avbd");
    if (existsSync(fixturesDir)) {
        cpSync(fixturesDir, join(testTemp, "fixtures", "avbd"), { recursive: true });
    }

    console.log("Installing Playwright dependencies...");
    Bun.spawnSync(
        [
            "powershell.exe",
            "-Command",
            `cd '${winTestTemp}'; bun install --silent; bunx playwright install chromium`,
        ],
        { stdout: "inherit", stderr: "inherit" },
    );

    const envPrefix = Object.entries(gpuEnv)
        .map(([k, v]) => `$env:${k} = '${v}';`)
        .join(" ");

    console.log("Running GPU tests via Windows...");
    const result = Bun.spawnSync(
        [
            "powershell.exe",
            "-Command",
            `${envPrefix} $env:PLAYWRIGHT_BROWSERS_PATH = "$env:LOCALAPPDATA\\ms-playwright"; cd '${winTestTemp}'; bunx playwright test --config gym.config.ts gpu.pw.ts${passthroughArgs.length > 0 ? ` ${passthroughArgs.join(" ")}` : ""}`,
        ],
        { stdout: "pipe", stderr: "inherit", timeout: computeTimeout() },
    );

    gymProc.kill();

    const stdout = new TextDecoder().decode(result.stdout);
    process.stdout.write(stdout);
    return { exitCode: result.exitCode, stdout };
}

console.log("Running GPU tests...");
const result = await runPlaywright();
writeResults(result.stdout);

if (result.exitCode === 0) {
    console.log("GPU tests passed");
} else {
    console.error("GPU tests failed");
    process.exit(result.exitCode);
}
