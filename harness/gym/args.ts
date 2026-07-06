export interface GymArgs {
    scenario: string;
    seed: number;
    count?: number;
    warmup: number;
    frames: number;
    // overall run budget in ms — also raises the build/settle ready-window (default 30s), for a scenario
    // whose build outlasts it (a large physics pile). Omitted = the warmup+frames-derived default.
    timeoutMs?: number;
    // extra `key=value` URL params a scenario reads from window.location (e.g. dist=clustered, viz=0)
    params: string[];
    // absolute path to write a post-run canvas screenshot (debug/visual smoke test); undefined = none
    screenshot?: string;
    // forwarded to Playwright (e.g. --headed, --debug)
    passthrough: string[];
}

// Scenario-agnostic by design: the launcher forwards `--scenario` to the URL and routes
// the returned verdict. The gym example owns the scenario list and validates the name
// (an unknown scenario throws in the page, surfacing as a run failure here). The launcher
// never imports the example — dependencies point inward, into the harness core only.
function help(): void {
    console.log(`Usage: bun bench [options]

Runs one gym scenario (examples/gym) headless under Playwright and routes its
verdict: metrics → printed frame-time, checks → pass/fail gate.

Options:
  --scenario <name>    which scenario to run (default: render). See examples/gym.
  --seed <n>           determinism seed (default: 1)
  --count <n>          per-scenario size param (scenario default if omitted)
  --warmup <n>         warmup frames (default: 60)
  --frames <n>         measurement frames (default: 240)
  --timeout <ms>       overall run budget; also raises the build/settle ready-window (default 30s) for a heavy scenario
  --param <key=value>  extra URL param a scenario reads (repeatable; e.g. --param dist=clustered --param viz=0)
  --screenshot <path>  write a post-run canvas screenshot to <path> (PNG; visual smoke test)

Extra args pass through to Playwright (e.g. --headed, --debug).`);
}

export function parseArgs(argv: string[]): GymArgs {
    if (argv.includes("--help") || argv.includes("-h")) {
        help();
        process.exit(0);
    }

    const out: GymArgs = {
        scenario: "render",
        seed: 1,
        count: undefined,
        warmup: 60,
        frames: 240,
        params: [],
        passthrough: [],
    };

    const take = (name: string, i: number): string => {
        if (i + 1 >= argv.length) throw new Error(`--${name} requires a value`);
        return argv[i + 1];
    };

    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        if (!arg.startsWith("--")) {
            out.passthrough.push(arg);
            continue;
        }
        const name = arg.slice(2);
        switch (name) {
            case "scenario":
                out.scenario = take(name, i++);
                break;
            case "seed":
                out.seed = parseInt(take(name, i++), 10);
                break;
            case "count":
                out.count = parseInt(take(name, i++), 10);
                break;
            case "warmup":
                out.warmup = parseInt(take(name, i++), 10);
                break;
            case "frames":
                out.frames = parseInt(take(name, i++), 10);
                break;
            case "timeout":
                out.timeoutMs = parseInt(take(name, i++), 10);
                break;
            case "param":
                out.params.push(take(name, i++));
                break;
            case "screenshot":
                out.screenshot = take(name, i++);
                break;
            default:
                out.passthrough.push(arg);
                if (i + 1 < argv.length && !argv[i + 1].startsWith("--")) {
                    out.passthrough.push(argv[++i]);
                }
        }
    }

    return out;
}
