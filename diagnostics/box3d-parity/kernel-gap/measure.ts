// Build roots first; one scene's measurement excludes toolchain compilation.
const [profileRoot, countRoot, native, scene = "junkyard"] = process.argv.slice(2);
if (!profileRoot || !countRoot || !native) throw new Error("usage: bun diagnostics/box3d-parity/kernel-gap/measure.ts PROFILE_ROOT COUNT_ROOT NATIVE_COUNTS rain|junkyard|joint_grid");
const start = performance.now();
for (const [script, args, env] of [
    ["run.ts", [profileRoot, scene], {}],
    ["run.ts", [countRoot, scene], { NATIVE_COUNTS: native }],
    ["times.ts", [native, scene], {}],
] as const) {
    const result = Bun.spawnSync(["bun", `${import.meta.dir}/${script}`, ...args], { env: { ...process.env, ...env }, stdout: "inherit", stderr: "inherit" });
    if (result.exitCode) throw new Error(`${script}: ${result.exitCode}`);
}
console.log(`${scene}: complete diagnostic ${(performance.now()-start).toFixed(0)} ms (builds excluded)`);
