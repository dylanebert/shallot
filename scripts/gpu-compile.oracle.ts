const child = Bun.spawn([process.execPath, "test", "gpu.test", ...process.argv.slice(2)], {
    env: { ...process.env, SHALLOT_GPU_COMPILE_ORACLE: "1" },
    stdin: "inherit",
    stdout: "inherit",
    stderr: "inherit",
});
process.exit(await child.exited);
