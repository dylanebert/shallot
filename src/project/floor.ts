/** System-webview availability; portable CEF supplies its own Chromium. */
export function verdict(target: string, portable: boolean): string[] {
    if (portable || target !== "linux") return [];
    return [
        `Cannot build ${target} without --portable: its system webview has no usable WebGPU.`,
        "Rebuild with --portable for the bundled Chromium runtime.",
    ];
}

/** Refuse before native emission or launch when the selected backend cannot run the project. */
export function requireBackend(target: string, portable: boolean) {
    const lines = verdict(target, portable);
    if (lines.length === 0) return;
    for (const line of lines) console.error(`  ${line}`);
    process.exit(1);
}
