/** Emit a browser `shallot:startup` observation; no state is retained. @internal */
export function reportStartup(detail: { kind: string; [field: string]: unknown }): void {
    if (typeof window !== "undefined")
        window.dispatchEvent(new CustomEvent("shallot:startup", { detail }));
}
