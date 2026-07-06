import { Compute, type Mirror } from "@dylanebert/shallot";

// The project's own tiny test/boot helpers — published-surface-only (no reach into the repo harness).
// A consumer testing a shallot project needs exactly these two primitives plus their own browser driver;
// they're a few lines over the public `Compute` + `Mirror` surface, so we own them here rather than
// importing anything repo-internal. {@link Check} is the gate's verdict shape, read by `window.__voxelGate`.

export interface Check {
    name: string;
    pass: boolean;
    detail: string;
}

/** await `n` animation frames — lets the running render loop advance a known amount. */
export function frames(n: number): Promise<void> {
    return new Promise((resolve) => {
        let i = 0;
        const tick = () => (++i >= n ? resolve() : requestAnimationFrame(tick));
        requestAnimationFrame(tick);
    });
}

// Mirror is 1-2 frames stale by design (a staging ring + async map). After mutating state the GPU reads,
// wait until a snapshot encoded *after* now lands, so a readback reflects the new state. Bounded — a stuck
// map resolves to the loop cap.
export async function settle(m: Mirror, max = 120): Promise<void> {
    const target = Compute.frame + 2;
    for (let i = 0; i < max; i++) {
        await frames(1);
        if (m.snapshot && m.snapshot.frame >= target) return;
    }
}
