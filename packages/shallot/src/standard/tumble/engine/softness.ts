// Soft-constraint coefficients (Box3D's b3Softness / b3MakeSoft, Erin Catto, MIT). Extracted into
// its own leaf module (math-only) so both the contact solver and the joints can pull it without an
// import cycle through body.ts. fround discipline per the README.

import { f32, PI } from "./math";

/** Soft-constraint coefficients derived from a target frequency (b3Softness). */
export type Softness = { biasRate: number; massScale: number; impulseScale: number };

/** Write soft-constraint coefficients into `out` (b3MakeSoft, in place — reuses the object across steps). */
export function writeSoft(out: Softness, hertz: number, zeta: number, h: number): void {
    if (hertz === 0) {
        out.biasRate = 0;
        out.massScale = 0;
        out.impulseScale = 0;
        return;
    }
    const omega = f32(f32(2.0 * PI) * hertz);
    const a1 = f32(f32(2.0 * zeta) + f32(h * omega));
    const a2 = f32(f32(h * omega) * a1);
    const a3 = f32(1.0 / f32(1.0 + a2));
    out.biasRate = f32(omega / a1);
    out.massScale = f32(a2 * a3);
    out.impulseScale = a3;
}

/** @returns soft-constraint coefficients for a frequency, damping ratio, and substep dt (b3MakeSoft). */
export function makeSoft(hertz: number, zeta: number, h: number): Softness {
    const out = { biasRate: 0, massScale: 0, impulseScale: 0 };
    writeSoft(out, hertz, zeta, h);
    return out;
}
