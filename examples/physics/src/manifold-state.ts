import type { HullData, LocalManifold } from "@dylanebert/shallot/standard/physics";
import type { CullingState, WedgeState } from "./manifold-diagnostics";
import { pose, vector } from "./manifold-math";
export const manifoldNames = [
    "sphereVsSphere",
    "capsuleVsSphere",
    "hullVsSphere",
    "triangleVsSphere",
    "capsuleVsCapsule",
    "capsuleVsHull",
    "triangleVsCapsule",
    "hullVsHull",
    "triangleVsHull",
] as const;
export const manifoldDiagnosticNames = ["hullCulling", "edgeAxisWedge"] as const;
export const allManifoldNames = [...manifoldNames, ...manifoldDiagnosticNames] as const;
export type ManifoldScene = (typeof allManifoldNames)[number];
export type CollisionManifoldScene = (typeof manifoldNames)[number];
export const manifoldPointLimits = [1, 1, 1, 1, 2, 2, 2, 4, 7] as const;
export const simplex = () => ({ metric: 0, count: 0, indexA: [0, 0, 0, 0], indexB: [0, 0, 0, 0] });
export const sat = () => ({ separation: 0, type: 0, indexA: 0, indexB: 0, hit: 0 });
export const ManifoldSample = {
    create: () => ({
        scene: "" as ManifoldScene | "",
        a: pose(),
        b: pose(),
        sphere: { center: vector(), radius: 1 },
        capsule: { center1: vector(-2), center2: vector(2), radius: 1 },
        hullA: null as HullData | null,
        hullB: null as HullData | null,
        triangle: [vector(), vector(), vector()] as [
            ReturnType<typeof vector>,
            ReturnType<typeof vector>,
            ReturnType<typeof vector>,
        ],
        simplex: simplex(),
        sat: sat(),
        useCache: false,
        drawPoints: true,
        manualFeature: 0,
        manifold: null as LocalManifold | null,
        tracking: false,
        origin: vector(),
        base: vector(),
        culling: null as CullingState | null,
        wedge: null as WedgeState | null,
    }),
};
export type ManifoldState = ReturnType<typeof ManifoldSample.create>;
