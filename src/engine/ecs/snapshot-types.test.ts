import type { World, WorldSnapshot } from "@dylanebert/shallot";
import type { WorldSnapshot as EngineSnapshot } from "./world";

// The public name must identify its owner's return type, not only exist in a barrel.
type Assert<T extends true> = T;
type Equal<A, B> =
    (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2 ? true : false;
export type RootWorldSnapshot = Assert<Equal<WorldSnapshot, EngineSnapshot>>;
export type WorldSnapshotResult = Assert<Equal<WorldSnapshot, ReturnType<World["snapshot"]>>>;

// @ts-expect-error solver-only snapshots are internal; recovery uses WorldSnapshot.
export type SolverSnapshotIsInternal = import("@dylanebert/shallot").PhysicsSnapshot;
