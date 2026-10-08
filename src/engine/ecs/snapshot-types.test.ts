import type { PhysicsSnapshot, PhysicsWorld, World, WorldSnapshot } from "@dylanebert/shallot";
import type { WorldSnapshot as EngineSnapshot } from "./world";

// The public names must identify their owners' return types, not only exist in a barrel.
type Assert<T extends true> = T;
type Equal<A, B> =
    (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2 ? true : false;
export type RootWorldSnapshot = Assert<Equal<WorldSnapshot, EngineSnapshot>>;
export type WorldSnapshotResult = Assert<Equal<WorldSnapshot, ReturnType<World["snapshot"]>>>;
export type PhysicsSnapshotResult = Assert<
    Equal<PhysicsSnapshot, ReturnType<PhysicsWorld["snapshot"]>>
>;
