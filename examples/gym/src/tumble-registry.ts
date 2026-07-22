// The registry every stage-4 sample twin plugs into (spec tumble-inline stage 4): one entry per ported
// tumble.js sample — its committed gold, its build()/update() through the escape hatch, and 2-3 non-default
// knob points for the boundedness probe (the gold trajectory exists at defaults only). `tumble-golds.test.ts`
// iterates this list; a new sample twin adds ONE entry here — see that file's header for the full recipe.

import goldJsonBodyType from "../../../packages/shallot/tests/tumble/samples/bodies-body-type.json";
import goldJsonMotionLocks from "../../../packages/shallot/tests/tumble/samples/bodies-motion-locks.json";
import goldJsonSpinningBook from "../../../packages/shallot/tests/tumble/samples/bodies-spinning-book.json";
import goldJsonCharacterMover from "../../../packages/shallot/tests/tumble/samples/character-mover.json";
import goldJsonOverlapBox from "../../../packages/shallot/tests/tumble/samples/collision-overlap-box.json";
import goldJsonRayCurtain from "../../../packages/shallot/tests/tumble/samples/collision-ray-curtain.json";
import goldJsonShapeCast from "../../../packages/shallot/tests/tumble/samples/collision-shape-cast.json";
import goldJsonCompoundSimple from "../../../packages/shallot/tests/tumble/samples/compound-simple.json";
import goldJsonCompoundSpheres from "../../../packages/shallot/tests/tumble/samples/compound-spheres.json";
import goldJsonTileFloor from "../../../packages/shallot/tests/tumble/samples/compound-tile-floor.json";
import goldJsonBulletVsStack from "../../../packages/shallot/tests/tumble/samples/continuous-bullet-vs-stack.json";
import goldJsonThinWall from "../../../packages/shallot/tests/tumble/samples/continuous-thin-wall.json";
import goldJsonFallingRagdolls from "../../../packages/shallot/tests/tumble/samples/determinism-falling-ragdolls.json";
import goldJsonHitEvents from "../../../packages/shallot/tests/tumble/samples/events-hit.json";
import goldJsonJointBreak from "../../../packages/shallot/tests/tumble/samples/events-joint-break.json";
import goldJsonSensorSweep from "../../../packages/shallot/tests/tumble/samples/events-sensor-sweep.json";
import goldJsonConvexHull from "../../../packages/shallot/tests/tumble/samples/geometry-convex-hull.json";
import goldJsonConvexPrimitives from "../../../packages/shallot/tests/tumble/samples/geometry-convex-primitives.json";
import goldJsonHullReduction from "../../../packages/shallot/tests/tumble/samples/geometry-hull-reduction.json";
import goldJsonBridge from "../../../packages/shallot/tests/tumble/samples/joints-bridge.json";
import goldJsonCantilever from "../../../packages/shallot/tests/tumble/samples/joints-cantilever.json";
import goldJsonDriving from "../../../packages/shallot/tests/tumble/samples/joints-driving.json";
import goldJsonElevator from "../../../packages/shallot/tests/tumble/samples/joints-elevator.json";
import goldJsonFilter from "../../../packages/shallot/tests/tumble/samples/joints-filter.json";
import goldJsonPaddle from "../../../packages/shallot/tests/tumble/samples/joints-paddle.json";
import goldJsonParallel from "../../../packages/shallot/tests/tumble/samples/joints-parallel.json";
import goldJsonPendulum from "../../../packages/shallot/tests/tumble/samples/joints-pendulum.json";
import goldJsonRope from "../../../packages/shallot/tests/tumble/samples/joints-rope.json";
import goldJsonSuspension from "../../../packages/shallot/tests/tumble/samples/joints-suspension.json";
import goldJsonTerrain from "../../../packages/shallot/tests/tumble/samples/mesh-terrain.json";
import goldJsonTorus from "../../../packages/shallot/tests/tumble/samples/mesh-torus.json";
import goldJsonRagdoll from "../../../packages/shallot/tests/tumble/samples/ragdoll-ragdoll.json";
import goldJsonInclinedPlane from "../../../packages/shallot/tests/tumble/samples/shapes-inclined-plane.json";
import goldJsonRestitution from "../../../packages/shallot/tests/tumble/samples/shapes-restitution.json";
import goldJsonShapeSoup from "../../../packages/shallot/tests/tumble/samples/shapes-shape-soup.json";
import goldJsonArch from "../../../packages/shallot/tests/tumble/samples/stacking-arch.json";
import goldJsonBoxPyramid from "../../../packages/shallot/tests/tumble/samples/stacking-box-pyramid.json";
import goldJsonDominoes from "../../../packages/shallot/tests/tumble/samples/stacking-dominoes.json";
import { buildArch } from "./tumble-arch";
import { buildBodyType, updateBodyType } from "./tumble-body-type";
import { buildBoxPyramid } from "./tumble-box-pyramid";
import { buildBridge } from "./tumble-bridge";
import { buildBulletVsStack } from "./tumble-bullet-vs-stack";
import { buildCantilever } from "./tumble-cantilever";
import { buildCharacterMover, updateCharacterMover } from "./tumble-character-mover";
import { buildCompoundSimple } from "./tumble-compound-simple";
import { buildCompoundSpheres } from "./tumble-compound-spheres";
import { buildTileFloor } from "./tumble-compound-tile-floor";
import { buildConvexHull } from "./tumble-convex-hull";
import { buildConvexPrimitives } from "./tumble-convex-primitives";
import { buildDominoes } from "./tumble-dominoes";
import { buildDriving } from "./tumble-driving";
import { buildElevator, updateElevator } from "./tumble-elevator";
import { buildFallingRagdolls } from "./tumble-falling-ragdolls";
import { buildFilter } from "./tumble-filter";
import { buildHitEvents } from "./tumble-hit";
import { buildHullReduction } from "./tumble-hull-reduction";
import { buildInclinedPlane } from "./tumble-inclined-plane";
import { buildJointBreak, updateJointBreak } from "./tumble-joint-break";
import { buildMotionLocks, updateMotionLocks } from "./tumble-motion-locks";
import type { SampleBuild, SampleGold, SampleParams, SampleUpdate } from "./tumble-oracle";
import { buildOverlapBox } from "./tumble-overlap-box";
import { buildPaddle } from "./tumble-paddle";
import { buildParallel, updateParallel } from "./tumble-parallel";
import { buildPendulum } from "./tumble-pendulum";
import { buildRagdoll } from "./tumble-ragdoll";
import { buildRayCurtain } from "./tumble-ray-curtain";
import { buildRestitution } from "./tumble-restitution";
import { buildRope } from "./tumble-rope";
import { buildSensorSweep, updateSensorSweep } from "./tumble-sensor-sweep";
import { buildShapeCast } from "./tumble-shape-cast";
import { buildShapeSoup } from "./tumble-shape-soup";
import { buildSpinningBook } from "./tumble-spinning-book";
import { buildSuspension } from "./tumble-suspension";
import { buildTerrain } from "./tumble-terrain";
import { buildThinWall } from "./tumble-thin-wall";
import { buildTorus } from "./tumble-torus";

/** one registry entry: a ported sample's committed gold + its build/update, plus 2-3 non-default knob
 *  points the boundedness probe steps through (never gold-checked — only asserted finite). Omit
 *  `knobPoints` for a sample with no knobs. */
export interface GoldEntry {
    slug: string;
    gold: SampleGold;
    build: SampleBuild;
    update?: SampleUpdate;
    knobPoints?: SampleParams[];
}

export const goldRegistry: GoldEntry[] = [
    {
        slug: "joints-paddle",
        gold: goldJsonPaddle as unknown as SampleGold,
        build: buildPaddle,
        knobPoints: [{ speed: -8 }, { speed: 0 }, { speed: 8 }],
    },
    {
        slug: "bodies-body-type",
        gold: goldJsonBodyType as unknown as SampleGold,
        build: buildBodyType,
        update: updateBodyType,
        knobPoints: [{ type: "dynamic" }, { type: "static" }],
    },
    {
        slug: "stacking-arch",
        gold: goldJsonArch as unknown as SampleGold,
        build: buildArch,
    },
    {
        slug: "stacking-box-pyramid",
        gold: goldJsonBoxPyramid as unknown as SampleGold,
        build: buildBoxPyramid,
        knobPoints: [{ rows: 3 }, { rows: 6 }, { rows: 14 }],
    },
    {
        slug: "stacking-dominoes",
        gold: goldJsonDominoes as unknown as SampleGold,
        build: buildDominoes,
        knobPoints: [{ rings: 1 }, { rings: 2 }, { rings: 8 }],
    },
    {
        slug: "shapes-inclined-plane",
        gold: goldJsonInclinedPlane as unknown as SampleGold,
        build: buildInclinedPlane,
    },
    {
        slug: "shapes-restitution",
        gold: goldJsonRestitution as unknown as SampleGold,
        build: buildRestitution,
        knobPoints: [{ shape: "box" }, { count: 4 }, { count: 40 }],
    },
    {
        slug: "shapes-shape-soup",
        gold: goldJsonShapeSoup as unknown as SampleGold,
        build: buildShapeSoup,
        knobPoints: [{ rows: 2 }, { rows: 5 }, { rows: 8 }],
    },
    {
        slug: "bodies-motion-locks",
        gold: goldJsonMotionLocks as unknown as SampleGold,
        build: buildMotionLocks,
        update: updateMotionLocks,
    },
    {
        slug: "bodies-spinning-book",
        gold: goldJsonSpinningBook as unknown as SampleGold,
        build: buildSpinningBook,
    },
    {
        slug: "character-mover",
        gold: goldJsonCharacterMover as unknown as SampleGold,
        build: buildCharacterMover,
        update: updateCharacterMover,
    },
    {
        slug: "continuous-bullet-vs-stack",
        gold: goldJsonBulletVsStack as unknown as SampleGold,
        build: buildBulletVsStack,
        knobPoints: [{ speed: 40 }, { speed: 80 }, { speed: 200 }],
    },
    {
        slug: "continuous-thin-wall",
        gold: goldJsonThinWall as unknown as SampleGold,
        build: buildThinWall,
    },
    {
        slug: "collision-overlap-box",
        gold: goldJsonOverlapBox as unknown as SampleGold,
        build: buildOverlapBox,
        knobPoints: [{ size: 1 }, { size: 2.5 }, { size: 4 }],
    },
    {
        slug: "collision-ray-curtain",
        gold: goldJsonRayCurtain as unknown as SampleGold,
        build: buildRayCurtain,
    },
    {
        slug: "collision-shape-cast",
        gold: goldJsonShapeCast as unknown as SampleGold,
        build: buildShapeCast,
    },
    {
        slug: "compound-simple",
        gold: goldJsonCompoundSimple as unknown as SampleGold,
        build: buildCompoundSimple,
    },
    {
        slug: "compound-spheres",
        gold: goldJsonCompoundSpheres as unknown as SampleGold,
        build: buildCompoundSpheres,
    },
    {
        slug: "compound-tile-floor",
        gold: goldJsonTileFloor as unknown as SampleGold,
        build: buildTileFloor,
    },
    {
        slug: "events-hit",
        gold: goldJsonHitEvents as unknown as SampleGold,
        build: buildHitEvents,
    },
    {
        slug: "events-joint-break",
        gold: goldJsonJointBreak as unknown as SampleGold,
        build: buildJointBreak,
        update: updateJointBreak,
    },
    {
        slug: "events-sensor-sweep",
        gold: goldJsonSensorSweep as unknown as SampleGold,
        build: buildSensorSweep,
        update: updateSensorSweep,
    },
    {
        slug: "geometry-convex-hull",
        gold: goldJsonConvexHull as unknown as SampleGold,
        build: buildConvexHull,
        knobPoints: [{ count: 4 }, { count: 12 }, { count: 24 }],
    },
    {
        slug: "geometry-convex-primitives",
        gold: goldJsonConvexPrimitives as unknown as SampleGold,
        build: buildConvexPrimitives,
    },
    {
        slug: "geometry-hull-reduction",
        gold: goldJsonHullReduction as unknown as SampleGold,
        build: buildHullReduction,
        knobPoints: [
            { vertices: 4, count: 1 },
            { vertices: 8, count: 6 },
            { vertices: 20, count: 12 },
        ],
    },
    {
        slug: "joints-bridge",
        gold: goldJsonBridge as unknown as SampleGold,
        build: buildBridge,
        knobPoints: [{ planks: 10 }, { planks: 18 }, { planks: 40 }],
    },
    {
        slug: "joints-cantilever",
        gold: goldJsonCantilever as unknown as SampleGold,
        build: buildCantilever,
        knobPoints: [{ stiffness: 2 }, { stiffness: 30 }, { stiffness: 60 }],
    },
    {
        slug: "joints-driving",
        gold: goldJsonDriving as unknown as SampleGold,
        build: buildDriving,
        knobPoints: [{ throttle: -10 }, { throttle: 0 }, { throttle: 20 }],
    },
    {
        slug: "joints-elevator",
        gold: goldJsonElevator as unknown as SampleGold,
        build: buildElevator,
        update: updateElevator,
    },
    {
        slug: "joints-filter",
        gold: goldJsonFilter as unknown as SampleGold,
        build: buildFilter,
    },
    {
        slug: "joints-parallel",
        gold: goldJsonParallel as unknown as SampleGold,
        build: buildParallel,
        update: updateParallel,
    },
    {
        slug: "joints-pendulum",
        gold: goldJsonPendulum as unknown as SampleGold,
        build: buildPendulum,
        knobPoints: [{ links: 2 }, { links: 6 }, { links: 14 }],
    },
    {
        slug: "joints-rope",
        gold: goldJsonRope as unknown as SampleGold,
        build: buildRope,
        knobPoints: [{ links: 4 }, { links: 12 }, { links: 20 }],
    },
    {
        slug: "joints-suspension",
        gold: goldJsonSuspension as unknown as SampleGold,
        build: buildSuspension,
        knobPoints: [{ stiffness: 1 }, { stiffness: 6 }, { stiffness: 12 }],
    },
    {
        slug: "mesh-terrain",
        gold: goldJsonTerrain as unknown as SampleGold,
        build: buildTerrain,
        knobPoints: [{ shape: "sphere" }, { shape: "box" }],
    },
    {
        slug: "mesh-torus",
        gold: goldJsonTorus as unknown as SampleGold,
        build: buildTorus,
    },
    {
        slug: "ragdoll-ragdoll",
        gold: goldJsonRagdoll as unknown as SampleGold,
        build: buildRagdoll,
    },
    {
        slug: "determinism-falling-ragdolls",
        gold: goldJsonFallingRagdolls as unknown as SampleGold,
        build: buildFallingRagdolls,
        knobPoints: [{ grid: 1 }, { grid: 2 }, { grid: 4 }],
    },
];
