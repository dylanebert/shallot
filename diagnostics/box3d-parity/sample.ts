import { readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { nativeBinary } from "./native";

/** Compile the selected upstream classes verbatim, without the samples app's graphics dependencies. */
export function sampleBinary(samples = join(process.env.BOX3D ?? "", "samples")): string {
    const manifold = readFileSync(join(samples, "sample_manifold.cpp"), "utf8");
    const geometry = readFileSync(join(samples, "sample_geometry.cpp"), "utf8");
    const shapes = readFileSync(join(samples, "sample_shapes.cpp"), "utf8");
    const bodies = readFileSync(join(samples, "sample_bodies.cpp"), "utf8");
    const character = readFileSync(join(samples, "sample_character.cpp"), "utf8");
    const continuous = readFileSync(join(samples, "sample_continuous.cpp"), "utf8");
    const compound = readFileSync(join(samples, "sample_compound.cpp"), "utf8");
    const collision = readFileSync(join(samples, "sample_collision.cpp"), "utf8");
    const ragdoll = readFileSync(join(samples, "sample_ragdoll.cpp"), "utf8");
    const mesh = readFileSync(join(samples, "sample_mesh.cpp"), "utf8");
    const tree = readFileSync(join(samples, "sample_tree.cpp"), "utf8");
    const stacking = readFileSync(join(samples, "sample_stacking.cpp"), "utf8");
    const world = readFileSync(join(samples, "sample_world.cpp"), "utf8");
    const determinism = readFileSync(join(samples, "sample_determinism.cpp"), "utf8");
    const robustness = readFileSync(join(samples, "sample_robustness.cpp"), "utf8");
    const joints = readFileSync(join(samples, "sample_joint.cpp"), "utf8");
    const sample = readFileSync(join(samples, "sample.cpp"), "utf8");
    const events = readFileSync(join(samples, "sample_events.cpp"), "utf8");
    const benchmark = readFileSync(join(samples, "sample_benchmark.cpp"), "utf8");
    function section(source: string, start: string, end?: string): string {
        const first = source.indexOf(start);
        const last = end === undefined ? source.length : source.indexOf(end, first);
        if (first < 0 || last < 0) throw new Error(`missing sample source boundary: ${start} / ${end}`);
        return source.slice(first, last);
    }
    const sources = [
        "namespace ManifoldSamples {",
        section(manifold, "class Manifold :", "class SphereAndSphere :"),
        ...[["SphereAndSphere","sampleCollideSpheres"],["CapsuleAndSphere","sampleSphereAndCapsule"],["HullAndSphere","sampleSphereAndHull"],["TriangleAndSphere","sampleSphereAndTriangle"],["CapsuleAndCapsule","sampleCapsuleAndCapsule"],["CapsuleAndHull","sampleCapsuleAndHull"],["TriangleAndCapsule","sampleCapsuleAndTriangle"],["HullAndHull","sampleCollideHulls"],["TriangleAndHull","sampleHullAndTriangle"]].map(([name,end])=>section(manifold,`class ${name} :`,`static int ${end}`)),
        "}",
        readFileSync(join(import.meta.dir,"sample-manifold.h"),"utf8"),
        "namespace Geometry {\n#define b3ComputeHullMass RecordedHullMass\n#define b3ComputeCapsuleMass RecordedCapsuleMass",
        ...["BoxHull", "Hull", "HullReduction", "HullTransform", "CapsuleMass"].map(name => section(geometry, `class ${name} :`, `static int sample${name}`)),
        "#undef b3ComputeHullMass\n#undef b3ComputeCapsuleMass\n}",
        ...[["RagdollOnBox", "sampleRagdollOnBox"], ["RagdollOnMesh", "sampleRagdollMesh"], ["RagdollPile", "sampleRagdollPile"], ["RagdollIncline", "sampleRagdollIncline"]].map(([name,end]) => section(ragdoll, `class ${name} :`, `static int ${end}`)),
        section(mesh, "enum class ShapeType", "class GridMesh"),
        section(mesh, "class MeshReflection :", "static int sampleMeshReflection"),
        section(mesh, "struct CastContext", "static int sampleHeightField"),
        section(mesh, "static float ComputeInternalSurfaceArea", "static int sampleMeshViewer"),
        section(mesh, "class MeshCreationBenchmark :", "static int sampleMeshCreationBenchmark"),
        'namespace TreeSample {\n#define fopen TreeOpen\n#define b3DynamicTree_Save TreeSave\n#define b3DynamicTree_Load TreeLoad\n#define printf(...) fprintf(stderr, __VA_ARGS__)',
        section(tree, "constexpr int colorCount", "static int sampleTree"),
        '#undef fopen\n#undef b3DynamicTree_Save\n#undef b3DynamicTree_Load\n#undef printf\n}',
        ...[["GridMesh", "sampleGridMesh"], ["BigBoxMesh", "sampleBigBoxMesh"], ["BoxMesh", "sampleBoxMesh"], ["VoxelMesh", "sampleVoxelMesh"], ["HollowBox", "sampleHollowBox"]].map(([name,end]) => section(mesh, `class ${name} :`, `static int ${end}`)),
        ...[["SimpleCompound", "sampleSimple"], ["CompoundSpheres", "sampleCompoundSpheres"], ["CompoundHulls", "sampleCompoundHulls"], ["TileFloor", "sampleTileFloor"], ["MeshTile", "sampleCompoundMesh"]].map(([name, end]) => section(compound, `class ${name} :`, `static int ${end}`)),
        section(readFileSync(join(samples, "mover.h"), "utf8"), "class Sample;"),
        section(readFileSync(join(samples, "mover.cpp"), "utf8"), "static bool MoverFilterCallback"),
        section(readFileSync(join(process.env.BOX3D ?? "", "samples/sample.h"), "utf8"), "struct CastClosestContext", "float CastClosestCallback"),
        section(sample, "float CastClosestCallback("),
        section(compound, "static bool OverlapResultFcn", "static int sampleVillage"),
        section(character, "struct MoverImpact", "static int sampleMover"),
        section(character, "struct ClosestShapeCastContext", "static int sampleRBCharacter"),
        ...["CapsulePlane", "MoverOverlap"].map(name => section(character, `class ${name} :`, `static int sample${name}`)),
        section(sample, "b3BodyId Sample::AddGroundBox(", "struct RowDef"),
        "#define printf(...) fprintf(stderr, __VA_ARGS__)",
        ...["ThinWall", "BounceHouse", "SpinningStick", "NeedleMesh", "HumpMesh", "IsFast", "Stall"].map(name => section(continuous, `class ${name} :`, `static int sample${name}`)),
        section(continuous, "class BulletVersusStack :", "static int sampleContinuous6"),
        section(continuous, "struct IndexPair", "static int sampleMeshDrop"),
        "#undef printf",
        ...["InclinedPlane", "RollingResistance", "HighResistance", "IsotropicFriction", "SlideTwist", "Restitution", "StaticInvoke", "Wind", "ConveyorBelt", "ConveyorMesh", "WindDrop", "WindFlap"].map(name => section(shapes, `class ${name} :`, `static int sample${name}`)),
        ...[["ClassRing", "sampleClassRing"], ["BodyCast", "sampleBodyCast"], ["GyroscopicPrecession", "sampleGyroscopicPrecession"], ["BodyType", "sampleBodyType"], ["Weeble", "sampleWeeble"], ["DisableBody", "sampleDisable"], ["SpinningBooks", "sampleBook"], ["GyroscopicTorque", "sampleGyroscopicTorque"], ["Kinematic", "sampleKinematic"], ["LockMixing", "sampleLockMixing"], ["FixedRotation", "sampleFixedRotation"], ["OffsetKinematic", "sampleOffsetKinematic"]].map(([name, end]) => section(bodies, `class ${name} :`, `static int ${end}`)),
        section(benchmark.slice(benchmark.lastIndexOf("class BenchmarkLargeWorld :")), "class BenchmarkLargeWorld :", "static int sampleLargeWorld"),
        ...["BenchmarkLargePyramid", "BenchmarkWidePyramid", "BenchmarkManyPyramids", "BenchmarkJointGrid", "FallingBoxes", "BenchmarkRain", "BenchmarkJunkyard", "BenchmarkWasher", "BenchmarkConvexPile", "BenchmarkFallingTrees", "BenchmarkExplosion", "BenchmarkDestruction", "BenchmarkChains", "BenchmarkHeightField", "BenchmarkHull"].map(name =>
            section(benchmark, `class ${name} :`, name === "BenchmarkHull" ? "static int sampleBenchmarkHull" : name === "BenchmarkHeightField" ? "static int sampleHeightFieldBenchmark" : name === "BenchmarkChains" ? "static int sampleBenchmarkChains" : name === "BenchmarkWasher" ? "static int benchmarkWasher" : `static int sample${name.replace("Benchmark", "")}`)),
        section(benchmark, "inline bool operator<", "class BenchmarkLargePyramid"),
        section(benchmark, "struct ShapeUserData", "static int benchmarkSensor"),
        section(benchmark, "class CandyCups :", "static int sampleSmallConvexes"),
        readFileSync(join(samples, "../shared/overflow_color.c"), "utf8"),
        '#include "stability.h"\n#include "determinism.h"',
        '#define printf(...) fprintf(stderr, __VA_ARGS__)',
        ...["FallingRagdolls", "WavePile", "QuerySpawn", "MeshDropDeterminism"].map(name => section(determinism, `class ${name} :`, `static int sample${name}`)),
        '#undef printf',
        ...[["FarStack", "sampleLargeWorld"], ["FarPyramid", "sampleFarPyramid"], ["FarRagdolls", "sampleFarRagdolls"], ["FarMeshDrop", "sampleFarMeshDrop"]].map(([name, end]) => section(world, `class ${name} :`, `static int ${end}`)),
        ...["HighMassRatio1", "TinyPyramid", "OverlapRecovery", "OverflowColorPile"].map(name => section(robustness, `class ${name} :`, `static int sample${name}`)),
        ...["BoxStack", "CardHouse", "SphereStack", "CapsuleStack", "JengaStack", "SingleBox", "Cylinder", "CylinderStack", "Dominoes", "Wedge", "Arch", "DoubleDomino", "Pyramid2D", "EdgeCrossing"].map(name => section(stacking, `class ${name} :`, `static int sample${name}`)),
        ...["DistanceJoint", "SphericalJoint", "MotionLocks", "BallAndChain", "FoldOut", "TopDownFriction", "WeldJoint", "GearLift", "Driving"].map(name => section(joints, `class ${name} :`, `static int sample${name}`)),
        section(joints, "class Door :", "static int sampleDoor").replace(/\n\tvoid MouseDown\([\s\S]*?(?=\n\tbool DrawControls)/, ""),
        section(joints, "class Bridge :", "static int sampleBridgeIndex"),
        ...["Filter", "Prismatic", "Parallel", "Revolute", "Wheel", "Motor"].map((name) =>
            section(joints, `class ${name}Joint :`, `static int sample${name}Joint`)),
        section(events, "class SensorVisit :", "static int sampleSensorVisit"),
        section(events, "class HitEvent :", "static int sampleHitEvent"),
        section(events, "class MoveEvent :", "static int sampleMoveEvent"),
        section(events, "class SensorHits :", "static int sampleSensorHits"),
        section(events, "class PersistentContact :", "static int samplePersistentContact"),
        section(events, "class JointEvent :", "static int sampleJointEvent"),
        section(events, "struct BodyUserData", "static int sampleContactEvent"),
        "#undef b3World_CastRayClosest\n#undef b3World_CastShape\n#undef b3Body_CastRay",
        readFileSync(join(import.meta.dir, "sample-collision.h"), "utf8"),
        "namespace Collision {\n#define b3World_CastRay Ray\n#define b3World_CastRayClosest Closest\n#define b3World_CastShape Shape\n#define b3World_OverlapShape Overlap\n#define b3Body_CastRay BodyRay\n#define b3ShapeDistance Distance\n#define b3ShapeCast Cast\n#define b3TimeOfImpact TOI",
        section(collision, "class RayCurtain :", "static int sampleRayCurtain"),
        section(collision, "struct CastContext", "static int sampleCastWorld"),
        ...["MeshScale", "ShapeCast", "OverlapWorld", "LongRayCast", "InitialOverlap", "CapsuleCastRay", "ShapeCastDebug", "DistanceDebug", "ShapeDistance", "TimeOfImpact"].map(name => section(collision, `class ${name} :`, `static int sample${name}`)),
        "#undef b3World_CastRay\n#undef b3World_CastRayClosest\n#undef b3World_CastShape\n#undef b3World_OverlapShape\n#undef b3Body_CastRay\n#undef b3ShapeDistance\n#undef b3ShapeCast\n#undef b3TimeOfImpact\n}",
    ].join("\n");
    const grab = section(sample, "\t\t// Ctrl click grabs", "\n\telse if ( modifiers & MOD_SHIFT )")
        .replace(/\t\tPickRay pickRay = m_camera->BuildPickRay\( p.x, p.y \);/, "")
        .replace(/\n\t}\s*$/, "");
    const up = section(sample, "void Sample::MouseUp(", "void Sample::MouseMove(");
    const meshLoader = [
        readFileSync(join(process.env.BOX3D ?? "", "samples/mesh_loader.h"), "utf8"),
        "#define TINYOBJLOADER_IMPLEMENTATION\n#define TINYOBJLOADER_USE_MAPBOX_EARCUT",
        readFileSync(join(process.env.BOX3D ?? "", "samples/tiny_obj_loader.h"), "utf8").replace('#include "earcut.h"', ''),
        readFileSync(join(process.env.BOX3D ?? "", "samples/mesh_loader.cpp"), "utf8")
            .replace('#include "mesh_loader.h"', '')
            .replace('#include "tiny_obj_loader.h"', '')
            .replace('reader.ParseFromFile( path, readerConfig )', 'reader.ParseFromFile( (std::string(getenv("BOX3D")) + "/" + path), readerConfig )'),
    ].join("\n");
    const source = readFileSync(join(import.meta.dir, "sample.cpp"), "utf8")
        .replace("// SAMPLE_EARCUT", readFileSync(join(process.env.BOX3D ?? "", "samples/earcut.h"), "utf8"))
        .replace("// SAMPLE_MESH_LOADER", meshLoader)
        .replace("// SAMPLE_GRAB", grab)
        .replace("// SAMPLE_UP", up.slice(up.indexOf("{")))
        .replace("// SAMPLE_TREE_RECORDS", readFileSync(join(import.meta.dir, "sample-tree.h"), "utf8"))
        .replace("// SAMPLE_EVENTS", readFileSync(join(import.meta.dir, "sample-events.h"), "utf8"))
        .replace("// SAMPLE_SOURCES", readFileSync(join(import.meta.dir, "sample-compounds.h"), "utf8") + "\n" + readFileSync(join(import.meta.dir, "sample-meshes.h"), "utf8") + "\n" + readFileSync(join(import.meta.dir, "sample-mesh-frames.h"), "utf8") + "\n" + sources);
    const path = join(tmpdir(), `box3d-sample-${Bun.hash(source).toString(16)}.cpp`);
    writeFileSync(path, source);
    const shared = ["benchmarks.c", "determinism.c", "stability.c"].map(name => {
        const text = readFileSync(join(samples, "../shared", name), "utf8");
        return name === "determinism.c" ? `#define CreateGroup CreateDeterminismGroup\n${text}\n#undef CreateGroup` : text;
    }).join("\n");
    const sharedPath = join(tmpdir(), `box3d-sample-shared-${Bun.hash(shared).toString(16)}.c`);
    writeFileSync(sharedPath, shared);
    return nativeBinary(path, undefined, sharedPath);
}
