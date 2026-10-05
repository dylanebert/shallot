export {
    Body,
    BodyType,
    DistanceJoint,
    FilterJoint,
    MotorJoint,
    ParallelJoint,
    PrismaticJoint,
    RevoluteJoint,
    ShapeKind,
    SphericalJoint,
    WeldJoint,
    WheelJoint,
} from "../core/physics";
export { composeGlobalTransform, GlobalTransform, Transform } from "../engine";
export {
    AudioPlugin,
    type InstrumentDef,
    Instruments,
    instrument,
    Listener,
    type ModulationDef,
    type NodeDef,
    type NodeType,
    play,
    type SfxPolicy,
    Sound,
    sample,
    sfx,
} from "../transitional/audio";
export { BvhPlugin } from "../transitional/bvh";
export {
    type LoadingOptions,
    minimalDark,
    minimalLight,
    type SplashOptions,
    type SplashProfile,
    shallotDark,
    shallotLight,
} from "./loading";
export {
    type BodyStateOut,
    Character,
    CharacterPlugin,
    GroundState,
    hashPhysics,
    PhysicsWorld,
    physicsWorld,
    readBody,
    restorePhysics,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
    type StepProfile,
    setKinematic,
    setVelocity,
    snapshotPhysics,
    type WorldSnapshot,
} from "./physics";
export {
    CameraBackground,
    DirectionalLightShadowMap,
    MAX_CASCADES,
    MAX_POINT_CASTERS,
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    PointShadows,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "./rendering";

import { BrowserInputPlugin, InputPlugin } from "../core/input";
import { CorePipelinePlugin, RenderingPlugin } from "../core/rendering";
import type { Plugin } from "../engine";
import { setDefaultLoading, setDefaultPlugins } from "../engine/app";
import { shallotDark } from "./loading";
import { MeshRenderPlugin, StandardRenderingPlugin } from "./rendering";

export const DEFAULT_PLUGINS: readonly Plugin[] = [
    InputPlugin,
    BrowserInputPlugin,
    RenderingPlugin,
    MeshRenderPlugin,
    StandardRenderingPlugin,
    CorePipelinePlugin,
];

setDefaultPlugins(DEFAULT_PLUGINS);
setDefaultLoading(shallotDark);
