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
export { Character, CharacterPlugin, SweepCharactersSystem } from "../transitional/character";
export {
    Body,
    BodyType,
    body,
    type ContactEvents,
    createParallelJoint,
    createRevoluteJoint,
    createSoftJoint,
    createSphericalJoint,
    createWheelJoint,
    getContactEvents,
    getJointEvents,
    hashPhysics,
    Joint,
    JointType,
    type ParallelJointConfig,
    Physics,
    PhysicsPlugin,
    PhysicsWorld,
    physicsCounters,
    physicsStepConfig,
    physicsWorld,
    type RevoluteJointConfig,
    readBody,
    restorePhysics,
    ShapeKind,
    SoftJoint,
    type SoftJointConfig,
    type SphericalJointConfig,
    Spring,
    setKinematic,
    setVelocity,
    snapshotPhysics,
    type WheelJointConfig,
} from "../transitional/physics";
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
    CameraBackground,
    MAX_CASCADES,
    MAX_POINT_CASTERS,
    Materials,
    MeshMaterial3d,
    PartPlugin,
    PointShadows,
    Shadow,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
    SunShadows,
} from "./rendering";

import { BrowserInputPlugin, InputPlugin } from "../core/input";
import { CorePipelinePlugin, RenderingPlugin } from "../core/rendering";
import type { Plugin } from "../engine";
import { setDefaultLoading, setDefaultPlugins } from "../engine/app";
import { shallotDark } from "./loading";
import { PartPlugin, StandardRenderingPlugin } from "./rendering";

export const DEFAULT_PLUGINS: readonly Plugin[] = [
    InputPlugin,
    BrowserInputPlugin,
    RenderingPlugin,
    PartPlugin,
    StandardRenderingPlugin,
    CorePipelinePlugin,
];

setDefaultPlugins(DEFAULT_PLUGINS);
setDefaultLoading(shallotDark);
