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
export { Glaze, GlazePlugin, Tonemap } from "../transitional/glaze";
export { Color, MeshInstance, PartPlugin } from "../transitional/part";
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
    DepthPrepass,
    MAX_CASCADES,
    MAX_POINT_CASTERS,
    Material,
    PickingPrepass,
    PointShadows,
    Shadow,
    StandardRenderer,
    StandardRenderingPlugin,
    SunShadows,
} from "./rendering";

import { BrowserInputPlugin, InputPlugin } from "../core/input";
import { RenderingPlugin } from "../core/rendering";
import type { Plugin } from "../engine";
import { setDefaultLoading, setDefaultPlugins } from "../engine/app";
import { GlazePlugin } from "../transitional/glaze";
import { PartPlugin } from "../transitional/part";
import { shallotDark } from "./loading";
import { StandardRenderingPlugin } from "./rendering";

export const DEFAULT_PLUGINS: readonly Plugin[] = [
    InputPlugin,
    BrowserInputPlugin,
    RenderingPlugin,
    PartPlugin,
    StandardRenderingPlugin,
    GlazePlugin,
];

setDefaultPlugins(DEFAULT_PLUGINS);
setDefaultLoading(shallotDark);
