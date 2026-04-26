import {
    Transform,
    Part,
    Shape,
    Body,
    Camera,
    Character,
    ChildOf,
    Viewport,
    Shadows,
    Tonemap,
    AmbientLight,
    DirectionalLight,
    PointLight,
    RenderPlugin,
    AudioPlugin,
    PhysicsPlugin,
    Sound,
    Listener,
    instrument,
    Player,
    PlayerPlugin,
} from "@dylanebert/shallot";
import {
    Audio,
    SoundVoices,
    instrumentRegistry,
    running,
    gate,
} from "@dylanebert/shallot/audio/core";
import type { Plugin, State, System } from "@dylanebert/shallot";
import {
    AcousticPlugin,
    Acoustic,
    AcousticMaterial,
    MaterialPreset,
} from "@dylanebert/shallot/extras";
import { BenchConfig } from "../config";

export type AudioRoom = "bathroom" | "living" | "cathedral" | "anechoic";

export const AUDIO_ROOMS: AudioRoom[] = ["bathroom", "living", "cathedral", "anechoic"];

interface RoomDef {
    width: number;
    height: number;
    depth: number;
    color: number;
    material: number;
    floorMaterial?: number;
}

const ROOM_DEFS: Record<AudioRoom, RoomDef> = {
    bathroom: {
        width: 6,
        height: 3,
        depth: 6,
        color: 0xc0c8cc,
        material: MaterialPreset.Ceramic,
    },
    living: {
        width: 6,
        height: 3,
        depth: 6,
        color: 0x8c7860,
        material: MaterialPreset.Plaster,
        floorMaterial: MaterialPreset.Carpet,
    },
    cathedral: { width: 30, height: 15, depth: 20, color: 0x908880, material: MaterialPreset.Rock },
    anechoic: { width: 6, height: 3, depth: 6, color: 0x303030, material: MaterialPreset.Carpet },
};

const ROOM_CENTER_Z = -5;
const KICK_INTERVAL_MS = 333;
const T = 0.2;

let audioState: State | null = null;
let wallEntities: number[] = [];
let bodyEntity = -1;
let lightEntity = -1;

function makeWall(
    state: State,
    x: number,
    y: number,
    z: number,
    sx: number,
    sy: number,
    sz: number,
    color: number,
    material: number,
): number {
    const eid = state.addEntity();
    state.addComponent(eid, Transform);
    state.addComponent(eid, Part);
    state.addComponent(eid, Body);
    state.addComponent(eid, AcousticMaterial);
    Transform.posX[eid] = x;
    Transform.posY[eid] = y;
    Transform.posZ[eid] = z;
    Part.shape[eid] = Shape.Box;
    Part.sizeX[eid] = sx;
    Part.sizeY[eid] = sy;
    Part.sizeZ[eid] = sz;
    Part.color[eid] = color;
    Part.roughness[eid] = 0.9;
    Body.mass[eid] = 0;
    AcousticMaterial.preset[eid] = material;
    return eid;
}

const DOOR_WIDTH = 1.0;
const DOOR_HEIGHT = 2.2;
const GROUND_SIZE = 50;

function buildRoom(state: State, room: AudioRoom) {
    for (const eid of wallEntities) state.removeEntity(eid);
    wallEntities = [];

    const def = ROOM_DEFS[room];
    const { width: S, height: H, depth: D, color, material: m } = def;
    const fm = def.floorMaterial ?? m;
    const cz = ROOM_CENTER_Z;

    wallEntities.push(makeWall(state, 0, -T / 2, cz, GROUND_SIZE, T, GROUND_SIZE, color, fm));
    wallEntities.push(makeWall(state, 0, H + T / 2, cz, S, T, D, color, m));
    wallEntities.push(makeWall(state, 0, H / 2, cz - D / 2 - T / 2, S, H, T, color, m));
    const frontZ = cz + D / 2 + T / 2;
    const pillarW = (S - DOOR_WIDTH) / 2;
    wallEntities.push(
        makeWall(state, -(pillarW + DOOR_WIDTH) / 2, H / 2, frontZ, pillarW, H, T, color, m),
    );
    wallEntities.push(
        makeWall(state, (pillarW + DOOR_WIDTH) / 2, H / 2, frontZ, pillarW, H, T, color, m),
    );
    const headerH = H - DOOR_HEIGHT;
    if (headerH > 0.01) {
        wallEntities.push(
            makeWall(state, 0, DOOR_HEIGHT + headerH / 2, frontZ, DOOR_WIDTH, headerH, T, color, m),
        );
    }
    wallEntities.push(makeWall(state, -S / 2 - T / 2, H / 2, cz, T, H, D, color, m));
    wallEntities.push(makeWall(state, S / 2 + T / 2, H / 2, cz, T, H, D, color, m));

    for (const eid of emitterEntities) {
        Transform.posY[eid] = Math.min(1, H / 2);
    }

    Transform.posY[lightEntity] = H - 0.3;
    Transform.posZ[lightEntity] = cz;
    PointLight.radius[lightEntity] = Math.max(S, D) * 1.5;

    Transform.posY[bodyEntity] = 0.9;
    Transform.posZ[bodyEntity] = cz + Math.min(4, D / 2 - 0.5);
}

export function setAudioRoom(room: AudioRoom) {
    if (!audioState) return;
    buildRoom(audioState, room);
}

function registerKick() {
    instrument(
        {
            nodes: {
                osc: { type: "oscillator" },
                filter: { type: "filter", input: "osc" },
                env: { type: "envelope", input: "filter" },
                vol: { type: "gain", input: "env" },
            },
            output: "vol",
            volumeParam: "vol.level",
            pitchParams: ["osc.frequency"],
            values: {
                "osc.frequency": 800,
                "osc.waveform": 2,
                "filter.cutoff": 4000,
                "filter.q": 0.7,
                "filter.mode": 0,
                "filter.mix": 1,
                "env.attack": 0.005,
                "env.decay": 0.06,
                "env.sustain": 0,
                "env.release": 0.02,
                "vol.level": 0.7,
            },
        },
        "kick",
    );
}

let emitterEntities: number[] = [];
let soundInitialized = false;
let nextKickAt = 0;

const KickSystem: System = {
    group: "simulation",
    update(state: State) {
        const audio = Audio.from(state);
        if (!audio || !running(audio)) return;
        if (emitterEntities.length === 0) return;

        if (!soundInitialized) {
            const instId = instrumentRegistry.getByName("kick");
            if (instId === undefined) return;

            for (const eid of emitterEntities) {
                state.addComponent(eid, Sound);
                Sound.instrument[eid] = instId;
                Sound.spatial[eid] = 1;
                Sound.loop[eid] = 1;
                state.addComponent(eid, Acoustic);
            }
            soundInitialized = true;
            nextKickAt = performance.now() + KICK_INTERVAL_MS;
            return;
        }

        const now = performance.now();
        if (now < nextKickAt) return;
        nextKickAt = now + KICK_INTERVAL_MS;

        const ss = SoundVoices.from(state);
        if (!ss) return;
        for (const eid of emitterEntities) {
            const voice = ss.voices.get(eid);
            if (!voice) continue;
            gate(audio, voice.slot, 0);
            gate(audio, voice.slot, 1);
        }
    },
};

export function buildAudioPlugin(room: AudioRoom = "living", sourceCount = 1): Plugin {
    registerKick();

    return {
        name: "AudioScenario",
        dependencies: [RenderPlugin, AudioPlugin, PhysicsPlugin, PlayerPlugin, AcousticPlugin],
        systems: [KickSystem],
        initialize(state: State) {
            audioState = state;

            const body = state.addEntity();
            state.addComponent(body, Transform);
            state.addComponent(body, Part);
            state.addComponent(body, Body);
            state.addComponent(body, Character);
            state.addComponent(body, Player);
            Part.shape[body] = Shape.Box;
            Part.sizeX[body] = 0.6;
            Part.sizeY[body] = 1.8;
            Part.sizeZ[body] = 0.6;
            Part.opacity[body] = 0;
            Body.mass[body] = 0;
            Body.friction[body] = 0.5;
            Player.sprint[body] = 3;
            bodyEntity = body;

            const cam = state.addEntity();
            state.addComponent(cam, Transform);
            state.addComponent(cam, Camera);
            state.addComponent(cam, Viewport);
            state.addComponent(cam, Tonemap);
            state.addComponent(cam, Shadows);
            state.addComponent(cam, Listener);
            state.addComponent(cam, BenchConfig);
            state.addRelation(cam, ChildOf, body);

            const ambient = state.addEntity();
            state.addComponent(ambient, Transform);
            state.addComponent(ambient, AmbientLight);
            AmbientLight.intensity[ambient] = 0.1;

            const sun = state.addEntity();
            state.addComponent(sun, Transform);
            state.addComponent(sun, DirectionalLight);
            DirectionalLight.intensity[sun] = 0.6;
            DirectionalLight.directionX[sun] = -0.5;
            DirectionalLight.directionY[sun] = -0.7;
            DirectionalLight.directionZ[sun] = -0.5;

            const light = state.addEntity();
            state.addComponent(light, Transform);
            state.addComponent(light, PointLight);
            PointLight.color[light] = 0xffeedd;
            PointLight.intensity[light] = 0.8;
            PointLight.shadows[light] = 1;
            lightEntity = light;

            const def = ROOM_DEFS[room];
            emitterEntities = [];
            for (let i = 0; i < sourceCount; i++) {
                const emitter = state.addEntity();
                state.addComponent(emitter, Transform);
                state.addComponent(emitter, Part);
                const spread = Math.min(def.width, def.depth) * 0.4;
                const angle = (i / sourceCount) * Math.PI * 2;
                const radius = sourceCount > 1 ? spread * Math.sqrt(i / sourceCount) : 0;
                Transform.posX[emitter] = radius * Math.cos(angle);
                Transform.posZ[emitter] = ROOM_CENTER_Z + radius * Math.sin(angle);
                Part.shape[emitter] = Shape.Sphere;
                Part.sizeX[emitter] = 0.3;
                Part.sizeY[emitter] = 0.3;
                Part.sizeZ[emitter] = 0.3;
                Part.color[emitter] = 0xdd6644;
                Part.emission[emitter] = 0xdd6644;
                Part.emissionIntensity[emitter] = 2;
                emitterEntities.push(emitter);
            }

            buildRoom(state, room);
        },
    };
}
