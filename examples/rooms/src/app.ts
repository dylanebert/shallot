import {
    Camera,
    capsule,
    DirectionalLight,
    GlobalAmbientLight,
    lookAtRotation,
    Materials,
    Meshes,
    MeshInstance,
    MeshMaterial,
    type Plugin,
    StandardMaterial,
    StandardRenderer,
    Transform,
} from "@dylanebert/shallot";
import {
    DiagnosticsOverlayPlugin,
    internText,
    Justify,
    Text,
    TextPlugin,
} from "@dylanebert/shallot/extras";
import { Anchor } from "@dylanebert/shallot/rendering";
import { Tint } from "./demo";
import { ClientControls, ClientSession, scene } from "./network";
import { LocalControls, LocalPlayer, UpdatePlayerControlSystem } from "./player";
export const RoomFrame = { create: (): { pump?: (delta: number) => void } => ({}) };

const RoomView = { create: () => ({ avatar: 0 }) };

export const AppPlugin: Plugin = {
    name: "RoomsApp",
    dependencies: [ClientControls, TextPlugin, DiagnosticsOverlayPlugin],
    recovery: "stateless",
    initialize(world) {
        world.resource(LocalControls).enabled = true;
        const ambient = world.resource(GlobalAmbientLight);
        ambient.color = 0xd8e4ff;
        ambient.brightness = 380;
        const sun = world.create();
        world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, 1, -2, -0.35) });
        world.add(sun, DirectionalLight, {
            color: 0xffe8cc,
            illuminance: 6000,
            maximumDistance: 50,
            depthBias: 0.0005,
            shadowNormalBias: 1.8,
        });
        world.storage(DirectionalLight).shadowMapsEnabled.set(sun, 1);
        const avatarMesh = world.resource(Meshes).add({ name: "Avatar", ...capsule(0.3, 1.2) });
        world.resource(RoomView).avatar = avatarMesh;
        for (const [slot, { eye }] of scene(world).players.entries()) {
            world.add(eye, Camera, { fov: 70, clearColor: 0x191816, order: slot });
            world.add(eye, StandardRenderer);
        }
        const players = scene(world).players.map(({ player }) => player);
        for (const eid of world.query([Tint])) {
            world.add(eid, MeshInstance, players.includes(eid) ? { mesh: avatarMesh } : undefined);
            world.add(
                eid,
                MeshMaterial,
                world
                    .resource(Materials)
                    .add(StandardMaterial({ baseColor: world.get(eid, Tint)!.color })),
            );
        }
        const hint = world.create();
        world.add(hint, Transform, {
            translation: [0, 0.505, 7.5],
            rotation: [-Math.SQRT1_2, 0, 0, Math.SQRT1_2],
        });
        world.add(hint, Anchor, { offset: Anchor.CENTER });
        const touch = matchMedia("(any-pointer: coarse)").matches;
        world.add(hint, Text, {
            content: internText(
                world,
                touch ? "STICK  Move\nDRAG  Look\nJump" : "WASD  Move\nSPACE  Jump\nMOUSE  Look",
            ),
            fontSize: touch ? 0.26 : 0.5,
            maxWidth: 5,
            justify: Justify.Center,
            color: 0xeee9df,
        });
    },
    systems: [
        {
            name: "pump-server",
            group: "setup",
            first: true,
            update(world) {
                world.resource(RoomFrame).pump?.(world.time.realDeltaTime);
            },
        },
        {
            name: "local-view",
            group: "simulation",
            before: [UpdatePlayerControlSystem],
            update(world) {
                const client = world.resource(ClientSession).client;
                if (!client || client.slot < 0) return;
                const { player, eye } = scene(world).players[client.slot]!;
                if (!world.has(player, LocalPlayer)) world.add(player, LocalPlayer);
                world.storage(LocalPlayer).camera.set(player, eye);
                for (const [slot, avatar] of scene(world).players.entries()) {
                    if (slot !== client.slot && world.has(avatar.eye, Camera)) {
                        world.remove(avatar.eye, Camera);
                        world.remove(avatar.eye, StandardRenderer);
                    }
                    const visible = slot !== client.slot && !!(client.roster & (1 << slot));
                    const hasMesh = world.has(avatar.player, MeshInstance);
                    if (visible && !hasMesh)
                        world.add(avatar.player, MeshInstance, {
                            mesh: world.resource(RoomView).avatar,
                        });
                    if (!visible && hasMesh) world.remove(avatar.player, MeshInstance);
                }
            },
        },
    ],
};
