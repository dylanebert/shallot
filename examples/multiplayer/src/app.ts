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
    registerDebugChord,
    Text,
    TextPlugin,
} from "@dylanebert/shallot/extras";
import { Anchor } from "@dylanebert/shallot/rendering";
import { Tint } from "./demo";
import { ClientControls, ClientSession, scene } from "./network";
import { LocalControls, LocalPlayer, UpdatePlayerControlSystem } from "./player";
import { network } from "./runtime";

export const AppPlugin: Plugin = {
    name: "MultiplayerApp",
    dependencies: [ClientControls, TextPlugin, DiagnosticsOverlayPlugin],
    recovery: "stateless",
    initialize(world) {
        const topology = network(world);
        if (new URLSearchParams(location.search).has("record")) {
            registerDebugChord(world, "KeyR", () => {
                const blob = new Blob([JSON.stringify(topology.recordings(world))], {
                    type: "application/json",
                });
                const url = URL.createObjectURL(blob);
                const link = document.createElement("a");
                link.href = url;
                link.download = "multiplayer-recording.json";
                link.click();
                setTimeout(() => URL.revokeObjectURL(url), 0);
            });
        }
        world.resource(LocalControls).enabled = true;
        world.onDispose(() => topology.dispose());
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
        world.add(hint, Text, {
            content: internText(world, "WASD  Move\nSPACE  Jump\nMOUSE  Look"),
            fontSize: 0.5,
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
                network(world).pump(world.time.realDeltaTime);
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
                const otherEye = scene(world).players[1 - client.slot]!.eye;
                world.remove(otherEye, Camera);
                world.remove(otherEye, StandardRenderer);
                if (!world.has(eye, Camera)) {
                    world.add(eye, Camera, { fov: 70, clearColor: 0x191816 });
                    world.add(eye, StandardRenderer);
                }
                world.remove(player, MeshInstance);
            },
        },
    ],
};
