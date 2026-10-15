import { CharacterPlugin, PhysicsPlugin, type Plugin } from "@dylanebert/shallot";
import { Demo } from "./demo";
import { PlayerPlugin } from "./player";

export const APP_PLUGINS: Plugin[] = [PhysicsPlugin, CharacterPlugin, PlayerPlugin, Demo];
