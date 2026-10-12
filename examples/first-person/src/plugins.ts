import { CharacterPlugin, PhysicsPlugin, PlayerPlugin, type Plugin } from "@dylanebert/shallot";
import { Demo } from "./demo";

export const APP_PLUGINS: Plugin[] = [PhysicsPlugin, CharacterPlugin, PlayerPlugin, Demo];
