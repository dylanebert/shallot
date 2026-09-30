import { build } from "@dylanebert/shallot/app";
import { World, Time } from "@dylanebert/shallot/ecs";
import { drainLog, probeTexture } from "@dylanebert/shallot/runtime";

void [build, World, Time, drainLog, probeTexture];
