import { build } from "@dylanebert/shallot/app";
import { State, Time } from "@dylanebert/shallot/ecs";
import { drainLog, probeTexture } from "@dylanebert/shallot/runtime";

void [build, State, Time, drainLog, probeTexture];
