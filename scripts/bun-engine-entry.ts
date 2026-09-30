import { createApp } from "@dylanebert/shallot/app";
import { Time, World } from "@dylanebert/shallot/ecs";
import { drainLog, probeTexture } from "@dylanebert/shallot/runtime";

void [createApp, World, Time, drainLog, probeTexture];
