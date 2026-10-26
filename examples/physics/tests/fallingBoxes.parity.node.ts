import { setDefaultTimeout } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { benchmarkParity } from "./benchmark-parity";

setDefaultTimeout(TIMEOUT.node);

for (const threads of [2, 4]) benchmarkParity("fallingBoxes", threads);
