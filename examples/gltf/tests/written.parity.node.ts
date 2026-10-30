import { setDefaultTimeout } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { parity } from "./parity";

setDefaultTimeout(TIMEOUT.node);
parity("written");
