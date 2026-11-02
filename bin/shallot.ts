#!/usr/bin/env node
import { main } from "../src/cli/index";

await main(process.argv.slice(2));
