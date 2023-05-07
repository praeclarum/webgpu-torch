export * from "./shape";
export * from "./tensor";
export * from "./factories";
export * from "./ops";

export const foo = 42;

export * as functions from "./functions";

export { hasWebGPU } from "./webgpu";

import { discoverWebGPUDevicesAsync } from "./devices";

export async function initWebGPUAsync() {
    await discoverWebGPUDevicesAsync();
}
