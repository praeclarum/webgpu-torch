import { discoverWebGPUDevicesAsync } from "./devices";

export function hasWebGPU() {
    const anavigator = navigator as any;
    if (!anavigator.gpu) { return false; }
    return true;
}

export async function initWebGPUAsync() {
    await discoverWebGPUDevicesAsync();
}
