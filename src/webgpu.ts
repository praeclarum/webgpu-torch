import { discoverWebGPUDevicesAsync } from "./devices";

/**
 * Returns true if WebGPU is available, false otherwise.
 * @returns true if WebGPU is available, false otherwise.
 */
export function hasWebGPU() {
    const anavigator = navigator as any;
    if (!anavigator.gpu) { return false; }
    return true;
}

/**
 * Initializes WebGPU. This function must be called before any other functions.
 * @returns true if WebGPU is available, false otherwise.
 */
export async function initWebGPUAsync(): Promise<boolean> {
    return await discoverWebGPUDevicesAsync();
}
