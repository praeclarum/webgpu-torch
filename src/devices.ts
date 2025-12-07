import type { Device, DeviceType, Deviceish } from "./device";
import { DeviceCPU } from "./device_cpu";
import { DeviceWebGPU } from "./device_webgpu";

export const cpuDevice = new DeviceCPU();
let webgpuDevice: DeviceWebGPU | null = null;

const devices: { [id: string]: Device } = {
    cpu: cpuDevice,
};

export async function discoverWebGPUDevicesAsync(): Promise<boolean> {
    if (!(navigator as any).gpu) {
        // console.warn("No WebGPU devices found");
        return false;
    }
    const id = "webgpu";
    if (id in devices) {
        return true;
    }
    const adapter = (await navigator.gpu.requestAdapter())!;
    const f16_supported = adapter.features.has("shader-f16");
    const timestamp_supported = adapter.features.has("timestamp-query");

    const { limits } = adapter;
    const requiredLimits: Record<string, number> = {};
    for (const prop of [
        "maxBufferSize",
        "maxComputeInvocationsPerWorkgroup",
        "maxComputeWorkgroupSizeX",
        "maxComputeWorkgroupSizeY",
        "maxComputeWorkgroupSizeZ",
        "maxComputeWorkgroupStorageSize",
        "maxComputeWorkgroupsPerDimension",
        "maxUniformBufferBindingSize",
        "maxUniformBuffersPerShaderStage",
        "maxStorageBufferBindingSize",
        "maxStorageBuffersPerShaderStage",
    ] as const) {
        requiredLimits[prop] = limits[prop];
    }
    const device = await adapter.requestDevice({
        requiredFeatures: [
            ...(f16_supported ? ["shader-f16" as const] : []),
            ...(timestamp_supported ? ["timestamp-query" as const] : []),
        ],
        requiredLimits,
    });
    const dev = new DeviceWebGPU(id, adapter, device);
    devices[id] = dev;
    webgpuDevice = dev;
    console.log("Found WebGPU device", device);
    return true;
}

export function getDevice(device?: Deviceish | null): Device {
    if (device === null || device === undefined) {
        return webgpuDevice || cpuDevice;
    } else if (typeof device === "string") {
        if (device in devices) {
            return devices[device];
        } else {
            const found = findDeviceWithType(device as DeviceType);
            if (found) {
                return found;
            }
            throw new Error(`Device ${device} not found`);
        }
    } else {
        return device;
    }
}

function findDeviceWithType(type: DeviceType): Device | null {
    for (const id in devices) {
        if (devices[id].type === type) {
            return devices[id];
        }
    }
    return null;
}
