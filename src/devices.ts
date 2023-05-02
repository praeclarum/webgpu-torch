import { Device, DeviceType, Deviceish } from "./device";
import { DeviceCPU } from "./device_cpu";
import { DeviceWebGPU } from "./device_webgpu";

const cpuDevice = new DeviceCPU();
let webgpuDevice: DeviceWebGPU | null = null;

const devices: { [id: string]: Device } = {
    cpu: cpuDevice,
};

export async function discoverWebGPUDevicesAsync() {
    if (!(navigator as any).gpu) {
        return;
    }
    const adapter = await (navigator as any).gpu.requestAdapter();
    console.log("adapter", adapter);
    const device = await adapter.requestDevice();
    console.log("device", device);
    const id = "webgpu";
    const dev = new DeviceWebGPU(id, adapter, device);
    devices[id] = dev;
    webgpuDevice = dev;
}

export function getDevice(device: Deviceish | null): Device {
    if (device === null) {
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

export function findDeviceWithType(type: DeviceType): Device | null {
    for (const id in devices) {
        if (devices[id].type === type) {
            return devices[id];
        }
    }
    return null;
}
