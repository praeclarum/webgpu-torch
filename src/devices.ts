import { Device, DeviceType, Deviceish } from "./device";
import { DeviceCPU } from "./device_cpu";

const cpuDevice = new DeviceCPU();

const devices: { [id: string]: Device } = {
    cpu: cpuDevice,
};

export function getDevice(device: Deviceish | null): Device {
    if (device === null) {
        return cpuDevice;
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
