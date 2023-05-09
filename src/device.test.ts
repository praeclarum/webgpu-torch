import { discoverWebGPUDevicesAsync, getDevice } from "./devices";

test("getDevice with cpu", () => {
    const d = getDevice("cpu");
    expect(d.id).toBe("cpu");
    expect(d.type).toBe("cpu");
});

test("fail to get bad device id/type", () => {
    expect(() => getDevice("bad")).toThrow();
});

test("discover WebGPU devices", async () => {
    if (await discoverWebGPUDevicesAsync()) {
        const d = getDevice("webgpu");
        expect(d.id).toBe("webgpu");
        expect(d.type).toBe("webgpu");
    }
});
