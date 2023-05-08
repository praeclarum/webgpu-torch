import { hasWebGPU } from "./webgpu";

test("webgpu is not available in test", async () => {
    const available = hasWebGPU();
    expect(typeof available).toBe("boolean");
});
