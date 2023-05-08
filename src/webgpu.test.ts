import { hasWebGPU } from "./webgpu";

test("hasWebGPU returns a result", async () => {
    const available = hasWebGPU();
    expect(typeof available).toBe("boolean");
});
