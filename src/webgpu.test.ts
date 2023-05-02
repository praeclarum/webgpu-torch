import { hasWebGPU } from "./index";

test("webgpu is not available in test", async () => {
    const available = hasWebGPU();
    expect(available).toBe(false);
});
