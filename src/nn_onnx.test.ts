import { ones } from "./factories";
import { ONNXModule } from "./nn_onnx";
import { Tensor } from "./tensor";

test("load from url", async () => {
    const model = await ONNXModule.loadUrlAsync("https://praeclarum.org/webgpu-torch/tests/models/matmul.onnx");
    expect(model).toBeInstanceOf(ONNXModule);
    const a = ones([2, 3]);
    const b = ones([3, 2]);
    const c = model.forward([a, b])[0];
    expect(c).toBeInstanceOf(Tensor);
    expect(c.shape).toEqual([2, 2]);
    expect(await c.toArrayAsync()).toEqual([[3, 3], [3, 3]]);
});

// DEV TESTS

// test("load t5 decoder", async () => {
//     const model = await ONNXModule.loadUrlAsync("http://localhost:8999/t5-small-decoder-quantized.onnx");
//     expect(model).toBeInstanceOf(ONNXModule);
// });
