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

// test("run t5 encoder", async () => {
//     const model = await ONNXModule.loadUrlAsync("http://localhost:8999/t5-small-encoder-quantized.onnx");
//     expect(model).toBeInstanceOf(ONNXModule);
//     const batchSize = 3;
//     const seqLength = 5;
//     const inputIds = ones([batchSize, seqLength], "int32");
//     const attentionMask = ones([batchSize, seqLength], "int32");
//     const encoderOutput = model.forward([inputIds, attentionMask])[0];
//     expect(encoderOutput).toBeInstanceOf(Tensor);
//     expect(encoderOutput.shape).toEqual([batchSize, seqLength, 512]);
// });

// test("load t5 decoder", async () => {
//     const model = await ONNXModule.loadUrlAsync("http://localhost:8999/t5-small-decoder-quantized.onnx");
//     expect(model).toBeInstanceOf(ONNXModule);
// });
