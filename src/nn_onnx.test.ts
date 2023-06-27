import { ONNXModule } from "./nn_onnx";

test("load from url", async () => {
    const model = await ONNXModule.loadUrlAsync("https://github.com/microsoft/onnxruntime-inference-examples/raw/main/js/api-usage_inference-session/model.onnx");
    expect(model).toBeInstanceOf(ONNXModule);
});
