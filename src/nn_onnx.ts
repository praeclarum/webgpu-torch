import { Module } from "./nn_module";
import { onnx } from "./onnx";
import { fetch } from "cross-fetch";

export class ONNXModule extends Module {
    constructor(model: onnx.ModelProto) {
        super();
    }
    static async loadUrlAsync(url: string): Promise<ONNXModule> {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const buffer = await response.arrayBuffer();
        const model = onnx.ModelProto.decode(new Uint8Array(buffer));
        return this.loadModel(model);
    }
    static loadModel(model: onnx.ModelProto): ONNXModule {
        return new ONNXModule(model);
    }
}
