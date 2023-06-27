import { Module } from "./nn_module";
import { onnx } from "./onnx";
import { fetch } from "cross-fetch";

export class ONNXModule extends Module {
    readonly inputs: onnx.IValueInfoProto[];
    readonly outputs: onnx.IValueInfoProto[];
    readonly nodes: onnx.INodeProto[];
    constructor(inputs: onnx.IValueInfoProto[], outputs: onnx.IValueInfoProto[], nodes: onnx.INodeProto[]) {
        super();
        this.inputs = inputs;
        this.outputs = outputs;
        this.nodes = nodes;
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
        const graph = model.graph;
        if (!graph) {
            throw new Error("Model does not contain a graph");
        }
        const inputs = graph.input;
        if (!inputs) {
            throw new Error("Model graph does not contain inputs");
        }
        const outputs = graph.output;
        if (!outputs) {
            throw new Error("Model graph does not contain outputs");
        }
        const nodes = graph.node;
        if (!nodes) {
            throw new Error("Model graph does not contain nodes");
        }
        // const initializer = graph.initializer;
        return new ONNXModule(inputs, outputs, nodes);
    }
}
