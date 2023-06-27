import { Module } from "./nn_module";
import { onnx } from "./onnx";
import { fetch } from "cross-fetch";
import type { Tensor } from "./tensor";
import { matmul } from "./ops_artisanal";

export class ONNXModule extends Module {
    readonly inputs: onnx.IValueInfoProto[];
    readonly outputs: onnx.IValueInfoProto[];
    readonly nodes: onnx.INodeProto[];
    readonly nodeFromOutput: { [outputName: string]: onnx.INodeProto } = {};
    constructor(
        inputs: onnx.IValueInfoProto[],
        outputs: onnx.IValueInfoProto[],
        nodes: onnx.INodeProto[]
    ) {
        super();
        this.inputs = inputs;
        this.outputs = outputs;
        this.nodes = nodes;
        for (const node of nodes) {
            for (const output of node.output || []) {
                this.nodeFromOutput[output] = node;
            }
        }
    }
    forward(inputs: Tensor[]): Tensor[] {
        const outputs: Tensor[] = [];
        const env: { [name: string]: Tensor } = {};
        for (const [i, input] of inputs.entries()) {
            const inputName = this.inputs[i].name || "";
            if (inputName.length === 0) {
                throw new Error(`Input ${i} does not have a name`);
            }
            env[inputName] = input;
        }
        for (const [i, output] of this.outputs.entries()) {
            const outputName = output.name || "";
            if (outputName.length === 0) {
                throw new Error(`Output ${i} does not have a name`);
            }
            const node = this.nodeFromOutput[outputName];
            if (!node) {
                throw new Error(`Output ${i} does not have a node`);
            }
            this.evaluateNode(node, inputs, env);
            outputs.push(env[outputName]);
        }
        return outputs;
    }
    private evaluateNode(
        node: onnx.INodeProto,
        inputs: Tensor[],
        env: { [name: string]: Tensor }
    ): Tensor[] {
        switch (node.opType) {
            case "MatMul": {
                const c = matmul(inputs[0], inputs[1]);
                env[node.output![0]] = c;
                return [c];
            }
            default:
                throw new Error(`Cannot evaluate ONNX node type '${node.opType}'`);
        }
    }
    static async loadUrlAsync(url: string): Promise<ONNXModule> {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP request for model failed. Status: ${response.status}`);
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
