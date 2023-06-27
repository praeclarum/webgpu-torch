import { Module } from "./nn_module";
import { onnx } from "./onnx";
import { fetch } from "cross-fetch";
import { Tensor, TensorSpec } from "./tensor";
import { matmul } from "./ops_artisanal";
import { Shape, defaultStrides } from "./shape";
import Long from "@xtuc/long";
import { Dtype } from "./dtype";

export class ONNXModule extends Module {
    readonly inputs: onnx.IValueInfoProto[];
    readonly outputs: onnx.IValueInfoProto[];
    readonly nodes: onnx.INodeProto[];
    readonly nodeFromOutput: { [outputName: string]: onnx.INodeProto } = {};
    readonly tensorSpecNameToBufferName: { [tensorSpecName: string]: string } = {};
    constructor(model: onnx.IModelProto) {
        super();
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
        this.inputs = inputs;
        this.outputs = outputs;
        this.nodes = nodes;
        for (const node of nodes) {
            for (const output of node.output || []) {
                this.nodeFromOutput[output] = node;
            }
        }
        for (const init of graph.initializer || []) {
            const name = init.name || "";
            if (name.length === 0) {
                throw new Error("Initializer does not have a name");
            }
            const bufferName = name.replace(/[^a-zA-Z0-9_]/g, "_");
            this.tensorSpecNameToBufferName[name] = bufferName;
            this.registerBuffer(name, initToTensor(init));
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
        return new ONNXModule(model);
    }
}

function longDimsToShape(dims: (Long|number)[]): Shape {
    const shape: number[] = [];
    for (const rawdim of dims) {
        if (rawdim instanceof Long) {
            const dim = rawdim.toNumber();
            if (dim < 0) {
                throw new Error("ONNX tensor dimension is negative");
            }
            shape.push(dim);
        } else {
            const dim = rawdim;
            if (dim < 0) {
                throw new Error("ONNX tensor dimension is negative");
            }
            shape.push(dim);
        }
    }
    return shape;
}

function onnxDataTypeToDType(onnxDataType: onnx.TensorProto.DataType): Dtype {
    switch (onnxDataType) {
        case onnx.TensorProto.DataType.FLOAT:
            return "float32";
        default:
            throw new Error(`ONNX data type ${onnxDataType} not supported`);
    }
}

function initToTensor(init: onnx.ITensorProto): Tensor {
    const shape = longDimsToShape(init.dims || []);
    const indata64 = init.int64Data || init.uint64Data;
    let data64: number[] | undefined;
    if (indata64 && indata64.length > 0) {
        data64 = indata64.map((x) => (x instanceof Long) ? x.toNumber() : x as number);
    }
    let dataString: Uint8Array | undefined;
    if (init.stringData && init.stringData.length > 0) {
        dataString = init.stringData[0];
        throw new Error("String data not supported");
    }
    const data = init.rawData || init.floatData || init.int32Data || data64 || dataString;
    if (!data) {
        throw new Error("Initializer does not contain data");
    }
    const spec: TensorSpec = {
        data,
        shape,
        strides: defaultStrides(shape),
        dtype: onnxDataTypeToDType(init.dataType || 1),
    };
    return new Tensor(spec);
}
