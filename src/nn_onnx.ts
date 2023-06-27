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
    readonly nodeFromOutput: { [outputName: string]: { node:onnx.INodeProto, outputIndex:number } } = {};
    readonly tensorSpecNameToBufferName: { [tensorSpecName: string]: string } =
        {};
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
            for (const [outputIndex, output] of (node.output || []).entries()) {
                this.nodeFromOutput[output] = {node, outputIndex};
            }
        }
        for (const init of graph.initializer || []) {
            const name = init.name || "";
            if (name.length === 0) {
                throw new Error("Initializer does not have a name");
            }
            const bufferName = name.replace(/[^a-zA-Z0-9_]/g, "_");
            this.tensorSpecNameToBufferName[name] = bufferName;
            this.registerBuffer(bufferName, initToTensor(init));
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
            const outputT = this.evaluateTensor(outputName, env);
            outputs.push(outputT);
        }
        return outputs;
    }
    private evaluateTensor(
        tensorSpecName: string,
        env: { [name: string]: Tensor }
    ): Tensor {
        // Has it already been evaluated?
        if (tensorSpecName in env) {
            return env[tensorSpecName];
        }
        // Has it been initialized?
        if (tensorSpecName in this.tensorSpecNameToBufferName) {
            const bufferName = this.tensorSpecNameToBufferName[tensorSpecName];
            const buffer = this.getBuffer(bufferName);
            if (!buffer) {
                throw new Error(`Buffer ${bufferName} not found`);
            }
            env[tensorSpecName] = buffer;
            return buffer;
        }
        // Is it a node?
        const node = this.nodeFromOutput[tensorSpecName];
        if (!node) {
            throw new Error(`Tensor ${tensorSpecName} does not have a node`);
        }
        // Evaluate inputs
        const inputs: Tensor[] = [];
        for (const tensorSpecName of node.node.input || []) {
            inputs.push(this.evaluateTensor(tensorSpecName, env));
        }
        // Evaluate node
        const nodeOutputNames = node.node.output || [];
        const outputs = evalNode(node.node, inputs);
        if (outputs.length !== nodeOutputNames.length) {
            throw new Error(
                `Node ${node.node.name} has ${nodeOutputNames.length} outputs but evaluated to ${outputs.length} outputs`
            );
        }
        for (const [i, output] of outputs.entries()) {
            const outputName = nodeOutputNames[i];
            env[outputName] = output;
        }
        const output = outputs[node.outputIndex];
        env[tensorSpecName] = output;
        return output;
    }
    static async loadUrlAsync(url: string): Promise<ONNXModule> {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(
                `HTTP request for model failed. Status: ${response.status}`
            );
        }
        const buffer = await response.arrayBuffer();
        const model = onnx.ModelProto.decode(new Uint8Array(buffer));
        return new ONNXModule(model);
    }
}

function longDimsToShape(dims: (Long | number)[]): Shape {
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
        case onnx.TensorProto.DataType.INT8:
            return "int8";
        case onnx.TensorProto.DataType.FLOAT:
            return "float32";
        case onnx.TensorProto.DataType.INT64:
            return "int64";
        default:
            throw new Error(`ONNX data type ${onnxDataType} not supported`);
    }
}

function initToTensor(init: onnx.ITensorProto): Tensor {
    const shape: Shape = longDimsToShape(init.dims || []);
    const strides = defaultStrides(shape);
    let dtype = onnxDataTypeToDType(init.dataType || 1);
    let data: number[];
    if (dtype === "int64") {
        const oldDtype = dtype;
        if (init.rawData && init.rawData.length > 0) {
            let bytes = init.rawData;
            let byteOffset = bytes.byteOffset;
            const length = init.rawData.byteLength / 8;
            if (byteOffset % 8 !== 0) {
                const newBuffer = new ArrayBuffer(bytes.byteLength);
                const newBytes = new Uint8Array(newBuffer);
                newBytes.set(bytes);
                bytes = newBytes;
                byteOffset = bytes.byteOffset;
            }
            const in64 = new BigInt64Array(bytes.buffer, byteOffset, length);
            const minInt32 = -2147483648n;
            const maxInt32 = 2147483647n;
            data = Array.from(in64).map((x) => {
                if (x < minInt32) {
                    console.warn(`Clipping int64 value ${x} to -2147483648`);
                    return -2147483648;
                }
                if (x > maxInt32) {
                    console.warn(`Clipping int64 value ${x} to 2147483647`);
                    return 2147483647;
                }
                return Number(x);
            });
            dtype = "int32";
        } else if (init.int64Data && init.int64Data.length > 0) {
            data = init.int64Data.map((x) =>
                x instanceof Long ? x.toNumber() : (x as number)
            );
            dtype = "int32";
        } else if (init.uint64Data && init.uint64Data.length > 0) {
            data = init.uint64Data.map((x) =>
                x instanceof Long ? x.toNumber() : (x as number)
            );
            dtype = "uint32";
        } else {
            throw new Error("Initializer for int64 data cannot be found");
        }
    } else {
        if (init.rawData && init.rawData.length > 0) {
            data = Array.from(init.rawData);
        } else if (init.floatData && init.floatData.length > 0) {
            data = init.floatData;
        } else if (init.int32Data && init.int32Data.length > 0) {
            data = init.int32Data;
        } else {
            throw new Error("Initializer does not contain data");
        }
    }
    const spec: TensorSpec = {
        data,
        shape,
        strides,
        dtype,
    };
    return new Tensor(spec);
}

function evalNode(node: onnx.INodeProto, inputs: Tensor[]): Tensor[] {
    switch (node.opType) {
        case "MatMul": {
            return [matmul(inputs[0], inputs[1])];
        }
        case "Mul": {
            return [inputs[0].mul(inputs[1])];
        }
        default:
            throw new Error(
                `Cannot evaluate ONNX node type '${node.opType}'`
            );
    }
}
