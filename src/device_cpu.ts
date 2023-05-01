import { Device  } from "./device";
import { Dtype } from "./dtype";
import { Shape, defaultStrides, shapeSize } from "./shape";
import { TensorArrayData, TensorImpl } from "./tensor_if";
import { TensorCPU } from "./tensor_cpu";

export class DeviceCPU extends Device {
    constructor() {
        super("cpu", "cpu");
    }
    ones(shape: Shape, dtype: Dtype): TensorImpl {
        const data = arrayForDtype(shapeSize(shape), dtype);
        data.fill(1);
        return new TensorCPU(data, shape, defaultStrides(shape), this);
    }
    tensor(data: TensorArrayData, dtype: Dtype): TensorImpl {
        const shape: number[] = [];
        function getShape(data: TensorArrayData | number) {
            if (typeof data === "number") {
                return;
            }
            shape.push(data.length);
            getShape(data[0]);
        }
        getShape(data);
        const strides = defaultStrides(shape);
        const size = shape.reduce((a, b) => a * b);
        if (dtype !== "float32") {
            throw new Error("Only float32 is supported");
        }
        const flatData = new Float32Array(size);
        let flatIndex = 0;
        function flatten(data: TensorArrayData | number) {
            if (typeof data === "number") {
                flatData[flatIndex] = data;
                flatIndex++;
                return;
            }
            for (let i = 0; i < data.length; i++) {
                let d = data[i];
                if (typeof d === "number") {
                    for (let j = 0; j < data.length; j++) {
                        flatData[flatIndex] = data[j] as number;
                        flatIndex++;
                    }
                    return;
                }
                flatten(data[i]);
            }
        }
        flatten(data);
        return new TensorCPU(flatData, shape, strides, this);
    }
    zeros(shape: Shape, dtype: Dtype): TensorImpl {
        const data = arrayForDtype(shapeSize(shape), dtype);
        return new TensorCPU(data, shape, defaultStrides(shape), this);
    }
}

function arrayForDtype(length: number, dtype: Dtype) {
    switch (dtype) {
        case "float32":
            return new Float32Array(length);
        case "int32":
            return new Int32Array(length);
        case "boolean":
            return new Uint8Array(length);
        default:
            throw new Error(`Unsupported dtype: ${dtype}`);
    }
}

