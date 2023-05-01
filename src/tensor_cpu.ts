import { Device } from "./device";
import { Dtype } from "./dtype";
import { Shape, Strides } from "./shape";
import { ITensor, TensorArrayData, TensorImpl } from "./tensor_if";

type ArrayType = Float32Array | Int32Array | Uint8Array;

export class TensorCPU extends TensorImpl {
    private _data: ArrayType;
    private _dtype: Dtype;
    private _shape: number[];
    private _strides: number[];
    private _device: Device;

    get data(): ArrayType {
        return this._data;
    }
    get dtype(): Dtype {
        return this._dtype;
    }
    get shape(): number[] {
        return this._shape;
    }
    get strides(): number[] {
        return this._strides;
    }
    get device(): Device {
        return this._device;
    }

    get(...indices: number[]): number | ITensor {
        let offset = 0;
        for (let i = 0; i < indices.length; i++) {
            offset += indices[i] * this._strides[i];
        }
        if (this._strides.length === indices.length) {
            return this._data[offset];
        }
        const data = this._data.subarray(offset, offset + this._strides[indices.length]);
        const shape = this._shape.slice(indices.length);
        const strides = this._strides.slice(indices.length);
        return new TensorCPU(data, shape, strides, this._device);
    }

    constructor(data: ArrayType, shape: Shape, strides: Strides, device: Device) {
        super();
        this._data = data;
        this._dtype = "float32";
        if (data instanceof Int32Array) {
            this._dtype = "int32";
        } else if (data instanceof Uint8Array) {
            this._dtype = "boolean";
        }
        this._shape = shape;
        this._strides = strides;
        this._device = device;
    }
    add_(other: ITensor): ITensor {
        throw new Error("Method not implemented.");
    }
    mm(other: ITensor): ITensor {
        throw new Error("Method not implemented.");
    }
    sum(arg0: number): ITensor {
        throw new Error("Method not implemented.");
    }
    t(): ITensor {
        let newShape = this._shape.slice();
        newShape.reverse();
        let newStrides = this._strides.slice();
        newStrides.reverse();
        let newData = this._data.slice();
        return new TensorCPU(newData, newShape, newStrides, this._device);
    }
}
