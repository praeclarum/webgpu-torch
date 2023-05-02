import { Device } from "./device";
import { ATypedArray, Dtype, getDtype } from "./dtype";
import { Shape, Strides, defaultStrides, shapeGetAxis, shapeSize } from "./shape";
import { ITensor, TensorArrayData, TensorImpl } from "./tensor_if";

export class TensorWebGPU extends TensorImpl {
    private _data: ATypedArray;
    private _dtype: Dtype;
    private _shape: number[];
    private _strides: number[];
    private _device: Device;

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

    constructor(data: ATypedArray, shape: Shape, strides: Strides, device: Device) {
        super();
        this._data = data;
        this._dtype = getDtype(data);
        this._shape = shape;
        this._strides = strides;
        this._device = device;
    }

    get(...indices: number[]): number | ITensor {
        throw new Error("Method not implemented.");
    }
    add_(other: ITensor): ITensor {
        throw new Error("Method not implemented.");
    }
    expand(shape: Shape): ITensor {
        throw new Error("Method not implemented.");
    }
    mm(other: ITensor): ITensor {
        throw new Error("Method not implemented.");
    }
    sum(axis: number | null): ITensor {
        throw new Error("Method not implemented.");
    }
    t(): ITensor {
        throw new Error("Method not implemented.");
    }
}
