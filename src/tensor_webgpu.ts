import { Device } from "./device";
import { Dtype } from "./dtype";
import { Shape, Strides, defaultStrides, shapeGetAxis, shapeSize } from "./shape";
import { ITensor, TensorArrayData, TensorImpl } from "./tensor_if";

type ArrayType = Float32Array | Int32Array | Uint8Array;

export class TensorWebGPU extends TensorImpl {
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
}
