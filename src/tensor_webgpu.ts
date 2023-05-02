import { Device } from "./device";
import { Dtype } from "./dtype";
import { Shape, Strides } from "./shape";
import { GPUBufferStorage } from "./storage";
import { ITensor, TensorImpl } from "./tensor_if";

export class TensorWebGPU extends TensorImpl {
    private _storage: GPUBufferStorage;
    private _dtype: Dtype;
    private _shape: number[];
    private _strides: number[];
    private _device: Device;

    get storage(): GPUBufferStorage {
        return this._storage;
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

    constructor(
        storage: GPUBufferStorage,
        dtype: Dtype,
        shape: Shape,
        strides: Strides,
        device: Device
    ) {
        super();
        this._storage = storage;
        this._dtype = dtype;
        this._shape = shape;
        this._strides = strides;
        this._device = device;
    }

    get(...indices: number[]): number | TensorImpl {
        throw new Error("Method not implemented.");
    }
    add_(other: ITensor): TensorImpl {
        throw new Error("Method not implemented.");
    }
    expand(shape: Shape): TensorImpl {
        throw new Error("Method not implemented.");
    }
    mm(other: ITensor): TensorImpl {
        throw new Error("Method not implemented.");
    }
    sum(axis: number | null): TensorImpl {
        throw new Error("Method not implemented.");
    }
    t(): TensorImpl {
        throw new Error("Method not implemented.");
    }
}
