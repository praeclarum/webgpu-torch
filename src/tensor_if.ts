import { ATypedArray, Dtype } from "./dtype";
import { Shape, defaultStrides, shapeSize } from "./shape";
import { IDevice } from "./device_if";
import { Deviceish } from "./device";
import { UntypedStorage } from "./storage";

export type TensorArrayData = Array<number | TensorArrayData>;

export interface ITensor {
    get storage(): UntypedStorage;
    get dtype(): Dtype;
    get shape(): Shape;
    get device(): IDevice;
    get(...indices: number[]): number | ITensor;
    add_(other: ITensor): ITensor;
    expand(shape: Shape): ITensor;
    mm(other: ITensor): ITensor;
    sum(axis: number | null): ITensor;
    t(): ITensor;
}

export abstract class TensorImpl implements ITensor {
    abstract get storage(): UntypedStorage;
    getTypedArray(): ATypedArray { return this.storage.getTypedArray(this.dtype); }
    abstract get dtype(): Dtype;
    abstract get shape(): Shape;
    abstract get device(): IDevice;
    abstract get(...indices: number[]): number | TensorImpl;
    abstract add_(other: ITensor): TensorImpl;
    abstract expand(shape: Shape): TensorImpl;
    abstract mm(other: ITensor): TensorImpl;
    abstract sum(axis: number | null): TensorImpl;
    abstract t(): TensorImpl;
}

export type TensorJsonData = {
    data: TensorArrayData | TensorImpl;
    dtype: Dtype;
    requiresGrad?: boolean;
    device?: Deviceish;
};

