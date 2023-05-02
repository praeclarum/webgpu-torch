import { ATypedArray, Dtype } from "./dtype";
import { Shape, defaultStrides, shapeSize } from "./shape";
import { IDevice } from "./device_if";
import { Deviceish } from "./device";

export type TensorArrayData = Array<number | TensorArrayData>;

export interface ITensor {
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
    abstract get dtype(): Dtype;
    abstract get shape(): Shape;
    abstract get device(): IDevice;
    abstract get(...indices: number[]): number | ITensor;
    abstract add_(other: ITensor): ITensor;
    abstract expand(shape: Shape): ITensor;
    abstract mm(other: ITensor): ITensor;
    abstract sum(axis: number | null): ITensor;
    abstract t(): ITensor;
}

export type TensorJsonData = {
    data: TensorArrayData | TensorImpl;
    dtype: Dtype;
    requiresGrad?: boolean;
    device?: Deviceish;
};

