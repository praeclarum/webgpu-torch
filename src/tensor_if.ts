import { Dtype } from "./dtype";
import { Shape } from "./shape";
import { IDevice } from "./device_if";

export type TensorArrayData = Array<number | TensorArrayData>;

export interface ITensor {
    get dtype(): Dtype;
    get shape(): Shape;
    get device(): IDevice;
    get(...indices: number[]): number | ITensor;
    add_(other: ITensor): ITensor;
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
    abstract mm(other: ITensor): ITensor;
    abstract sum(axis: number | null): ITensor;
    abstract t(): ITensor;
}
