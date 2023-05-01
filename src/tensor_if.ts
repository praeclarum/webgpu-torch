import { Dtype } from "./dtype";

export type TensorArrayData = Array<number | TensorArrayData>;

export interface ITensor {
    get dtype(): Dtype;
    add_(other: ITensor): ITensor;
    mm(other: ITensor): ITensor;
    sum(arg0: number): ITensor;
    t(): ITensor;
}

export abstract class TensorImpl implements ITensor {
    abstract get dtype(): Dtype;
    abstract add_(other: ITensor): ITensor;
    abstract mm(other: ITensor): ITensor;
    abstract sum(arg0: number): ITensor;
    abstract t(): ITensor;
}
