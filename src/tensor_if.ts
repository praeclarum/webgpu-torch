import { ATypedArray, Dtype } from "./dtype";
import { Shape, Strides, defaultStrides, shapeSize } from "./shape";
import { IDevice } from "./device_if";
import { Deviceish } from "./device";
import { UntypedStorage } from "./storage";
import { TensorImpl } from "./tensor_impl";

export type TensorArrayData = Array<number | TensorArrayData>;

export type TensorJsonData = {
    data: TensorArrayData | TensorImpl;
    dtype: Dtype;
    requiresGrad?: boolean;
    device?: Deviceish;
};

export interface ITensor {
    get storage(): UntypedStorage;
    get dtype(): Dtype;
    get shape(): Shape;
    get device(): IDevice;
    get(...indices: number[]): number | ITensor;
    add_(other: ITensor, alpha?: number): ITensor;
    expand(shape: Shape): ITensor;
    mm(other: ITensor): ITensor;
    sum(axis: number | null): ITensor;
    t(): ITensor;
}

