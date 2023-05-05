import { Dtype } from "./dtype";
import { Shape } from "./shape";
import { IDevice } from "./device_if";
import { Deviceish } from "./device";
import { UntypedStorage } from "./storage";
import { TensorImpl } from "./tensor_impl";
import { KernelConfigInput, KernelParamsInput } from "./kernel";

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
    runKernel(name: string, config: KernelConfigInput, params: KernelParamsInput, outputShapes: Shape[], ...additionalInputs: ITensor[]): ITensor[];
    get(...indices: number[]): number | ITensor;
    add_(other: ITensor, alpha?: number): ITensor;
    expand(shape: Shape): ITensor;
    mm(other: ITensor): ITensor;
    sum(axis: number | null): ITensor;
    t(): ITensor;
    // Codegen marker
    abs(): ITensor;
    abs_(): ITensor;
    absGrad(): ITensor;
    acos(): ITensor;
    acos_(): ITensor;
    acosGrad(): ITensor;
    acosh(): ITensor;
    acosh_(): ITensor;
    acoshGrad(): ITensor;
    add(other: ITensor, alpha?: number): ITensor;
    add_(other: ITensor, alpha?: number): ITensor;
    asin(): ITensor;
    asin_(): ITensor;
    asinGrad(): ITensor;
    asinh(): ITensor;
    asinh_(): ITensor;
    asinhGrad(): ITensor;
    atan(): ITensor;
    atan_(): ITensor;
    atanGrad(): ITensor;
    atan2(other: ITensor): ITensor;
    atan2_(other: ITensor): ITensor;
    mul(other: ITensor, alpha?: number): ITensor;
    mul_(other: ITensor, alpha?: number): ITensor;
    sub(other: ITensor, alpha?: number): ITensor;
    sub_(other: ITensor, alpha?: number): ITensor;
    // End codegen marker
}
