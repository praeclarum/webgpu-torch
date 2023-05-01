import { Dtype } from "./dtype";
import { Shape } from "./shape";
import { TensorArrayData, TensorImpl } from "./tensor_if";

export interface IDevice {
    ones(shape: Shape, dtype: Dtype): TensorImpl;
    tensor(data: TensorArrayData, dtype: Dtype): TensorImpl;
    zeros(shape: Shape, dtype: Dtype): TensorImpl;
}
