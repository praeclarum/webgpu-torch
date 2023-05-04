import { Dtype } from "./dtype";
import { Shape } from "./shape";
import { TensorArrayData } from "./tensor_if";
import { TensorImpl } from "./tensor_impl";

export interface IDevice {
    ones(shape: Shape, dtype: Dtype): TensorImpl;
    tensor(data: TensorArrayData, dtype: Dtype): TensorImpl;
    zeros(shape: Shape, dtype: Dtype): TensorImpl;
}
