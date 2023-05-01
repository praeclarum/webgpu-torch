import { Device  } from "./device";
import { Dtype } from "./dtype";
import { Shape } from "./shape";
import { TensorArrayData, TensorImpl } from "./tensor_if";
import { TensorCPU } from "./tensor_cpu";

export class DeviceCPU extends Device {
    constructor() {
        super("cpu", "cpu");
    }
    ones(shape: Shape, dtype: Dtype): TensorImpl {
        throw new Error("Method not implemented.");
    }
    tensor(data: TensorArrayData, dtype: Dtype): TensorImpl {
        return new TensorCPU(data, dtype);
    }
    zeros(shape: Shape, dtype: Dtype): TensorImpl {
        throw new Error("Method not implemented.");
    }
}
