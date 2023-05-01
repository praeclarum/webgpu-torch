import { Shape, Shapeish, shapeishToShape } from "./shape";
import { Tensor } from "./tensor";
import { UntypedStorage } from "./storage";
import { getDevice, Deviceish } from "./devices";
import { Dtype, dtypeishToDtype } from "./dtype";

export function ones(
    shape: Shapeish,
    dtype: Dtype | null = null,
    device: Deviceish | null = null
): Tensor {
    return new Tensor(getDevice(device).ones(shapeishToShape(shape), dtypeishToDtype(dtype)));
}

export function zeros(
    shape: Shapeish,
    dtype: Dtype | null = null,
    device: Deviceish | null = null
): Tensor {
    return new Tensor(getDevice(device).zeros(shapeishToShape(shape), dtypeishToDtype(dtype)));
}
