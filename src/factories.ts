import { Shapeish, getShape } from "./shape";
import { Deviceish } from "./device";
import { getDevice } from "./devices";
import { Dtypeish, getDtype } from "./dtype";
import { Tensor } from "./tensor";

export function ones(
    shape: Shapeish,
    dtype: Dtypeish | null = null,
    device: Deviceish | null = null
): Tensor {
    return new Tensor(getDevice(device).ones(getShape(shape), getDtype(dtype)));
}

export function zeros(
    shape: Shapeish,
    dtype: Dtypeish | null = null,
    device: Deviceish | null = null
): Tensor {
    return new Tensor(
        getDevice(device).zeros(getShape(shape), getDtype(dtype))
    );
}
