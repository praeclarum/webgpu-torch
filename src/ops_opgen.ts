import * as functions from "./functions";
import { Deviceish } from "./device";
import { Dtype } from "./dtype";
import { Tensor } from "./tensor";
import { TensorArrayData, TensorJsonData } from "./tensor_if";
import { TensorImpl } from "./tensor_impl";
import { shouldCreateGradient } from "./autograd";
export function abs(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        throw new Error("Gradient of abs is not supported");
    }
    return new Tensor(input.impl.abs());
}
export function acos(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        throw new Error("Gradient of acos is not supported");
    }
    return new Tensor(input.impl.acos());
}
export function acosh(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        throw new Error("Gradient of acosh is not supported");
    }
    return new Tensor(input.impl.acosh());
}
export function add(input: Tensor, other: Tensor, alpha?: number): Tensor {
    if (shouldCreateGradient(input)) {
        throw new Error("Gradient of add is not supported");
    }
    return new Tensor(input.impl.add(other.impl, alpha));
}
export function asin(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        throw new Error("Gradient of asin is not supported");
    }
    return new Tensor(input.impl.asin());
}
export function asinh(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        throw new Error("Gradient of asinh is not supported");
    }
    return new Tensor(input.impl.asinh());
}
export function atan(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        throw new Error("Gradient of atan is not supported");
    }
    return new Tensor(input.impl.atan());
}
export function atan2(input: Tensor, other: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        throw new Error("Gradient of atan2 is not supported");
    }
    return new Tensor(input.impl.atan2(other.impl));
}
export function sub(input: Tensor, other: Tensor, alpha?: number): Tensor {
    if (shouldCreateGradient(input)) {
        throw new Error("Gradient of sub is not supported");
    }
    return new Tensor(input.impl.sub(other.impl, alpha));
}
