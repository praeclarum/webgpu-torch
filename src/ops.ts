import { SumFunction, SumAxisFunction } from "./autograd";
import { Device, DeviceType, Deviceish } from "./device";
import { Dtype } from "./dtype";
import { Tensor } from "./tensor";
import { TensorArrayData, TensorJsonData } from "./tensor_if";
import { TensorImpl } from "./tensor_impl";

function shouldCreateGradient(...inputs: Tensor[]): boolean {
    for (const input of inputs) {
        if (input.requiresGrad) {
            return true;
        }
    }
    return false;
}

export function add_(input: Tensor, other: Tensor, alpha?: number): Tensor {
    if (input.shape.length !== other.shape.length) {
        throw new Error(
            `Shape dimensions must match. Got ${input.shape} and ${other.shape}`
        );
    }
    for (let i = 0; i < input.shape.length; i++) {
        if (input.shape[i] !== other.shape[i]) {
            throw new Error(
                `Shapes must match at index ${i}. Got ${input.shape} and ${other.shape}`
            );
        }
    }
    return new Tensor(input.impl.add_(other.impl, alpha));
}

export function mm(a: Tensor, b: Tensor): Tensor {
    if (shouldCreateGradient(a, b)) {
        throw new Error("mm gradient not supported yet");
    } else {
        if (a.shape.length !== 2 || b.shape.length !== 2) {
            throw new Error(
                `Expected 2D tensors, got ${a.shape} and ${b.shape}`
            );
        }
        if (a.shape[1] !== b.shape[0]) {
            throw new Error(
                `Expected tensors inner dimensions to be compatible, got ${a.shape} and ${b.shape}`
            );
        }
        return new Tensor(a.impl.mm(b.impl));
    }
}

export function sum(input: Tensor, axis: number | null = 0): Tensor {
    if (shouldCreateGradient(input)) {
        if (axis === null) {
            return SumFunction.apply(input);
        }
        return SumAxisFunction.apply(input, axis);
    } else {
        return new Tensor(input.impl.sum(axis));
    }
}

export function t(input: Tensor): Tensor {
    if (input.shape.length !== 2) {
        throw new Error(`Expected 2D tensor, got ${input.shape}`);
    }
    if (shouldCreateGradient(input)) {
        throw new Error("t gradient not supported yet");
        // return TransposeFunction.apply(input, 0, 1);
    } else {
        return new Tensor(input.impl.t());
    }
}

export function tensor(
    data: TensorArrayData | TensorImpl | TensorJsonData,
    dtype: Dtype = "float32",
    device: Deviceish | null = null,
    requiresGrad: boolean = false
) {
    return new Tensor(data, dtype, device, requiresGrad);
}

// Codegen marker
// End codegen marker
