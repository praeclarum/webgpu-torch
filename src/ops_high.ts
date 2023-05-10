// High level ops

import { IAutoFunction, shouldCreateGradient } from "./autograd";
import type { Tensor } from "./tensor";

export function unary(func: IAutoFunction, input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return func.apply(input);
    }
    return func.forward([input]);
}

export function unaryWithAlpha(func: IAutoFunction, input: Tensor, alpha?: number): Tensor {
    if (shouldCreateGradient(input)) {
        return func.apply(input, alpha);
    }
    return func.forward([input, alpha]);
}

export function binary(func: IAutoFunction, input: Tensor, other: Tensor): Tensor {
    if (input.shape.length !== other.shape.length) {
        throw new Error(`Shape dimensions must match. Got ${input.shape} and ${other.shape}`);
    }
    if (shouldCreateGradient(input)) {
        return func.apply(input, other);
    }
    return func.forward([input, other]);
}

export function binaryWithAlpha(func: IAutoFunction, input: Tensor, other: Tensor, alpha?: number): Tensor {
    if (input.shape.length !== other.shape.length) {
        throw new Error(`Shape dimensions must match. Got ${input.shape} and ${other.shape}`);
    }
    if (shouldCreateGradient(input)) {
        return func.apply(input, other, alpha);
    }
    return func.forward([input, other, alpha]);
}

export function reduction(func: IAutoFunction, input: Tensor, axis?: number, keepDim?: boolean): Tensor {
    if (shouldCreateGradient(input)) {
        return func.apply(input, axis, keepDim);
    }
    return func.forward([input, axis, keepDim]);
}
