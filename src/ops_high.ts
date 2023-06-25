// High level ops

import { IAutoFunction, shouldCreateGradient } from "./autograd";
import { broadcastShapes, shapeSize, stridedShapeIsContiguous } from "./shape";
import type { Tensor } from "./tensor";

export function unary(func: IAutoFunction, input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return func.apply(input);
    }
    return func.forward([input]);
}

export function unaryWithAlpha(
    func: IAutoFunction,
    input: Tensor,
    alpha?: number
): Tensor {
    if (shouldCreateGradient(input)) {
        return func.apply(input, alpha);
    }
    return func.forward([input, alpha]);
}

export function binary(
    func: IAutoFunction,
    input: Tensor,
    other: number | Tensor
): Tensor {
    if (typeof other === "number") {
        if (shouldCreateGradient(input)) {
            return func.apply(input, other);
        }
    } else {
        if (input.shape.length !== other.shape.length) {
            throw new Error(
                `Shape dimensions must match. Got ${input.shape} and ${other.shape}`
            );
        }
        if (shouldCreateGradient(input) || shouldCreateGradient(other)) {
            return func.apply(input, other);
        }
    }
    return func.forward([input, other]);
}

export function binaryWithAlpha(
    func: IAutoFunction,
    input: Tensor,
    other: number | Tensor,
    alpha?: number
): Tensor {
    if (typeof other === "number") {
        if (shouldCreateGradient(input)) {
            return func.apply(input, other, alpha);
        }
    } else {
        if (shouldCreateGradient(input, other)) {
            return func.apply(input, other, alpha);
        }
    }
    return func.forward([input, other, alpha]);
}

export function reduction(
    func: IAutoFunction,
    input: Tensor,
    dim?: number | number[],
    keepdim?: boolean
): Tensor {
    if (shouldCreateGradient(input)) {
        return func.apply(input, dim, keepdim);
    }
    return func.forward([input, dim, keepdim]);
}
