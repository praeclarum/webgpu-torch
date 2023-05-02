import { SumFunction, SumAxisFunction } from './autograd';
import { Tensor } from './tensor';

function shouldCreateGradient(...inputs: Tensor[]): boolean {
    for (const input of inputs) {
        if (input.requiresGrad) {
            return true;
        }
    }
    return false;
}

export function add_(a: Tensor, b: Tensor): Tensor {
    return new Tensor(a.impl.add_(b.impl));
}

export function mm(a: Tensor, b: Tensor): Tensor {
    if (shouldCreateGradient(a, b)) {
        throw new Error("mm gradient not supported yet");
    }
    else {
        if (a.shape.length !== 2 || b.shape.length !== 2) {
            throw new Error(`Expected 2D tensors, got ${a.shape} and ${b.shape}`);
        }
        if (a.shape[1] !== b.shape[0]) {
            throw new Error(`Expected tensors inner dimensions to be compatible, got ${a.shape} and ${b.shape}`);
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
    }
    else {
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
    }
    else {
        return new Tensor(input.impl.t());
    }
}
