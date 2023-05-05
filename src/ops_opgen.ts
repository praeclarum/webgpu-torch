import * as functions from "./functions";
import { Tensor } from "./tensor";
import { shouldCreateGradient } from "./autograd";
export function abs(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.AbsFunction.apply(input);
    }
    return new Tensor(input.impl.abs());
}
export function absGrad(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.AbsGradFunction.apply(input);
    }
    return new Tensor(input.impl.absGrad());
}
export function acos(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.AcosFunction.apply(input);
    }
    return new Tensor(input.impl.acos());
}
export function acosGrad(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.AcosGradFunction.apply(input);
    }
    return new Tensor(input.impl.acosGrad());
}
export function acosh(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.AcoshFunction.apply(input);
    }
    return new Tensor(input.impl.acosh());
}
export function acoshGrad(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.AcoshGradFunction.apply(input);
    }
    return new Tensor(input.impl.acoshGrad());
}
export function add(input: Tensor, other: Tensor, alpha?: number): Tensor {
    if (input.shape.length !== other.shape.length) {
        throw new Error(`Shape dimensions of add must match. Got ${input.shape} and ${other.shape}`);
    }
    if (shouldCreateGradient(input, other)) {
        return functions.AddFunction.apply(input, other);
    }
    return new Tensor(input.impl.add(other.impl, alpha));
}
export function asin(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.AsinFunction.apply(input);
    }
    return new Tensor(input.impl.asin());
}
export function asinGrad(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.AsinGradFunction.apply(input);
    }
    return new Tensor(input.impl.asinGrad());
}
export function asinh(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.AsinhFunction.apply(input);
    }
    return new Tensor(input.impl.asinh());
}
export function asinhGrad(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.AsinhGradFunction.apply(input);
    }
    return new Tensor(input.impl.asinhGrad());
}
export function atan(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.AtanFunction.apply(input);
    }
    return new Tensor(input.impl.atan());
}
export function atanGrad(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.AtanGradFunction.apply(input);
    }
    return new Tensor(input.impl.atanGrad());
}
export function atan2(input: Tensor, other: Tensor): Tensor {
    if (input.shape.length !== other.shape.length) {
        throw new Error(`Shape dimensions of atan2 must match. Got ${input.shape} and ${other.shape}`);
    }
    if (shouldCreateGradient(input, other)) {
        return functions.Atan2Function.apply(input, other);
    }
    return new Tensor(input.impl.atan2(other.impl));
}
export function mul(input: Tensor, other: Tensor, alpha?: number): Tensor {
    if (input.shape.length !== other.shape.length) {
        throw new Error(`Shape dimensions of mul must match. Got ${input.shape} and ${other.shape}`);
    }
    if (shouldCreateGradient(input, other)) {
        return functions.MulFunction.apply(input, other);
    }
    return new Tensor(input.impl.mul(other.impl, alpha));
}
export function sub(input: Tensor, other: Tensor, alpha?: number): Tensor {
    if (input.shape.length !== other.shape.length) {
        throw new Error(`Shape dimensions of sub must match. Got ${input.shape} and ${other.shape}`);
    }
    if (shouldCreateGradient(input, other)) {
        return functions.SubFunction.apply(input, other);
    }
    return new Tensor(input.impl.sub(other.impl, alpha));
}
