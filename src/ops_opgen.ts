import * as functions from "./functions";
import { Tensor } from "./tensor";
import { shouldCreateGradient } from "./autograd";
export function abs(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.AbsFunction.apply(input);
    }
    return new Tensor(input.impl.abs());
}
export function absolute(input: Tensor): Tensor {
    return abs(input);
}
export function acos(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.AcosFunction.apply(input);
    }
    return new Tensor(input.impl.acos());
}
export function arccos(input: Tensor): Tensor {
    return acos(input);
}
export function acosh(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.AcoshFunction.apply(input);
    }
    return new Tensor(input.impl.acosh());
}
export function arccosh(input: Tensor): Tensor {
    return acosh(input);
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
export function arcsin(input: Tensor): Tensor {
    return asin(input);
}
export function asinh(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.AsinhFunction.apply(input);
    }
    return new Tensor(input.impl.asinh());
}
export function arcsinh(input: Tensor): Tensor {
    return asinh(input);
}
export function atan(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.AtanFunction.apply(input);
    }
    return new Tensor(input.impl.atan());
}
export function arctan(input: Tensor): Tensor {
    return atan(input);
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
export function arctan2(input: Tensor, other: Tensor): Tensor {
    return atan2(input, other);
}
export function ceil(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.CeilFunction.apply(input);
    }
    return new Tensor(input.impl.ceil());
}
export function copysign(input: Tensor, other: Tensor): Tensor {
    if (input.shape.length !== other.shape.length) {
        throw new Error(`Shape dimensions of copysign must match. Got ${input.shape} and ${other.shape}`);
    }
    if (shouldCreateGradient(input, other)) {
        return functions.CopysignFunction.apply(input, other);
    }
    return new Tensor(input.impl.copysign(other.impl));
}
export function cos(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.CosFunction.apply(input);
    }
    return new Tensor(input.impl.cos());
}
export function cosh(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.CoshFunction.apply(input);
    }
    return new Tensor(input.impl.cosh());
}
export function deg2rad(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.Deg2radFunction.apply(input);
    }
    return new Tensor(input.impl.deg2rad());
}
export function div(input: Tensor, other: Tensor, alpha?: number): Tensor {
    if (input.shape.length !== other.shape.length) {
        throw new Error(`Shape dimensions of div must match. Got ${input.shape} and ${other.shape}`);
    }
    if (shouldCreateGradient(input, other)) {
        return functions.DivFunction.apply(input, other);
    }
    return new Tensor(input.impl.div(other.impl, alpha));
}
export function divide(input: Tensor, other: Tensor, alpha?: number): Tensor {
    return div(input, other, alpha);
}
export function exp(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.ExpFunction.apply(input);
    }
    return new Tensor(input.impl.exp());
}
export function exp2(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.Exp2Function.apply(input);
    }
    return new Tensor(input.impl.exp2());
}
export function expm1(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.Expm1Function.apply(input);
    }
    return new Tensor(input.impl.expm1());
}
export function floor(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.FloorFunction.apply(input);
    }
    return new Tensor(input.impl.floor());
}
export function floor_divide(input: Tensor, other: Tensor): Tensor {
    if (input.shape.length !== other.shape.length) {
        throw new Error(`Shape dimensions of floor_divide must match. Got ${input.shape} and ${other.shape}`);
    }
    if (shouldCreateGradient(input, other)) {
        return functions.Floor_divideFunction.apply(input, other);
    }
    return new Tensor(input.impl.floor_divide(other.impl));
}
export function frac(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.FracFunction.apply(input);
    }
    return new Tensor(input.impl.frac());
}
export function hypot(input: Tensor, other: Tensor): Tensor {
    if (input.shape.length !== other.shape.length) {
        throw new Error(`Shape dimensions of hypot must match. Got ${input.shape} and ${other.shape}`);
    }
    if (shouldCreateGradient(input, other)) {
        return functions.HypotFunction.apply(input, other);
    }
    return new Tensor(input.impl.hypot(other.impl));
}
export function ldexp(input: Tensor, other: Tensor): Tensor {
    if (input.shape.length !== other.shape.length) {
        throw new Error(`Shape dimensions of ldexp must match. Got ${input.shape} and ${other.shape}`);
    }
    if (shouldCreateGradient(input, other)) {
        return functions.LdexpFunction.apply(input, other);
    }
    return new Tensor(input.impl.ldexp(other.impl));
}
export function log(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.LogFunction.apply(input);
    }
    return new Tensor(input.impl.log());
}
export function log10(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.Log10Function.apply(input);
    }
    return new Tensor(input.impl.log10());
}
export function log1p(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.Log1pFunction.apply(input);
    }
    return new Tensor(input.impl.log1p());
}
export function log2(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.Log2Function.apply(input);
    }
    return new Tensor(input.impl.log2());
}
export function logaddexp(input: Tensor, other: Tensor): Tensor {
    if (input.shape.length !== other.shape.length) {
        throw new Error(`Shape dimensions of logaddexp must match. Got ${input.shape} and ${other.shape}`);
    }
    if (shouldCreateGradient(input, other)) {
        return functions.LogaddexpFunction.apply(input, other);
    }
    return new Tensor(input.impl.logaddexp(other.impl));
}
export function logaddexp2(input: Tensor, other: Tensor): Tensor {
    if (input.shape.length !== other.shape.length) {
        throw new Error(`Shape dimensions of logaddexp2 must match. Got ${input.shape} and ${other.shape}`);
    }
    if (shouldCreateGradient(input, other)) {
        return functions.Logaddexp2Function.apply(input, other);
    }
    return new Tensor(input.impl.logaddexp2(other.impl));
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
export function multiply(input: Tensor, other: Tensor, alpha?: number): Tensor {
    return mul(input, other, alpha);
}
export function neg(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.NegFunction.apply(input);
    }
    return new Tensor(input.impl.neg());
}
export function negative(input: Tensor): Tensor {
    return neg(input);
}
export function positive(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.PositiveFunction.apply(input);
    }
    return new Tensor(input.impl.positive());
}
export function pow(input: Tensor, other: Tensor): Tensor {
    if (input.shape.length !== other.shape.length) {
        throw new Error(`Shape dimensions of pow must match. Got ${input.shape} and ${other.shape}`);
    }
    if (shouldCreateGradient(input, other)) {
        return functions.PowFunction.apply(input, other);
    }
    return new Tensor(input.impl.pow(other.impl));
}
export function rad2deg(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.Rad2degFunction.apply(input);
    }
    return new Tensor(input.impl.rad2deg());
}
export function reciprocal(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.ReciprocalFunction.apply(input);
    }
    return new Tensor(input.impl.reciprocal());
}
export function round(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.RoundFunction.apply(input);
    }
    return new Tensor(input.impl.round());
}
export function rsqrt(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.RsqrtFunction.apply(input);
    }
    return new Tensor(input.impl.rsqrt());
}
export function sigmoid(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.SigmoidFunction.apply(input);
    }
    return new Tensor(input.impl.sigmoid());
}
export function sign(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.SignFunction.apply(input);
    }
    return new Tensor(input.impl.sign());
}
export function sin(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.SinFunction.apply(input);
    }
    return new Tensor(input.impl.sin());
}
export function sinc(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.SincFunction.apply(input);
    }
    return new Tensor(input.impl.sinc());
}
export function sinh(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.SinhFunction.apply(input);
    }
    return new Tensor(input.impl.sinh());
}
export function sqrt(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.SqrtFunction.apply(input);
    }
    return new Tensor(input.impl.sqrt());
}
export function square(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.SquareFunction.apply(input);
    }
    return new Tensor(input.impl.square());
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
export function subtract(input: Tensor, other: Tensor, alpha?: number): Tensor {
    return sub(input, other, alpha);
}
export function tan(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.TanFunction.apply(input);
    }
    return new Tensor(input.impl.tan());
}
export function tanh(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.TanhFunction.apply(input);
    }
    return new Tensor(input.impl.tanh());
}
export function trunc(input: Tensor): Tensor {
    if (shouldCreateGradient(input)) {
        return functions.TruncFunction.apply(input);
    }
    return new Tensor(input.impl.trunc());
}
export function fix(input: Tensor): Tensor {
    return trunc(input);
}
export function xlogy(input: Tensor, other: Tensor): Tensor {
    if (input.shape.length !== other.shape.length) {
        throw new Error(`Shape dimensions of xlogy must match. Got ${input.shape} and ${other.shape}`);
    }
    if (shouldCreateGradient(input, other)) {
        return functions.XlogyFunction.apply(input, other);
    }
    return new Tensor(input.impl.xlogy(other.impl));
}
