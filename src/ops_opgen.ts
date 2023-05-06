import * as functions from "./functions";
import { Tensor } from "./tensor";
import { unary, unaryWithAlpha, binary, binaryWithAlpha } from "./ops_high";
export function abs(input: Tensor): Tensor {
    return unary(functions.AbsFunction, input);
}
export function absolute(input: Tensor): Tensor {
    return abs(input);
}
export function acos(input: Tensor): Tensor {
    return unary(functions.AcosFunction, input);
}
export function arccos(input: Tensor): Tensor {
    return acos(input);
}
export function acosh(input: Tensor): Tensor {
    return unary(functions.AcoshFunction, input);
}
export function arccosh(input: Tensor): Tensor {
    return acosh(input);
}
export function add(input: Tensor, other: Tensor, alpha?: number): Tensor {
    return binaryWithAlpha(functions.AddFunction, input, other, alpha);
}
export function asin(input: Tensor): Tensor {
    return unary(functions.AsinFunction, input);
}
export function arcsin(input: Tensor): Tensor {
    return asin(input);
}
export function asinh(input: Tensor): Tensor {
    return unary(functions.AsinhFunction, input);
}
export function arcsinh(input: Tensor): Tensor {
    return asinh(input);
}
export function atan(input: Tensor): Tensor {
    return unary(functions.AtanFunction, input);
}
export function arctan(input: Tensor): Tensor {
    return atan(input);
}
export function atan2(input: Tensor, other: Tensor): Tensor {
    return binary(functions.Atan2Function, input, other);
}
export function arctan2(input: Tensor, other: Tensor): Tensor {
    return atan2(input, other);
}
export function ceil(input: Tensor): Tensor {
    return unary(functions.CeilFunction, input);
}
export function copysign(input: Tensor, other: Tensor): Tensor {
    return binary(functions.CopysignFunction, input, other);
}
export function cos(input: Tensor): Tensor {
    return unary(functions.CosFunction, input);
}
export function cosh(input: Tensor): Tensor {
    return unary(functions.CoshFunction, input);
}
export function deg2rad(input: Tensor): Tensor {
    return unary(functions.Deg2radFunction, input);
}
export function div(input: Tensor, other: Tensor, alpha?: number): Tensor {
    return binaryWithAlpha(functions.DivFunction, input, other, alpha);
}
export function divide(input: Tensor, other: Tensor, alpha?: number): Tensor {
    return div(input, other, alpha);
}
export function exp(input: Tensor): Tensor {
    return unary(functions.ExpFunction, input);
}
export function exp2(input: Tensor): Tensor {
    return unary(functions.Exp2Function, input);
}
export function expm1(input: Tensor): Tensor {
    return unary(functions.Expm1Function, input);
}
export function floor(input: Tensor): Tensor {
    return unary(functions.FloorFunction, input);
}
export function floor_divide(input: Tensor, other: Tensor): Tensor {
    return binary(functions.Floor_divideFunction, input, other);
}
export function frac(input: Tensor): Tensor {
    return unary(functions.FracFunction, input);
}
export function hypot(input: Tensor, other: Tensor): Tensor {
    return binary(functions.HypotFunction, input, other);
}
export function ldexp(input: Tensor, other: Tensor): Tensor {
    return binary(functions.LdexpFunction, input, other);
}
export function log(input: Tensor): Tensor {
    return unary(functions.LogFunction, input);
}
export function log10(input: Tensor): Tensor {
    return unary(functions.Log10Function, input);
}
export function log1p(input: Tensor): Tensor {
    return unary(functions.Log1pFunction, input);
}
export function log2(input: Tensor): Tensor {
    return unary(functions.Log2Function, input);
}
export function logaddexp(input: Tensor, other: Tensor): Tensor {
    return binary(functions.LogaddexpFunction, input, other);
}
export function logaddexp2(input: Tensor, other: Tensor): Tensor {
    return binary(functions.Logaddexp2Function, input, other);
}
export function mul(input: Tensor, other: Tensor, alpha?: number): Tensor {
    return binaryWithAlpha(functions.MulFunction, input, other, alpha);
}
export function multiply(input: Tensor, other: Tensor, alpha?: number): Tensor {
    return mul(input, other, alpha);
}
export function neg(input: Tensor): Tensor {
    return unary(functions.NegFunction, input);
}
export function negative(input: Tensor): Tensor {
    return neg(input);
}
export function positive(input: Tensor): Tensor {
    return unary(functions.PositiveFunction, input);
}
export function pow(input: Tensor, other: Tensor): Tensor {
    return binary(functions.PowFunction, input, other);
}
export function rad2deg(input: Tensor): Tensor {
    return unary(functions.Rad2degFunction, input);
}
export function reciprocal(input: Tensor): Tensor {
    return unary(functions.ReciprocalFunction, input);
}
export function round(input: Tensor): Tensor {
    return unary(functions.RoundFunction, input);
}
export function rsqrt(input: Tensor): Tensor {
    return unary(functions.RsqrtFunction, input);
}
export function sigmoid(input: Tensor): Tensor {
    return unary(functions.SigmoidFunction, input);
}
export function sign(input: Tensor): Tensor {
    return unary(functions.SignFunction, input);
}
export function sin(input: Tensor): Tensor {
    return unary(functions.SinFunction, input);
}
export function sinc(input: Tensor): Tensor {
    return unary(functions.SincFunction, input);
}
export function sinh(input: Tensor): Tensor {
    return unary(functions.SinhFunction, input);
}
export function sqrt(input: Tensor): Tensor {
    return unary(functions.SqrtFunction, input);
}
export function square(input: Tensor): Tensor {
    return unary(functions.SquareFunction, input);
}
export function sub(input: Tensor, other: Tensor, alpha?: number): Tensor {
    return binaryWithAlpha(functions.SubFunction, input, other, alpha);
}
export function subtract(input: Tensor, other: Tensor, alpha?: number): Tensor {
    return sub(input, other, alpha);
}
export function tan(input: Tensor): Tensor {
    return unary(functions.TanFunction, input);
}
export function tanh(input: Tensor): Tensor {
    return unary(functions.TanhFunction, input);
}
export function trunc(input: Tensor): Tensor {
    return unary(functions.TruncFunction, input);
}
export function fix(input: Tensor): Tensor {
    return trunc(input);
}
export function xlogy(input: Tensor, other: Tensor): Tensor {
    return binary(functions.XlogyFunction, input, other);
}
