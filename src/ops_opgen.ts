import * as functions from "./functions_opgen";
import { Tensor } from "./tensor";
import { unary, unaryWithAlpha, binary, binaryWithAlpha, reduction } from "./ops_high";
/**
* ![Plot of abs and its gradient](../../plots/abs.svg)
*
* Calculates:
* ```js
* output = abs(input)
* ```
*
* Gradient:
* ```js
* inputGrad = input == 0 ? 0 : (input > 0 ? outputGrad : -outputGrad)
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function abs(input: Tensor): Tensor {
    return unary(functions.AbsFunction, input);
}
/**
* Alias for `abs`.
*
* ![Plot of abs and its gradient](../../plots/abs.svg)
*
* Calculates:
* ```js
* output = abs(input)
* ```
*
* Gradient:
* ```js
* inputGrad = input == 0 ? 0 : (input > 0 ? outputGrad : -outputGrad)
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function absolute(input: Tensor): Tensor {
    return abs(input);
}
/**
* ![Plot of acos and its gradient](../../plots/acos.svg)
*
* Calculates:
* ```js
* output = acos(input)
* ```
*
* Gradient:
* ```js
* inputGrad = -outputGrad / sqrt(1 - input * input)
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function acos(input: Tensor): Tensor {
    return unary(functions.AcosFunction, input);
}
/**
* Alias for `acos`.
*
* ![Plot of acos and its gradient](../../plots/acos.svg)
*
* Calculates:
* ```js
* output = acos(input)
* ```
*
* Gradient:
* ```js
* inputGrad = -outputGrad / sqrt(1 - input * input)
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function arccos(input: Tensor): Tensor {
    return acos(input);
}
/**
* ![Plot of acosh and its gradient](../../plots/acosh.svg)
*
* Calculates:
* ```js
* output = acosh(input)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad / sqrt(input * input - 1)
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function acosh(input: Tensor): Tensor {
    return unary(functions.AcoshFunction, input);
}
/**
* Alias for `acosh`.
*
* ![Plot of acosh and its gradient](../../plots/acosh.svg)
*
* Calculates:
* ```js
* output = acosh(input)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad / sqrt(input * input - 1)
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function arccosh(input: Tensor): Tensor {
    return acosh(input);
}
/**
* Calculates:
* ```js
* output = input + other * alpha
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad; otherGrad = outputGrad
* ```
*
* @param input the input tensor of any shape
* @param other the other tensor whose shape is broadcastable with the input tensor
* @param alpha the alpha value to multiply `other` with
* @returns the output tensor
*/
export function add(input: Tensor, other: number | Tensor, alpha?: number): Tensor {
    return binaryWithAlpha(functions.AddFunction, input, other, alpha);
}
/**
* ![Plot of asin and its gradient](../../plots/asin.svg)
*
* Calculates:
* ```js
* output = asin(input)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad / sqrt(1 - input * input)
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function asin(input: Tensor): Tensor {
    return unary(functions.AsinFunction, input);
}
/**
* Alias for `asin`.
*
* ![Plot of asin and its gradient](../../plots/asin.svg)
*
* Calculates:
* ```js
* output = asin(input)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad / sqrt(1 - input * input)
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function arcsin(input: Tensor): Tensor {
    return asin(input);
}
/**
* ![Plot of asinh and its gradient](../../plots/asinh.svg)
*
* Calculates:
* ```js
* output = asinh(input)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad / sqrt(input * input + 1)
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function asinh(input: Tensor): Tensor {
    return unary(functions.AsinhFunction, input);
}
/**
* Alias for `asinh`.
*
* ![Plot of asinh and its gradient](../../plots/asinh.svg)
*
* Calculates:
* ```js
* output = asinh(input)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad / sqrt(input * input + 1)
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function arcsinh(input: Tensor): Tensor {
    return asinh(input);
}
/**
* ![Plot of atan and its gradient](../../plots/atan.svg)
*
* Calculates:
* ```js
* output = atan(input)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad / (input * input + 1)
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function atan(input: Tensor): Tensor {
    return unary(functions.AtanFunction, input);
}
/**
* Alias for `atan`.
*
* ![Plot of atan and its gradient](../../plots/atan.svg)
*
* Calculates:
* ```js
* output = atan(input)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad / (input * input + 1)
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function arctan(input: Tensor): Tensor {
    return atan(input);
}
/**
* Calculates:
* ```js
* output = atan2(input, other)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad * other / (input * input + other * other); otherGrad = -outputGrad * input / (input * input + other * other)
* ```
*
* @param input the input tensor of any shape
* @param other the other tensor whose shape is broadcastable with the input tensor
* @returns the output tensor
*/
export function atan2(input: Tensor, other: number | Tensor): Tensor {
    return binary(functions.Atan2Function, input, other);
}
/**
* Alias for `atan2`.
*
* Calculates:
* ```js
* output = atan2(input, other)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad * other / (input * input + other * other); otherGrad = -outputGrad * input / (input * input + other * other)
* ```
*
* @param input the input tensor of any shape
* @param other the other tensor whose shape is broadcastable with the input tensor
* @returns the output tensor
*/
export function arctan2(input: Tensor, other: number | Tensor): Tensor {
    return atan2(input, other);
}
/**
* ![Plot of ceil and its gradient](../../plots/ceil.svg)
*
* Calculates:
* ```js
* output = ceil(input)
* ```
*
* Gradient:
* ```js
* inputGrad = 0
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function ceil(input: Tensor): Tensor {
    return unary(functions.CeilFunction, input);
}
/**
* Calculates:
* ```js
* output = other >= 0 ? abs(input) : -abs(input)
* ```
*
* Gradient:
* ```js
* var dir = other >= 0 ? (input >= 0 ? 1.0 : -1.0) : (input >= 0 ? -1.0 : 1.0); inputGrad = input == 0.0 ? 0.0 : outputGrad * dir; otherGrad = 0
* ```
*
* @param input the input tensor of any shape
* @param other the other tensor whose shape is broadcastable with the input tensor
* @returns the output tensor
*/
export function copysign(input: Tensor, other: number | Tensor): Tensor {
    return binary(functions.CopysignFunction, input, other);
}
/**
* ![Plot of cos and its gradient](../../plots/cos.svg)
*
* Calculates:
* ```js
* output = cos(input)
* ```
*
* Gradient:
* ```js
* inputGrad = -outputGrad * sin(input)
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function cos(input: Tensor): Tensor {
    return unary(functions.CosFunction, input);
}
/**
* ![Plot of cosh and its gradient](../../plots/cosh.svg)
*
* Calculates:
* ```js
* output = cosh(input)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad * sinh(input)
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function cosh(input: Tensor): Tensor {
    return unary(functions.CoshFunction, input);
}
/**
* ![Plot of deg2rad and its gradient](../../plots/deg2rad.svg)
*
* Calculates:
* ```js
* output = input * 0.017453292519943295
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad * 0.017453292519943295
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function deg2rad(input: Tensor): Tensor {
    return unary(functions.Deg2radFunction, input);
}
/**
* Calculates:
* ```js
* output = input / other
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad / other; otherGrad = -outputGrad * input / (other * other)
* ```
*
* @param input the input tensor of any shape
* @param other the other tensor whose shape is broadcastable with the input tensor
* @returns the output tensor
*/
export function div(input: Tensor, other: number | Tensor): Tensor {
    return binary(functions.DivFunction, input, other);
}
/**
* Alias for `div`.
*
* Calculates:
* ```js
* output = input / other
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad / other; otherGrad = -outputGrad * input / (other * other)
* ```
*
* @param input the input tensor of any shape
* @param other the other tensor whose shape is broadcastable with the input tensor
* @returns the output tensor
*/
export function divide(input: Tensor, other: number | Tensor): Tensor {
    return div(input, other);
}
/**
* ![Plot of exp and its gradient](../../plots/exp.svg)
*
* Calculates:
* ```js
* output = exp(input)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad * exp(input)
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function exp(input: Tensor): Tensor {
    return unary(functions.ExpFunction, input);
}
/**
* ![Plot of exp2 and its gradient](../../plots/exp2.svg)
*
* Calculates:
* ```js
* output = exp2(input)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad * exp2(input) * 0.6931471805599453
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function exp2(input: Tensor): Tensor {
    return unary(functions.Exp2Function, input);
}
/**
* ![Plot of expm1 and its gradient](../../plots/expm1.svg)
*
* Calculates:
* ```js
* output = exp(input) - 1.0
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad * exp(input)
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function expm1(input: Tensor): Tensor {
    return unary(functions.Expm1Function, input);
}
/**
* ![Plot of floor and its gradient](../../plots/floor.svg)
*
* Calculates:
* ```js
* output = floor(input)
* ```
*
* Gradient:
* ```js
* inputGrad = 0
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function floor(input: Tensor): Tensor {
    return unary(functions.FloorFunction, input);
}
/**
* ![Plot of frac and its gradient](../../plots/frac.svg)
*
* Calculates:
* ```js
* output = input >= 0.0 ? fract(input) : -fract(-input)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function frac(input: Tensor): Tensor {
    return unary(functions.FracFunction, input);
}
/**
* Calculates:
* ```js
* output = sqrt(input * input + other * other)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad * input / sqrt(input * input + other * other); otherGrad = outputGrad * other / sqrt(input * input + other * other)
* ```
*
* @param input the input tensor of any shape
* @param other the other tensor whose shape is broadcastable with the input tensor
* @returns the output tensor
*/
export function hypot(input: Tensor, other: number | Tensor): Tensor {
    return binary(functions.HypotFunction, input, other);
}
/**
* Calculates:
* ```js
* output = input * pow(2.0, other)
* ```
*
* Gradient:
* ```js
* var out = pow(2.0, other); inputGrad = outputGrad * out; otherGrad = outputGrad * input * out * 0.6931471805599453
* ```
*
* @param input the input tensor of any shape
* @param other the other tensor whose shape is broadcastable with the input tensor
* @returns the output tensor
*/
export function ldexp(input: Tensor, other: number | Tensor): Tensor {
    return binary(functions.LdexpFunction, input, other);
}
/**
* ![Plot of log and its gradient](../../plots/log.svg)
*
* Calculates:
* ```js
* output = log(input)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad / input
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function log(input: Tensor): Tensor {
    return unary(functions.LogFunction, input);
}
/**
* ![Plot of log10 and its gradient](../../plots/log10.svg)
*
* Calculates:
* ```js
* output = log(input) * 0.4342944819032518
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad / (input * 2.302585092994046)
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function log10(input: Tensor): Tensor {
    return unary(functions.Log10Function, input);
}
/**
* ![Plot of log1p and its gradient](../../plots/log1p.svg)
*
* Calculates:
* ```js
* output = log(input + 1.0)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad / (input + 1.0)
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function log1p(input: Tensor): Tensor {
    return unary(functions.Log1pFunction, input);
}
/**
* ![Plot of log2 and its gradient](../../plots/log2.svg)
*
* Calculates:
* ```js
* output = log2(input)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad / (input * 0.6931471805599453)
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function log2(input: Tensor): Tensor {
    return unary(functions.Log2Function, input);
}
/**
* Calculates:
* ```js
* output = log(exp(input) + exp(other))
* ```
*
* Gradient:
* ```js
* var ein = exp(input); var eoth = exp(other); var addeinv = outputGrad/(ein + eoth); inputGrad = addeinv * ein; otherGrad = addeinv * eoth
* ```
*
* @param input the input tensor of any shape
* @param other the other tensor whose shape is broadcastable with the input tensor
* @returns the output tensor
*/
export function logaddexp(input: Tensor, other: number | Tensor): Tensor {
    return binary(functions.LogaddexpFunction, input, other);
}
/**
* Calculates:
* ```js
* output = log2(exp2(input) + exp2(other))
* ```
*
* Gradient:
* ```js
* var ein = exp2(input); var eoth = exp2(other); var sum_ein_eoth = ein + eoth; inputGrad = outputGrad * (ein / sum_ein_eoth); otherGrad = outputGrad * (eoth / sum_ein_eoth );
* ```
*
* @param input the input tensor of any shape
* @param other the other tensor whose shape is broadcastable with the input tensor
* @returns the output tensor
*/
export function logaddexp2(input: Tensor, other: number | Tensor): Tensor {
    return binary(functions.Logaddexp2Function, input, other);
}
/**
* Calculates:
* ```js
* output = input * other
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad * other; otherGrad = outputGrad * input
* ```
*
* @param input the input tensor of any shape
* @param other the other tensor whose shape is broadcastable with the input tensor
* @returns the output tensor
*/
export function mul(input: Tensor, other: number | Tensor): Tensor {
    return binary(functions.MulFunction, input, other);
}
/**
* Alias for `mul`.
*
* Calculates:
* ```js
* output = input * other
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad * other; otherGrad = outputGrad * input
* ```
*
* @param input the input tensor of any shape
* @param other the other tensor whose shape is broadcastable with the input tensor
* @returns the output tensor
*/
export function multiply(input: Tensor, other: number | Tensor): Tensor {
    return mul(input, other);
}
/**
* ![Plot of neg and its gradient](../../plots/neg.svg)
*
* Calculates:
* ```js
* output = -input
* ```
*
* Gradient:
* ```js
* inputGrad = -outputGrad
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function neg(input: Tensor): Tensor {
    return unary(functions.NegFunction, input);
}
/**
* Alias for `neg`.
*
* ![Plot of neg and its gradient](../../plots/neg.svg)
*
* Calculates:
* ```js
* output = -input
* ```
*
* Gradient:
* ```js
* inputGrad = -outputGrad
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function negative(input: Tensor): Tensor {
    return neg(input);
}
/**
* ![Plot of positive and its gradient](../../plots/positive.svg)
*
* Calculates:
* ```js
* output = input
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function positive(input: Tensor): Tensor {
    return unary(functions.PositiveFunction, input);
}
/**
* Calculates:
* ```js
* output = input
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad * other * pow(input, other - 1.0); otherGrad = outputGrad * pow(input, other) * log(input)
* ```
*
* @param input the input tensor of any shape
* @param other the other tensor whose shape is broadcastable with the input tensor
* @returns the output tensor
*/
export function pow(input: Tensor, other: number | Tensor): Tensor {
    return binary(functions.PowFunction, input, other);
}
/**
* ![Plot of rad2deg and its gradient](../../plots/rad2deg.svg)
*
* Calculates:
* ```js
* output = input * 57.29577951308232
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad * 57.29577951308232
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function rad2deg(input: Tensor): Tensor {
    return unary(functions.Rad2degFunction, input);
}
/**
* ![Plot of reciprocal and its gradient](../../plots/reciprocal.svg)
*
* Calculates:
* ```js
* output = 1.0 / input
* ```
*
* Gradient:
* ```js
* inputGrad = -outputGrad / (input * input)
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function reciprocal(input: Tensor): Tensor {
    return unary(functions.ReciprocalFunction, input);
}
/**
* ![Plot of relu and its gradient](../../plots/relu.svg)
*
* Calculates:
* ```js
* output = max(input, 0.0)
* ```
*
* Gradient:
* ```js
* inputGrad = input > 0.0 ? outputGrad : 0.0
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function relu(input: Tensor): Tensor {
    return unary(functions.ReluFunction, input);
}
/**
* ![Plot of round and its gradient](../../plots/round.svg)
*
* Calculates:
* ```js
* output = round(input)
* ```
*
* Gradient:
* ```js
* inputGrad = 0
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function round(input: Tensor): Tensor {
    return unary(functions.RoundFunction, input);
}
/**
* ![Plot of rsqrt and its gradient](../../plots/rsqrt.svg)
*
* Calculates:
* ```js
* output = 1.0 / sqrt(input)
* ```
*
* Gradient:
* ```js
* inputGrad = -outputGrad / (2.0 * sqrt(input) * input)
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function rsqrt(input: Tensor): Tensor {
    return unary(functions.RsqrtFunction, input);
}
/**
* ![Plot of sigmoid and its gradient](../../plots/sigmoid.svg)
*
* Calculates:
* ```js
* output = 1.0 / (1.0 + exp(-input))
* ```
*
* Gradient:
* ```js
* var out = 1.0 / (1.0 + exp(-input)); inputGrad = outputGrad * out * (1.0 - out)
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function sigmoid(input: Tensor): Tensor {
    return unary(functions.SigmoidFunction, input);
}
/**
* ![Plot of sign and its gradient](../../plots/sign.svg)
*
* Calculates:
* ```js
* output = sign(input)
* ```
*
* Gradient:
* ```js
* inputGrad = 0
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function sign(input: Tensor): Tensor {
    return unary(functions.SignFunction, input);
}
/**
* ![Plot of silu and its gradient](../../plots/silu.svg)
*
* Calculates:
* ```js
* output = input / (1.0 + exp(-input))
* ```
*
* Gradient:
* ```js
* var out = 1.0 / (1.0 + exp(-input)); inputGrad = outputGrad * (out + input * out * (1.0 - out))
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function silu(input: Tensor): Tensor {
    return unary(functions.SiluFunction, input);
}
/**
* ![Plot of sin and its gradient](../../plots/sin.svg)
*
* Calculates:
* ```js
* output = sin(input)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad * cos(input)
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function sin(input: Tensor): Tensor {
    return unary(functions.SinFunction, input);
}
/**
* ![Plot of sinc and its gradient](../../plots/sinc.svg)
*
* Calculates:
* ```js
* var inpi = input * 3.141592653589793; output = input == 0.0 ? 1.0 : sin(inpi) / inpi
* ```
*
* Gradient:
* ```js
* var inpi = input * 3.141592653589793; inputGrad = input == 0.0 ? 0.0 : (outputGrad * 3.141592653589793 * (inpi*cos(inpi) - sin(inpi)) / (inpi*inpi))
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function sinc(input: Tensor): Tensor {
    return unary(functions.SincFunction, input);
}
/**
* ![Plot of sinh and its gradient](../../plots/sinh.svg)
*
* Calculates:
* ```js
* output = sinh(input)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad * cosh(input)
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function sinh(input: Tensor): Tensor {
    return unary(functions.SinhFunction, input);
}
/**
* ![Plot of sqrt and its gradient](../../plots/sqrt.svg)
*
* Calculates:
* ```js
* output = sqrt(input)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad / (2.0 * sqrt(input))
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function sqrt(input: Tensor): Tensor {
    return unary(functions.SqrtFunction, input);
}
/**
* ![Plot of square and its gradient](../../plots/square.svg)
*
* Calculates:
* ```js
* output = input * input
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad * 2.0 * input
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function square(input: Tensor): Tensor {
    return unary(functions.SquareFunction, input);
}
/**
* Calculates:
* ```js
* output = input - other * alpha
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad; otherGrad = -outputGrad
* ```
*
* @param input the input tensor of any shape
* @param other the other tensor whose shape is broadcastable with the input tensor
* @param alpha the alpha value to multiply `other` with
* @returns the output tensor
*/
export function sub(input: Tensor, other: number | Tensor, alpha?: number): Tensor {
    return binaryWithAlpha(functions.SubFunction, input, other, alpha);
}
/**
* Alias for `sub`.
*
* Calculates:
* ```js
* output = input - other * alpha
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad; otherGrad = -outputGrad
* ```
*
* @param input the input tensor of any shape
* @param other the other tensor whose shape is broadcastable with the input tensor
* @param alpha the alpha value to multiply `other` with
* @returns the output tensor
*/
export function subtract(input: Tensor, other: number | Tensor, alpha?: number): Tensor {
    return sub(input, other, alpha);
}
/**
* ![Plot of tan and its gradient](../../plots/tan.svg)
*
* Calculates:
* ```js
* output = tan(input)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad / (cos(input) * cos(input))
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function tan(input: Tensor): Tensor {
    return unary(functions.TanFunction, input);
}
/**
* ![Plot of tanh and its gradient](../../plots/tanh.svg)
*
* Calculates:
* ```js
* output = tanh(input)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad * (1.0 - tanh(input) * tanh(input))
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function tanh(input: Tensor): Tensor {
    return unary(functions.TanhFunction, input);
}
/**
* ![Plot of trunc and its gradient](../../plots/trunc.svg)
*
* Calculates:
* ```js
* output = trunc(input)
* ```
*
* Gradient:
* ```js
* inputGrad = 0
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function trunc(input: Tensor): Tensor {
    return unary(functions.TruncFunction, input);
}
/**
* Alias for `trunc`.
*
* ![Plot of trunc and its gradient](../../plots/trunc.svg)
*
* Calculates:
* ```js
* output = trunc(input)
* ```
*
* Gradient:
* ```js
* inputGrad = 0
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function fix(input: Tensor): Tensor {
    return trunc(input);
}
/**
* Calculates:
* ```js
* output = input == 0.0 ? 0.0 : input * log(other)
* ```
*
* Gradient:
* ```js
* inputGrad = input == 0.0 ? 0.0 : outputGrad * log(other); otherGrad = input == 0.0 ? 0.0 : outputGrad * (input / other);
* ```
*
* @param input the input tensor of any shape
* @param other the other tensor whose shape is broadcastable with the input tensor
* @returns the output tensor
*/
export function xlogy(input: Tensor, other: number | Tensor): Tensor {
    return binary(functions.XlogyFunction, input, other);
}
/**
* Calculates:
* ```js
* output = output && input
* ```
*
* with an initial value of `output = 1`.
*
* Gradient:
* ```js
* inputGrad = output ? outputGrad : 0.0
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function all(input: Tensor, dim?: number | number[], keepdim?: boolean): Tensor {
    return reduction(functions.AllFunction, input, dim, keepdim);
}
/**
* Calculates:
* ```js
* output = output || input
* ```
*
* with an initial value of `output = 0`.
*
* Gradient:
* ```js
* inputGrad = output ? outputGrad : 0.0
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function any(input: Tensor, dim?: number | number[], keepdim?: boolean): Tensor {
    return reduction(functions.AnyFunction, input, dim, keepdim);
}
/**
* Calculates:
* ```js
* output = output + input
* ```
*
* with an initial value of `output = 0.0`.
*
* Gradient:
* ```js
* inputGrad = outputGrad / f32(inputSize)
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function mean(input: Tensor, dim?: number | number[], keepdim?: boolean): Tensor {
    return reduction(functions.MeanFunction, input, dim, keepdim);
}
/**
* Calculates:
* ```js
* output = output + input * input
* ```
*
* with an initial value of `output = 0.0`.
*
* Gradient:
* ```js
* inputGrad = outputGrad * input / output
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function norm(input: Tensor, dim?: number | number[], keepdim?: boolean): Tensor {
    return reduction(functions.NormFunction, input, dim, keepdim);
}
/**
* Calculates:
* ```js
* output = output * input
* ```
*
* with an initial value of `output = 1.0`.
*
* Gradient:
* ```js
* inputGrad = outputGrad * output / input
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function prod(input: Tensor, dim?: number | number[], keepdim?: boolean): Tensor {
    return reduction(functions.ProdFunction, input, dim, keepdim);
}
/**
* Calculates:
* ```js
* output = output + input
* ```
*
* with an initial value of `output = 0.0`.
*
* Gradient:
* ```js
* inputGrad = outputGrad
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function sum(input: Tensor, dim?: number | number[], keepdim?: boolean): Tensor {
    return reduction(functions.SumFunction, input, dim, keepdim);
}
/**
* Calculates:
* ```js
* output = output + (input != 0)
* ```
*
* with an initial value of `output = 0.0`.
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function countNonzero(input: Tensor, dim?: number | number[], keepdim?: boolean): Tensor {
    return reduction(functions.CountNonzeroFunction, input, dim, keepdim);
}
