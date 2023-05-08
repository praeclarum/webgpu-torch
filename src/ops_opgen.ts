import * as functions from "./functions";
import { Tensor } from "./tensor";
import { unary, unaryWithAlpha, binary, binaryWithAlpha } from "./ops_high";
/**
* Calculates:
* ```js
* output = abs(input)
* ```
*
* Gradient:
* ```js
* inputGrad = input > 0 ? outputGrad : -outputGrad
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
* Calculates:
* ```js
* output = abs(input)
* ```
*
* Gradient:
* ```js
* inputGrad = input > 0 ? outputGrad : -outputGrad
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function absolute(input: Tensor): Tensor {
    return abs(input);
}
/**
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
* output = input + other
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
export function add(input: Tensor, other: Tensor, alpha?: number): Tensor {
    return binaryWithAlpha(functions.AddFunction, input, other, alpha);
}
/**
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
export function atan2(input: Tensor, other: Tensor): Tensor {
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
export function arctan2(input: Tensor, other: Tensor): Tensor {
    return atan2(input, other);
}
/**
* Calculates:
* ```js
* output = ceil(input)
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
export function ceil(input: Tensor): Tensor {
    return unary(functions.CeilFunction, input);
}
/**
* Calculates:
* ```js
* output = other >= 0.0 ? abs(input) : -abs(input)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad * (other >= 0.0 ? 1.0 : -1.0)
* ```
*
* @param input the input tensor of any shape
* @param other the other tensor whose shape is broadcastable with the input tensor
* @returns the output tensor
*/
export function copysign(input: Tensor, other: Tensor): Tensor {
    return binary(functions.CopysignFunction, input, other);
}
/**
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
* @param alpha the alpha value to multiply `other` with
* @returns the output tensor
*/
export function div(input: Tensor, other: Tensor, alpha?: number): Tensor {
    return binaryWithAlpha(functions.DivFunction, input, other, alpha);
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
* @param alpha the alpha value to multiply `other` with
* @returns the output tensor
*/
export function divide(input: Tensor, other: Tensor, alpha?: number): Tensor {
    return div(input, other, alpha);
}
/**
* Calculates:
* ```js
* output = exp(input)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad * output
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function exp(input: Tensor): Tensor {
    return unary(functions.ExpFunction, input);
}
/**
* Calculates:
* ```js
* output = exp2(input)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad * output * 0.6931471805599453
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function exp2(input: Tensor): Tensor {
    return unary(functions.Exp2Function, input);
}
/**
* Calculates:
* ```js
* output = exp(input) - 1.0
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad * output
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function expm1(input: Tensor): Tensor {
    return unary(functions.Expm1Function, input);
}
/**
* Calculates:
* ```js
* output = floor(input)
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
export function floor(input: Tensor): Tensor {
    return unary(functions.FloorFunction, input);
}
/**
* Calculates:
* ```js
* output = floor(input / other)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad / other; otherGrad = -outputGrad * floor(input / other) / other
* ```
*
* @param input the input tensor of any shape
* @param other the other tensor whose shape is broadcastable with the input tensor
* @returns the output tensor
*/
export function floor_divide(input: Tensor, other: Tensor): Tensor {
    return binary(functions.Floor_divideFunction, input, other);
}
/**
* Calculates:
* ```js
* output = fract(input)
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
export function hypot(input: Tensor, other: Tensor): Tensor {
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
* inputGrad = outputGrad * pow(2.0, other); otherGrad = outputGrad * input * log(2.0)
* ```
*
* @param input the input tensor of any shape
* @param other the other tensor whose shape is broadcastable with the input tensor
* @returns the output tensor
*/
export function ldexp(input: Tensor, other: Tensor): Tensor {
    return binary(functions.LdexpFunction, input, other);
}
/**
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
* inputGrad = outputGrad * exp(input) / (exp(input) + exp(other)); otherGrad = outputGrad * exp(other) / (exp(input) + exp(other))
* ```
*
* @param input the input tensor of any shape
* @param other the other tensor whose shape is broadcastable with the input tensor
* @returns the output tensor
*/
export function logaddexp(input: Tensor, other: Tensor): Tensor {
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
* inputGrad = outputGrad * exp2(input) / (exp2(input) + exp2(other)) * 1.4426950408889634; otherGrad = outputGrad * exp2(other) / (exp2(input) + exp2(other)) * 1.4426950408889634
* ```
*
* @param input the input tensor of any shape
* @param other the other tensor whose shape is broadcastable with the input tensor
* @returns the output tensor
*/
export function logaddexp2(input: Tensor, other: Tensor): Tensor {
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
* @param alpha the alpha value to multiply `other` with
* @returns the output tensor
*/
export function mul(input: Tensor, other: Tensor, alpha?: number): Tensor {
    return binaryWithAlpha(functions.MulFunction, input, other, alpha);
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
* @param alpha the alpha value to multiply `other` with
* @returns the output tensor
*/
export function multiply(input: Tensor, other: Tensor, alpha?: number): Tensor {
    return mul(input, other, alpha);
}
/**
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
* output = pow(input, other)
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
export function pow(input: Tensor, other: Tensor): Tensor {
    return binary(functions.PowFunction, input, other);
}
/**
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
* Calculates:
* ```js
* output = round(input)
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
export function round(input: Tensor): Tensor {
    return unary(functions.RoundFunction, input);
}
/**
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
* Calculates:
* ```js
* output = 1.0 / (1.0 + exp(-input))
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad * output * (1.0 - output)
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function sigmoid(input: Tensor): Tensor {
    return unary(functions.SigmoidFunction, input);
}
/**
* Calculates:
* ```js
* output = sign(input)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad * (input == 0.0 ? 0.0 : 1.0)
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function sign(input: Tensor): Tensor {
    return unary(functions.SignFunction, input);
}
/**
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
* Calculates:
* ```js
* output = input == 0.0 ? 1.0 : sin(input * 3.141592653589793) / (input * 3.141592653589793)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad * (input == 0.0 ? 0.0 : cos(input * 3.141592653589793) / (input * 3.141592653589793) - sin(input * 3.141592653589793) / (input * input * 3.141592653589793))
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function sinc(input: Tensor): Tensor {
    return unary(functions.SincFunction, input);
}
/**
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
* output = input - other
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
export function sub(input: Tensor, other: Tensor, alpha?: number): Tensor {
    return binaryWithAlpha(functions.SubFunction, input, other, alpha);
}
/**
* Alias for `sub`.
*
* Calculates:
* ```js
* output = input - other
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
export function subtract(input: Tensor, other: Tensor, alpha?: number): Tensor {
    return sub(input, other, alpha);
}
/**
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
* Calculates:
* ```js
* output = tanh(input)
* ```
*
* Gradient:
* ```js
* inputGrad = outputGrad * (1.0 - output * output)
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function tanh(input: Tensor): Tensor {
    return unary(functions.TanhFunction, input);
}
/**
* Calculates:
* ```js
* output = trunc(input)
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
export function trunc(input: Tensor): Tensor {
    return unary(functions.TruncFunction, input);
}
/**
* Alias for `trunc`.
*
* Calculates:
* ```js
* output = trunc(input)
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
* inputGrad = input == 0.0 ? 0.0 : outputGrad * log(other)
* ```
*
* @param input the input tensor of any shape
* @param other the other tensor whose shape is broadcastable with the input tensor
* @returns the output tensor
*/
export function xlogy(input: Tensor, other: Tensor): Tensor {
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
export function all(input: Tensor): Tensor {
    return unary(functions.AllFunction, input);
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
export function any(input: Tensor): Tensor {
    return unary(functions.AnyFunction, input);
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
* inputGrad = outputGrad / inputSize
* ```
*
* @param input the input tensor of any shape
* @returns the output tensor
*/
export function mean(input: Tensor): Tensor {
    return unary(functions.MeanFunction, input);
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
export function norm(input: Tensor): Tensor {
    return unary(functions.NormFunction, input);
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
export function prod(input: Tensor): Tensor {
    return unary(functions.ProdFunction, input);
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
export function sum(input: Tensor): Tensor {
    return unary(functions.SumFunction, input);
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
export function countNonzero(input: Tensor): Tensor {
    return unary(functions.CountNonzeroFunction, input);
}
