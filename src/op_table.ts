import { AnOpSpec } from "./op_spec";

export const registry: AnOpSpec[] = [
    // https://pytorch.org/docs/stable/torch.html
    // === Math operations / Pointwise Ops ===
    {
        name: "abs",
        aliases: ["absolute"],
        type: "unary",
        forward: "output = abs(input)",
        backward: "inputGrad = input == 0 ? 0 : (input > 0 ? outputGrad : -outputGrad)",
    },
    {
        name: "acos",
        aliases: ["arccos"],
        type: "unary",
        forward: "output = acos(input)",
        backward: "inputGrad = -outputGrad / sqrt(1 - input * input)",
    },
    {
        name: "acosh",
        aliases: ["arccosh"],
        type: "unary",
        forward: "output = acosh(input)",
        backward: "inputGrad = outputGrad / sqrt(input * input - 1)",
    },
    {
        name: "add",
        type: "binary",
        alpha: true,
        forward: "output = input + other * alpha",
        backward: "inputGrad = outputGrad; otherGrad = outputGrad",
    },
    // addcdiv: trinary
    // addcmul: trinary
    // angle: complex
    {
        name: "asin",
        aliases: ["arcsin"],
        type: "unary",
        forward: "output = asin(input)",
        backward: "inputGrad = outputGrad / sqrt(1 - input * input)",
    },
    {
        name: "asinh",
        aliases: ["arcsinh"],
        type: "unary",
        forward: "output = asinh(input)",
        backward: "inputGrad = outputGrad / sqrt(input * input + 1)",
    },
    {
        name: "atan",
        aliases: ["arctan"],
        type: "unary",
        forward: "output = atan(input)",
        backward: "inputGrad = outputGrad / (input * input + 1)",
    },
    {
        name: "atan2",
        aliases: ["arctan2"],
        type: "binary",
        forward: "output = atan2(input, other)",
        backward: "inputGrad = outputGrad * other / (input * input + other * other); otherGrad = -outputGrad * input / (input * input + other * other)",
    },
    // bitwise_not: integral or boolean type
    // bitwise_and: integral or boolean type
    // bitwise_or: integral or boolean type
    // bitwise_xor: integral or boolean type
    // bitwise_left_shift: integral or boolean type
    // bitwise_right_shift: integral or boolean type
    {
        name: "ceil",
        type: "unary",
        forward: "output = ceil(input)",
        backward: "inputGrad = 0",
    },
    // clamp: beta
    // clip: alias for clamp
    // conj_physical: complex
    {
        name: "copysign",
        type: "binary",
        forward: "output = other >= 0 ? abs(input) : -abs(input)",
        backward: "var dir = other >= 0 ? (input >= 0 ? 1.0 : -1.0) : (input >= 0 ? -1.0 : 1.0); inputGrad = input == 0.0 ? 0.0 : outputGrad * dir; otherGrad = 0",
    },
    {
        name: "cos",
        type: "unary",
        forward: "output = cos(input)",
        backward: "inputGrad = -outputGrad * sin(input)",
    },
    {
        name: "cosh",
        type: "unary",
        forward: "output = cosh(input)",
        backward: "inputGrad = outputGrad * sinh(input)",
    },
    {
        name: "deg2rad",
        type: "unary",
        forward: "output = input * 0.017453292519943295",
        backward: "inputGrad = outputGrad * 0.017453292519943295",
    },
    {
        name: "div",
        aliases: ["divide"],
        type: "binary",
        forward: "output = input / other",
        backward: "inputGrad = outputGrad / other; otherGrad = -outputGrad * input / (other * other)",
    },
    // digamma: special
    // erf: special
    // erfc: special
    // erfinv: special
    {
        name: "exp",
        type: "unary",
        forward: "output = exp(input)",
        backward: "inputGrad = outputGrad * exp(input)",
    },
    {
        name: "exp2",
        type: "unary",
        forward: "output = exp2(input)",
        backward: "inputGrad = outputGrad * exp2(input) * 0.6931471805599453",
    },
    {
        name: "expm1",
        type: "unary",
        forward: "output = exp(input) - 1.0",
        backward: "inputGrad = outputGrad * exp(input)",
    },
    // fake_quantize_per_channel_affine: quantization
    // fake_quantize_per_tensor_affine: quantization
    // float_power: double
    {
        name: "floor",
        type: "unary",
        forward: "output = floor(input)",
        backward: "inputGrad = 0",
    },
    // floor_divide: deprecated
    // fmod: meh
    {
        name: "frac",
        type: "unary",
        forward: "output = input >= 0.0 ? fract(input) : -fract(-input)",
        backward: "inputGrad = outputGrad",
    },
    // frexp: jfc
    // gradient: jfc
    {
        name: "hypot",
        type: "binary",
        forward: "output = sqrt(input * input + other * other)",
        backward: "inputGrad = outputGrad * input / sqrt(input * input + other * other); otherGrad = outputGrad * other / sqrt(input * input + other * other)",
    },
    // i0: special
    // igamma: special
    // igammac: special
    // imag: complex
    {
        name: "ldexp",
        type: "binary",
        forward: "output = input * pow(2.0, other)",
        backward: "var out = pow(2.0, other); inputGrad = outputGrad * out; otherGrad = outputGrad * input * out * 0.6931471805599453",
    },
    // lerp: trinary
    // lgamma: idk
    {
        name: "log",
        type: "unary",
        forward: "output = log(input)",
        backward: "inputGrad = outputGrad / input",
    },
    {
        name: "log10",
        type: "unary",
        forward: "output = log(input) * 0.4342944819032518",
        backward: "inputGrad = outputGrad / (input * 2.302585092994046)",
    },
    {
        name: "log1p",
        type: "unary",
        forward: "output = log(input + 1.0)",
        backward: "inputGrad = outputGrad / (input + 1.0)",
    },
    {
        name: "log2",
        type: "unary",
        forward: "output = log2(input)",
        backward: "inputGrad = outputGrad / (input * 0.6931471805599453)",
    },
    {
        name: "logaddexp",
        type: "binary",
        forward: "output = log(exp(input) + exp(other))",
        backward: "var ein = exp(input); var eoth = exp(other); var addeinv = outputGrad/(ein + eoth); inputGrad = addeinv * ein; otherGrad = addeinv * eoth",
    },
    {
        name: "logaddexp2",
        type: "binary",
        forward: "output = log2(exp2(input) + exp2(other))",
        backward: "var ein = exp2(input); var eoth = exp2(other); var sum_ein_eoth = ein + eoth; inputGrad = outputGrad * (ein / sum_ein_eoth); otherGrad = outputGrad * (eoth / sum_ein_eoth );",
    },
    // logical_and: boolean
    // logical_not: boolean
    // logical_or: boolean
    // logical_xor: boolean
    // logit: yikes
    {
        name: "mul",
        aliases: ["multiply"],
        type: "binary",
        forward: "output = input * other",
        backward: "inputGrad = outputGrad * other; otherGrad = outputGrad * input",
    },
    // mvlgamma: special
    // nan_to_num: special
    {
        name: "neg",
        aliases: ["negative"],
        type: "unary",
        forward: "output = -input",
        backward: "inputGrad = -outputGrad",
    },
    // nextafter: jfc
    // polygamma: special
    {
        name: "positive",
        type: "unary",
        forward: "output = input",
        backward: "inputGrad = outputGrad",
    },
    {
        name: "pow",
        type: "binary",
        // forward: "output = pow(input, other)",
        forward: `output = input >= 0 || fract(other) != 0 ? pow(input, other) :
            pow(-input, other) * ((i32(other) & 1) != 0 ? -1f : 1f)`,
        backward: `inputGrad = input >= 0 || fract(other) != 0 ? outputGrad * other * pow(input, other - 1.0) :
            outputGrad * other * pow(-input, other - 1) * ((i32(other - 1) & 1) != 0 ? -1f : 1f);
        otherGrad = outputGrad * pow(input, other) * log(input)`,
    },
    // quantized_batch_norm: quantization
    // quantized_max_pool1d: quantization
    // quantized_max_pool2d: quantization
    {
        name: "rad2deg",
        type: "unary",
        forward: "output = input * 57.29577951308232",
        backward: "inputGrad = outputGrad * 57.29577951308232",
    },
    // real: complex
    {
        name: "reciprocal",
        type: "unary",
        forward: "output = 1.0 / input",
        backward: "inputGrad = -outputGrad / (input * input)",
    },
    {
        name: "relu",
        nnName: "ReLU",
        nnOp: true,
        type: "unary",
        forward: "output = max(input, 0.0)",
        backward: "inputGrad = input > 0.0 ? outputGrad : 0.0",
    },
    // remainder: meh
    {
        name: "round",
        type: "unary",
        forward: "output = round(input)",
        backward: "inputGrad = 0",
    },
    {
        name: "rsqrt",
        type: "unary",
        forward: "output = 1.0 / sqrt(input)",
        backward: "inputGrad = -outputGrad / (2.0 * sqrt(input) * input)",
    },
    // sgn: complex
    {
        name: "sigmoid",
        nnOp: true,
        type: "unary",
        forward: "output = 1.0 / (1.0 + exp(-input))",
        backward: "var out = 1.0 / (1.0 + exp(-input)); inputGrad = outputGrad * out * (1.0 - out)",
    },
    {
        name: "sign",
        type: "unary",
        forward: "output = sign(input)",
        backward: "inputGrad = 0",
    },
    {
        name: "silu",
        nnName: "SiLU",
        torchName: "torch.nn.functional.silu",
        nnOp: true,
        type: "unary",
        forward: "output = input / (1.0 + exp(-input))",
        backward: "var out = 1.0 / (1.0 + exp(-input)); inputGrad = outputGrad * (out + input * out * (1.0 - out))",
    },
    // signbit: integer
    {
        name: "sin",
        type: "unary",
        forward: "output = sin(input)",
        backward: "inputGrad = outputGrad * cos(input)",
    },
    {
        name: "sinc",
        type: "unary",
        forward: "var inpi = input * 3.141592653589793; output = input == 0.0 ? 1.0 : sin(inpi) / inpi",
        backward: "var inpi = input * 3.141592653589793; inputGrad = input == 0.0 ? 0.0 : (outputGrad * 3.141592653589793 * (inpi*cos(inpi) - sin(inpi)) / (inpi*inpi))",
    },
    {
        name: "sinh",
        type: "unary",
        forward: "output = sinh(input)",
        backward: "inputGrad = outputGrad * cosh(input)",
    },
    // softmax: artisanal
    {
        name: "sqrt",
        type: "unary",
        forward: "output = sqrt(input)",
        backward: "inputGrad = outputGrad / (2.0 * sqrt(input))",
    },
    {
        name: "square",
        type: "unary",
        forward: "output = input * input",
        backward: "inputGrad = outputGrad * 2.0 * input",
    },
    {
        name: "sub",
        aliases: ["subtract"],
        type: "binary",
        alpha: true,
        forward: "output = input - other * alpha",
        backward: "inputGrad = outputGrad; otherGrad = -outputGrad",
    },
    {
        name: "tan",
        type: "unary",
        forward: "output = tan(input)",
        backward: "inputGrad = outputGrad / (cos(input) * cos(input))",
        precision: 5, // fak: I don't know why but tan seems pretty inaccurate
    },
    {
        name: "tanh",
        nnOp: true,
        type: "unary",
        forward: "output = tanh(input)",
        backward: "inputGrad = outputGrad * (1.0 - tanh(input) * tanh(input))",
    },
    // true_divide: meh
    {
        name: "trunc",
        aliases: ["fix"],
        type: "unary",
        forward: "output = trunc(input)",
        backward: "inputGrad = 0",
    },
    {
        name: "xlogy",
        type: "binary",
        forward: "output = input == 0.0 ? 0.0 : input * log(other)",
        backward: "inputGrad = input == 0.0 ? 0.0 : outputGrad * log(other); otherGrad = input == 0.0 ? 0.0 : outputGrad * (input / other);",
    },
    // === Math operations / Reduction Ops ===
    // argmax: indices
    // argmin: indices
    // amax: research
    // amin: research
    // minmax: two outputs
    {
        name: "all",
        type: "reduction",
        combineOp: "&&",
        init: "output = 1",
        forward: "output = output && input",
        backward: "inputGrad = output ? outputGrad : 0.0",
    },
    {
        name: "any",
        type: "reduction",
        combineOp: "||",
        init: "output = 0",
        forward: "output = output || input",
        backward: "inputGrad = output ? outputGrad : 0.0",
    },
    // dist: idk
    // logsumexp: init is -inf
    // max: init is input
    {
        name: "mean",
        type: "reduction",
        combineOp: "+",
        init: "output = 0.0",
        forward: "output = output + input",
        reduce: "output = output / f32(inputSize)",
        backward: "inputGrad = outputGrad / f32(inputSize)",
    },
    // median: artisanal
    // min: init is input
    // mode: indices
    // nanmean: artisanal
    // nanmedian: artisanal
    // nansum: artisanal
    {
        name: "norm",
        type: "reduction",
        combineOp: "+",
        init: "output = 0.0",
        forward: "output = output + input * input",
        reduce: "output = sqrt(output)",
        backward: "inputGrad = outputGrad * input / output",
    },
    {
        name: "prod",
        type: "reduction",
        combineOp: "*",
        init: "output = 1.0",
        forward: "output = output * input",
        backward: "inputGrad = outputGrad * output / input",
    },
    // quantile: wtf
    // std: two step
    // std_mean: two step
    {
        name: "sum",
        type: "reduction",
        combineOp: "+",
        init: "output = 0.0",
        forward: "output = output + input",
        backward: "inputGrad = outputGrad",
    },
    // unique: omg
    // unique_consecutive: omg
    // var: two step
    // var_mean: two step
    {
        name: "countNonzero",
        type: "reduction",
        combineOp: "+",
        init: "output = 0.0",
        forward: "output = output + (input != 0)",
    }
];
