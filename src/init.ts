import { noGrad } from "./autograd";
import { t } from "./ops_artisanal";
import { Tensor } from "./tensor";

export type LinearFunction =
    | "linear"
    | "conv1d"
    | "conv2d"
    | "conv3d"
    | "convTranspose2d"
    | "convTranspose3d";
export type NonlinearFunction =
    | "sigmoid"
    | "tanh"
    | "relu"
    | "leakyRelu"
    | "selu";

export function calculateFanInAndFanOut(tensor: Tensor): [number, number] {
    const dimensions = tensor.ndim;
    if (dimensions < 2) {
        throw new Error(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        );
    }
    const numInputFmaps = tensor.shape[1];
    const numOutputFmaps = tensor.shape[0];
    const receptiveFieldSize =
        dimensions > 2 ? tensor.shape.slice(2).reduce((a, b) => a * b, 1) : 1;
    const fanIn = numInputFmaps * receptiveFieldSize;
    const fanOut = numOutputFmaps * receptiveFieldSize;
    return [fanIn, fanOut];
}

const linearFns: { [key: string]: null } = {
    linear: null,
    conv1d: null,
    conv2d: null,
    conv3d: null,
    conv_transpose1d: null,
    conv_transpose2d: null,
    conv_transpose3d: null,
};

function calculateGain(
    nonlinearity: LinearFunction | NonlinearFunction,
    param?: number
): number {
    if (nonlinearity in linearFns || nonlinearity === "sigmoid") {
        return 1;
    }
    else if (nonlinearity === "tanh") {
        return 5.0 / 3.0;
    }
    else if (nonlinearity === "relu") {
        return Math.sqrt(2.0);
    }
    else if (nonlinearity === "leakyRelu") {
        const negativeSlope = param || 0.01;
        return Math.sqrt(2.0 / (1 + Math.pow(negativeSlope, 2)));
    }
    else if (nonlinearity === "selu") {
        return 3.0 / 4.0; // Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    }
    else {
        throw new Error(
            `Unsupported nonlinearity ${nonlinearity}`
        );
    }
}

/** Fills the input `Tensor` with values according to the method
 *   described in `Delving deep into rectifiers: Surpassing human-level
 *   performance on ImageNet classification` - He, K. et al. (2015), using a
 *   uniform distribution.
 *
 * The resulting tensor will have values sampled from
 *   `U(-bound, bound)` where
 * `bound = gain * sqrt(3/fanMode)`
 *
 * Also known as He initialization.
 * 
 * This is the default initializer for `Conv` and `Linear` layers.
 * 
 * @param tensor n-dimensional
 * @param a the negative slope of the rectifier used after this layer (only used with `'leakyRelu'`)
 * @param mode either `'fanIn'` (default) or `'fanOut'`. Choosing `'fanIn'`
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing `'fanOut'` preserves the magnitudes in the
            backwards pass.
 * @param nonlinearity the non-linear function, 
            recommended to use only with `'relu'` or `'leakyRelu'` (default).
 */
export function kaimingUniform_(
    tensor: Tensor,
    a: number = 0,
    mode: "fanIn" | "fanOut" = "fanIn",
    nonlinearity: LinearFunction | NonlinearFunction = "leakyRelu"
): Tensor {
    if (tensor.shape.includes(0)) {
        return tensor;
    }
    const [fanIn, fanOut] = calculateFanInAndFanOut(tensor);
    const fan = mode === "fanIn" ? fanIn : fanOut;
    const gain = calculateGain(nonlinearity, a);
    const std = gain / Math.sqrt(fan);
    const bound = Math.sqrt(3.0) * std;
    noGrad(() => {
        tensor.uniform_(-bound, bound);
    });
    return tensor;
}

export function uniform_(
    tensor: Tensor,
    lowerBound: number,
    upperBound: number
): Tensor {
    noGrad(() => {
        tensor.uniform_(lowerBound, upperBound);
    });
    return tensor;
}
