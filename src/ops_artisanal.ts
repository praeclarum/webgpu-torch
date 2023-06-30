import { shouldCreateGradient } from "./autograd";
import { Tensor } from "./tensor";
import type { Deviceish } from "./device";
import type { Dtype } from "./dtype";
import {
    broadcastBatchedMatmul,
    contiguousStridedShape,
    defaultStrides,
    reshapeBatchedMatmul,
    shapesAreEqual,
    type Shape,
    type StridedShape,
    type Strides,
    shapeSize,
    check,
    validateIdx,
    validateDimLength,
    canonicalizeDim,
} from "./shape";
import type { TensorData, TensorSpec, MemoryFormat } from "./tensor";
import { KernelParamsInput } from "./kernel";
import { GatherFunction, LinearFunction } from "./functions_artisanal";

export function cat(inputs: Tensor[], dim: number): Tensor {
    throw new Error("cat not implemented yet");
}

export function clone(
    input: Tensor,
    memoryFormat: MemoryFormat = "preserveFormat"
): Tensor {
    if (shouldCreateGradient(input)) {
        throw new Error("clone gradient not supported yet");
        // return CloneFunction.apply(input);
    } else {
        const newStorage = input.storage.clone();
        return new Tensor({
            data: newStorage,
            shape: input.shape,
            dtype: input.dtype,
            requiresGrad: input.requiresGrad,
        });
    }
}

/**
 * Applies a 2D convolution over an input image composed of several input planes.
 *
 * #### Forward
 * ```
 * output[y, x] = sum(Ky, sum(Kx, input[y + ky, x + kx] * weight[ky, kx])) + bias
 * ```
 *
 * @param input input tensor of shape [B, inChannels, iH, iW]
 * @param weight filters of shape [outChannels, inChannels, kH, kW]
 * @param bias optional bias tensor of shape [outChannels]
 * @param stride the stride of the convolving kernel. Can be a single number or a tuple (sH, sW). Default: 1
 * @param padding implicit padding on both sides of the kernel. Can be a single number or a tuple (padH, padW). Default: 0
 *     `padding="valid"` is the same as no padding. `padding="same"` pads the input so the output has the shape as the input.
 *     However, this mode can only be used when `stride` is 1.
 * @returns
 */
export function conv2d(
    input: Tensor,
    weight: Tensor,
    bias?: Tensor,
    stride?: number | [number, number],
    padding?: number | [number, number] | "valid" | "same"
): Tensor {
    if (shouldCreateGradient(input, weight)) {
        throw new Error("conv2d gradient not supported yet");
    } else {
        if (input.shape.length !== 4 || weight.shape.length !== 4) {
            throw new Error(
                `Expected image tensor, got ${input.shape} and kernel ${weight.shape}`
            );
        }
        if (input.shape[1] !== weight.shape[1]) {
            throw new Error(
                `Expected number of chennels in input image to match number of channels in kernel, got ${input.shape} and ${weight.shape}`
            );
        }
        const params = {
            batchSize: input.shape[0],
            inputChannels: input.shape[1],
            outputChannels: weight.shape[0],
            inputHeight: input.shape[2],
            inputWidth: input.shape[3],
            kernelHeight: weight.shape[2],
            kernelWidth: weight.shape[3],
            outputHeight: input.shape[2] - weight.shape[2] + 1,
            outputWidth: input.shape[3] - weight.shape[3] + 1,
        };
        return input.runKernel(
            "conv2d",
            { dtype: input.dtype },
            params,
            [
                [
                    params.batchSize,
                    params.outputChannels,
                    params.outputHeight,
                    params.outputWidth,
                ],
            ],
            weight
        )[0];
    }
}

function collapseView(a: Tensor, start: number, end: number): Tensor {
    const newShape = collapseViewHelper(a, start, end);
    if (newShape === null) {
        throw new Error("Attempting to view a collapsed tensor, but no such view exists!");
    }
    return a.withShape(newShape.shape, newShape.strides);
}

function collapse(a: Tensor, start: number, end: number): Tensor {
    // const newShape = collapsedShape(a, start, end);
    throw new Error("collapse not implemented yet");
}

/** Flattens a contiguous range of dims into a 1D tensor.
 *  
 * `flatten`, unlike other shape operators, returns the input tensor on a no-op
 * (unless a 0D tensor is flattened, in which case it's returned in 1D).
 * */
export function flatten(
    a: Tensor,
    startDim: number = 0,
    endDim: number = -1
): Tensor {
    startDim = canonicalizeDim(a.ndim, startDim);
    endDim = canonicalizeDim(a.ndim, endDim);
    // Short-circuits on no-op
    if (startDim == endDim && a.ndim != 0) {
        return a;
    }
    // Tries to take a view
    const newShape = collapseViewHelper(a, startDim, endDim);
    if (newShape !== null) {
        return collapseView(a, startDim, endDim);
    }
    return collapse(a, startDim, endDim);
}

/** Gathers values along an axis specified by dim. */
export function gather(input: Tensor, dim: number, index: Tensor): Tensor {
    if (shouldCreateGradient(input, index)) {
        return GatherFunction.apply(input, dim, index);
    }
    return GatherFunction.forward([input, dim, index]);
}

/**
 * Applies a linear transformation to the incoming data: `y = xA^T + b`.
 * @param input `(*, inFeatures)` where `*` means any number of additional dimensions, including none
 * @param weight `(outFeatures, inFeatures)` or `(inFeatures)`
 * @param bias `(outFeatures)` or `()`
 * @returns `(*, outFeatures)` or `(*)`, based on the shape of the weight
 */
export function linear(input: Tensor, weight: Tensor, bias?: Tensor): Tensor {
    if (shouldCreateGradient(input, weight, bias)) {
        return LinearFunction.apply(input, weight, bias);
    }
    return LinearFunction.forward([input, weight, bias]);
}

export function matmul(input: Tensor, other: Tensor): Tensor {
    const a: StridedShape = { shape: input.shape, strides: input.strides };
    const b: StridedShape = { shape: other.shape, strides: other.strides };
    const adims = a.shape.length;
    const bdims = b.shape.length;
    if (adims === 0 || bdims === 0) {
        throw new Error("matmul requires at least 1D tensors");
    }
    let atensor = input;
    let btensor = other;
    let op: string;
    let aop: StridedShape;
    let bop: StridedShape;
    let outputShape: Shape;
    // If both tensors are 1-dimensional, the dot product (scalar) is returned
    if (adims === 1 && bdims === 1) {
        if (a.shape[0] !== b.shape[0]) {
            throw new Error(
                `inconsistent tensor size, expected tensor [${a.shape}] and src [${b.shape}] to have the same number of elements, but got ${a.shape[0]} and ${b.shape[0]} elements respectively`
            );
        }
        op = "dot";
        aop = a;
        bop = b;
        outputShape = [];
    }
    // If both arguments are 2-dimensional, the matrix-matrix product is returned
    else if (adims === 2 && bdims === 2) {
        op = "mm";
        aop = a;
        bop = b;
        outputShape = [a.shape[0], b.shape[1]];
        if (aop.shape[1] !== b.shape[0]) {
            throw new Error(
                `mat1 and mat2 shapes cannot be multiplied (${a.shape[0]}x${a.shape[1]} and ${b.shape[0]}x${b.shape[1]})`
            );
        }
    }
    // If the first argument is 1-dimensional and the second argument is 2-dimensional, a 1 is prepended to its dimension
    else if (adims === 1 && bdims === 2) {
        const aopshape = b.shape.slice();
        const aopstrides = b.strides.slice();
        aopshape[bdims - 1] = b.shape[bdims - 2];
        aopshape[bdims - 2] = b.shape[bdims - 1];
        aopstrides[bdims - 1] = b.strides[bdims - 2];
        aopstrides[bdims - 2] = b.strides[bdims - 1];
        op = "mv";
        aop = { shape: aopshape, strides: aopstrides };
        bop = a;
        outputShape = [b.shape[bdims - 1]];
        atensor = other;
        btensor = input;
        if (aop.shape[1] !== bop.shape[0]) {
            throw new Error(
                `mat1 and mat2 shapes cannot be multiplied (1x${bop.shape[0]} and ${aop.shape[1]}x${aop.shape[0]})`
            );
        }
    }
    // If the first argument is 2-dimensional and the second argument is 1-dimensional, the matrix-vector product is returned
    else if (adims === 2 && bdims == 1) {
        op = "mv";
        aop = a;
        bop = b;
        outputShape = [a.shape[0]];
        if (aop.shape[1] !== bop.shape[0]) {
            throw new Error(
                `size mismatch, got ${aop.shape[0]}, ${aop.shape[0]}x${aop.shape[1]},${b.shape[0]}`
            );
        }
    } else if (adims >= 1 && bdims >= 1 && (adims > 2 || bdims > 2)) {
        op = "bmm";
        const broadcast = broadcastBatchedMatmul(input, other);
        aop = reshapeBatchedMatmul(broadcast.a);
        bop = reshapeBatchedMatmul(broadcast.b);
        outputShape = broadcast.output.shape;
        if (aop.shape[2] !== bop.shape[1]) {
            throw new Error(
                `mat1 and mat2 shapes cannot be multiplied (${aop.shape[1]}x${aop.shape[2]} and ${bop.shape[1]}x${bop.shape[2]})`
            );
        }
    } else {
        throw new Error(
            `matmul not supported for ${adims}D and ${bdims}D tensors`
        );
    }
    let params: KernelParamsInput = {};
    if (op === "bmm") {
        params = {
            batchSize: Math.max(aop.shape[0], bop.shape[0]),
            aRows: aop.shape[1],
            aCols: aop.shape[2],
            bCols: bop.shape[2],
            aBatchStride: aop.strides[0],
            aRowStride: aop.strides[1],
            aColStride: aop.strides[2],
            bBatchStride: bop.strides[0],
            bRowStride: bop.strides[1],
            bColStride: bop.strides[2],
            alpha: 1.0,
        };
    } else if (op === "mm") {
        params = {
            aRows: aop.shape[0],
            aCols: aop.shape[1],
            bCols: bop.shape[1],
            aRowStride: aop.strides[0],
            aColStride: aop.strides[1],
            bRowStride: bop.strides[0],
            bColStride: bop.strides[1],
            alpha: 1.0,
        };
    } else if (op === "mv") {
        params = {
            aRows: aop.shape[0],
            aCols: aop.shape[1],
            aRowStride: aop.strides[0],
            aColStride: aop.strides[1],
            bRowStride: bop.strides[0],
        };
    } else if (op === "dot") {
        params = {
            aRows: aop.shape[0],
            aRowStride: aop.strides[0],
            bRowStride: bop.strides[0],
        };
    }
    return atensor.runKernel(
        op,
        { resultDtype: input.dtype },
        params,
        [outputShape],
        btensor
    )[0];
}

export function mm(input: Tensor, other: Tensor): Tensor {
    if (shouldCreateGradient(input, other)) {
        throw new Error("mm gradient not supported yet");
    } else {
        if (input.shape.length !== 2 || other.shape.length !== 2) {
            throw new Error(
                `Expected 2D tensors, got ${input.shape} and ${other.shape}`
            );
        }
        if (input.shape[1] !== other.shape[0]) {
            throw new Error(
                `Expected tensors inner dimensions to be compatible, got ${input.shape} and ${other.shape}`
            );
        }
        const params = {
            aRows: input.shape[0],
            aCols: input.shape[1],
            bCols: other.shape[1],
            aRowStride: input.strides[0],
            aColStride: input.strides[1],
            bRowStride: other.strides[0],
            bColStride: other.strides[1],
            alpha: 1.0,
        };
        return input.runKernel(
            "mm",
            { resultDtype: input.dtype },
            params,
            [[params.aRows, params.bCols]],
            other
        )[0];
    }
}

export function numel(input: Tensor): number {
    return shapeSize(input.shape);
}

function inferSize(shape: Shape, numel: number): Shape {
    let dim = null;
    let newsize = 1;
    for (let i = 0; i < shape.length; i++) {
        let d = shape[i];
        if (d == -1) {
            check(dim === null, () => "only one dimension can be inferred");
            dim = i;
        } else if (d >= 0) {
            newsize *= d;
        } else {
            check(false, () => `invalid shape dimension ${d}`);
        }
    }
    check(
        numel == newsize ||
            (dim !== null && newsize > 0 && numel % newsize == 0),
        () => `shape '[${shape}]' is invalid for input of size ${numel}`
    );
    if (dim !== null) {
        check(
            newsize != 0,
            () => `cannot reshape tensor of 0 elements into shape ${shape} because the unspecified dimension size -1 can be any 
value and is ambiguous`
        );
        shape[dim] = Math.floor(numel / newsize);
    }
    return shape;
}

function splitDim(a: Tensor, dim: number, outerLength: number): Tensor {
    validateIdx(a.ndim, dim);
    validateDimLength(outerLength);
    const innerLength = Math.floor(a.shape[dim] / outerLength);
    if (a.shape[dim] % outerLength !== 0) {
        throw new Error(
            `Attempting to split dimension of length ${a.shape[dim]}, but out length of ${outerLength} divides it with a remainder!`
        );
    }
    const newShape: Shape = [];
    const newStrides: Strides = [];
    for (let idx = 0; idx < a.ndim; idx++) {
        if (idx === dim) {
            newShape.push(outerLength);
            newShape.push(innerLength);
            newStrides.push(a.strides[idx] * innerLength);
            newStrides.push(a.strides[idx]);
        } else {
            newShape.push(a.shape[idx]);
            newStrides.push(a.strides[idx]);
        }
    }
    return a.withShape(newShape, newStrides);
}
function validateCollapseArgs(a: Tensor, start: number, end: number): void {
    const ndim = Math.max(1, a.ndim);
    validateIdx(ndim, start);
    validateIdx(ndim, end);
    check(
        end >= start,
        () =>
            `Attempting to collapse but end, ${end}, is less than start, ${start}`
    );
}
function collapseViewHelper(
    a: Tensor,
    start: number,
    end: number
): StridedShape | null {
    validateCollapseArgs(a, start, end);
    let shape: Shape;
    let strides: Strides;
    if (a.ndim === 0) {
        shape = [1];
        strides = [1];
    } else {
        shape = a.shape;
        strides = a.strides;
    }

    if (a.ndim === 0 || start === end) {
        return { shape, strides };
    }
    let length = shape[end];
    let stride = strides[end];
    for (let idx = end - 1; idx >= start; idx--) {
        if (shape[idx] === 0 || shape[idx + 1] === 0) {
            length = 0;
            stride = 0;
            break;
        }
        if (shape[idx] === 1) {
            continue;
        }
        length *= shape[idx];
        stride = Math.min(stride, strides[idx]);
        if (
            a.numel() > 0 &&
            shape[idx + 1] != 1 &&
            !(strides[idx] === strides[idx + 1] * shape[idx + 1])
        ) {
            return null;
        }
    }
    const newShape = shape
        .slice(0, start)
        .concat([length])
        .concat(shape.slice(end + 1));
    let newStrides = strides
        .slice(0, start)
        .concat([stride])
        .concat(strides.slice(end + 1));
    if (a.numel() === 0) {
        newStrides = defaultStrides(newShape);
    }
    return { shape: newShape, strides: newStrides };
}

function primitiveReshape(a: Tensor, shape: Shape): Tensor {
    throw new Error("Copying reshape not implemented");
}

function reshapeViewHelper(
    a: Tensor,
    shapeInput: Shape,
    allowCopy: boolean = false
): Tensor {
    const shape = inferSize(shapeInput, a.numel());

    // Short-circuits if shape is the same
    if (shapesAreEqual(a.shape, shape)) {
        return a.withShape(a.shape, a.strides);
    }

    // Special-cases tensors with no elements
    if (a.numel() === 0) {
        return a.withShape(shape, defaultStrides(shape));
    }

    // Special-cases reshaping zero dim tensors
    if (a.ndim === 0) {
        let _a = a;
        for (let length of shape) {
            if (length !== 1) {
                throw new Error("Expected length to be 1.");
            }
            _a = squeeze(_a, -1);
        }
        return _a;
    }

    // Special-cases reshaping to zero dim tensors
    if (shape.length === 0) {
        let _a = a;
        for (let length of a.shape) {
            if (length !== 1) {
                throw new Error("Expected length to be 1.");
            }
            _a = squeeze(_a, -1);
        }
        return _a;
    }

    // Handles general case: a 1+D tensor reshaped into a distinct 1+D shape
    let idx = 0;
    let a_ = a;
    for (let length of shape) {
        // Handles tail unsqueezes
        if (idx >= a_.ndim) {
            if (length !== 1) {
                throw new Error("Expected length to be 1.");
            }
            let lastDim = a_.ndim - 1;
            a_ = splitDim(a_, lastDim, a_.shape[lastDim]);
            idx++;
            continue;
        }

        // Skips dimensions that are already the correct length
        if (length === a_.shape[idx]) {
            idx++;
            continue;
        }
        // Gathers enough original dimensions such that this new dimension can be created
        // Note that this accumulation will terminate because we've verified a and the shape
        // specify the same number of elements above
        let accum = a_.shape[idx];
        let end = idx;
        while (accum % length !== 0) {
            end++;
            accum *= a_.shape[end];
        }
        if (end !== idx) {
            let newShapeStrides = collapseViewHelper(a_, idx, end);
            if (newShapeStrides === null) {
                if (allowCopy) {
                    return primitiveReshape(a, shape);
                }
                throw new Error(
                    `Cannot view a tensor with shape ${a.shape} and strides ${a.strides} as a tensor with shape ${shape}!`
                );
            }
            a_ = flatten(a_, idx, end);
        }
        if (accum !== length) {
            a_ = splitDim(a_, idx, length);
        }
        idx++;
    }
    while (idx < a_.ndim) {
        if (a_.shape[idx] !== 1) {
            throw new Error("Expected shape at index " + idx + " to be 1.");
        }
        a_ = squeeze(a_, idx);
    }
    return a_;
}

export function reshape(input: Tensor, shape: number[]): Tensor {
    return reshapeViewHelper(input, shape, true);
}

export function squeeze(input: Tensor, dim?: number | number[]): Tensor {
    let dims: number[];
    if (dim === undefined) {
        dims = [];
        for (let i = 0; i < input.shape.length; i++) {
            if (input.shape[i] === 1) {
                dims.push(i);
            }
        }
    } else if (typeof dim === "number") {
        dims = [dim];
    } else {
        dims = dim;
    }
    const inputRank = input.shape.length;
    const minDim = inputRank > 0 ? -inputRank : -1;
    const maxDim = inputRank > 0 ? inputRank - 1 : 0;
    for (let i in dims) {
        let d = dims[i];
        if (d < minDim || d > maxDim) {
            throw new Error(
                `Dimension out of range (expected to be in range of [${minDim}, ${maxDim}], but got ${d})`
            );
        }
        if (d < 0) {
            dims[i] = input.shape.length + d;
        }
    }
    dims.sort();
    const outputShape: Shape = [];
    const outputStrides: number[] = [];
    let j = 0;
    for (let i = 0; i < inputRank; i++) {
        if (j < dims.length && i === dims[j]) {
            if (input.shape[i] !== 1) {
                outputShape.push(input.shape[i]);
                outputStrides.push(input.strides[i]);
            }
            j++;
        } else {
            outputShape.push(input.shape[i]);
            outputStrides.push(input.strides[i]);
        }
    }
    return input.withShape(outputShape, outputStrides);
}

export function t(input: Tensor): Tensor {
    if (input.shape.length !== 2) {
        throw new Error(`Expected 2D tensor, got ${input.shape}`);
    }
    if (shouldCreateGradient(input)) {
        throw new Error("t gradient not supported yet");
        // return TransposeFunction.apply(input, 0, 1);
    } else {
        let newShape = input.shape.slice();
        newShape.reverse();
        let newStrides = input.strides.slice();
        newStrides.reverse();
        return input.withShape(newShape, newStrides);
    }
}

export function tensor(spec: TensorSpec): Tensor;
export function tensor(
    array: TensorData,
    dtype?: Dtype,
    device?: Deviceish,
    requiresGrad?: boolean
): Tensor;
export function tensor(
    arrayOrSpec: TensorData | TensorSpec,
    dtype?: Dtype,
    device?: Deviceish,
    requiresGrad?: boolean
): Tensor {
    if (arrayOrSpec.hasOwnProperty("data")) {
        return new Tensor(arrayOrSpec as TensorSpec);
    }
    return new Tensor(arrayOrSpec as TensorData, dtype, device, requiresGrad);
}

export function unsqueeze(input: Tensor, dim?: number): Tensor {
    const inputRank = input.shape.length;
    const minDim = inputRank > 0 ? -inputRank - 1 : -2;
    const maxDim = inputRank > 0 ? inputRank + 1 : 2;
    let unsqueezeOutputDim: number;
    if (dim === undefined) {
        unsqueezeOutputDim = 0;
    } else if (dim < minDim || dim >= maxDim) {
        throw new Error(
            `Dimension out of range (expected to be in range of [${minDim}, ${
                maxDim - 1
            }], but got ${dim})`
        );
    } else if (dim < 0) {
        unsqueezeOutputDim = dim + inputRank + 1;
    } else {
        unsqueezeOutputDim = dim;
    }
    const outputShape: Shape = [];
    const outputStrides: number[] = [];
    let inputDim = 0;
    for (let outputDim = 0; outputDim < inputRank + 1; outputDim++) {
        if (outputDim === unsqueezeOutputDim) {
            outputShape.push(1);
            if (outputDim === 0) {
                outputStrides.push(input.strides[0] * input.shape[0]);
            } else {
                outputStrides.push(outputStrides[outputDim - 1]);
            }
        } else {
            outputShape.push(input.shape[inputDim]);
            outputStrides.push(input.strides[inputDim]);
            inputDim++;
        }
    }
    return input.withShape(outputShape, outputStrides);
}

export function view(input: Tensor, shape: number[]): Tensor {
    return reshapeViewHelper(input, shape, false);
}
