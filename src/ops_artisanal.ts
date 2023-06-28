import { shouldCreateGradient } from "./autograd";
import { Tensor } from "./tensor";
import type { Deviceish } from "./device";
import type { Dtype } from "./dtype";
import { broadcastBatchedMatmul, contiguousStridedShape, reshapeBatchedMatmul, type Shape, type StridedShape, type Strides } from "./shape";
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

/** Gathers values along an axis specified by dim. */
export function gather(input: Tensor, dim: number, index: Tensor): Tensor {
    if (shouldCreateGradient(input, index)) {
        return GatherFunction.apply(input, dim, index);
    }
    return GatherFunction.forward([input, dim, index]);
}

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

export function reshape(input: Tensor, shape: number[]): Tensor {
    throw new Error("reshape not supported yet");
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
