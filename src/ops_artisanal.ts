import { Deviceish } from "./device";
import { Dtype } from "./dtype";
import { Tensor, TensorData, TensorJsonData } from "./tensor";
import { shouldCreateGradient } from "./autograd";

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
            resultRows: input.shape[0],
            resultCols: other.shape[1],
            innerDim: input.shape[1],
            alpha: 1.0,
        };
        return input.runKernel(
            "mm",
            { resultDtype: input.dtype },
            params,
            [[params.resultRows, params.resultCols]],
            other
        )[0];
    }
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


export function tensor(spec: TensorJsonData): Tensor;
export function tensor(
    array: TensorData,
    dtype?: Dtype,
    device?: Deviceish,
    requiresGrad?: boolean
): Tensor;
export function tensor(
    arrayOrSpec: TensorData | TensorJsonData,
    dtype?: Dtype,
    device?: Deviceish,
    requiresGrad?: boolean
): Tensor {
    return new Tensor(arrayOrSpec, dtype, device, requiresGrad);
}
