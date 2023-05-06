import { Tensor } from "./tensor";

export type FunctionInput = Tensor | number | boolean | string | undefined;
export type GradientFunctionOutput = Tensor | null;

export class GradientContext {
    [key: string]: any;
    needsInputGradient: boolean[];
    inputsWithGradient: (Tensor | null)[];
    savedTensors: Tensor[] = [];
    constructor(inputs: FunctionInput[]) {
        this.needsInputGradient = inputs.map(
            (input) => input instanceof Tensor && input.requiresGrad
        );
        this.inputsWithGradient = inputs.map((input) =>
            input instanceof Tensor && input.requiresGrad ? input : null
        );
    }
    saveForBackward(...tensors: Tensor[]) {
        this.savedTensors = tensors;
    }
}

export type GradientFunction = (
    ctx: GradientContext,
    output: Tensor
) => (Tensor | null)[];

export interface IAutoFunction {
    forward(inputs: FunctionInput[]): Tensor;
    apply(...inputs: FunctionInput[]): Tensor;
    backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[];
}

export class AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        throw new Error("Do not call forward on AutoFunction directly.");
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        throw new Error("Do not call setupContext on AutoFunction directly.");
    }
    static backward(
        ctx: GradientContext,
        outputGrad: Tensor
    ): GradientFunctionOutput[] {
        throw new Error("Do not call backward on AutoFunction directly.");
    }
    static apply(...inputs: FunctionInput[]): Tensor {
        const ctx = new GradientContext(inputs);
        const detachedInputs = inputs.map((input) =>
            input instanceof Tensor ? input.detach() : input
        );
        const output = this.forward(detachedInputs);
        this.setupContext(ctx, detachedInputs, output);
        output.setGradientFunction(ctx, this.backward);
        return output;
    }
}

export function shouldCreateGradient(...inputs: Tensor[]): boolean {
    for (const input of inputs) {
        if (input.requiresGrad) {
            return true;
        }
    }
    return false;
}
