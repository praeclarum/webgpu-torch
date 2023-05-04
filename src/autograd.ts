import { Tensor } from "./tensor";

export type FunctionInput = Tensor | number | boolean | string;
export type GradientFunctionOutput = Tensor | null;

export class GradientFunctionContext {
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
    ctx: GradientFunctionContext,
    output: Tensor
) => (Tensor | null)[];

export class AutoFunction {
    static forward(...inputs: FunctionInput[]): Tensor {
        throw new Error("Do not call forward on AutoFunction directly.");
    }
    static setupContext(
        ctx: GradientFunctionContext,
        inputs: FunctionInput[],
        output: Tensor
    ) {
        throw new Error("Do not call setupContext on AutoFunction directly.");
    }
    static backward(
        ctx: GradientFunctionContext,
        gradOutput: Tensor
    ): GradientFunctionOutput[] {
        throw new Error("Do not call backward on AutoFunction directly.");
    }
    static apply(...inputs: FunctionInput[]): Tensor {
        const ctx = new GradientFunctionContext(inputs);
        const detachedInputs = inputs.map((input) =>
            input instanceof Tensor ? input.detach() : input
        );
        const output = this.forward(...detachedInputs);
        this.setupContext(ctx, detachedInputs, output);
        output.setGradientFunction(ctx, this.backward);
        return output;
    }
}
