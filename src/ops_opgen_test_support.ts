import * as ops from "./ops_opgen";
import { TensorArrayData } from "./storage";
import { Tensor } from "./tensor";

export type TestArrayData = (number | "NaN" | "+Inf" | "-Inf" | TestArrayData)[];

export function runOpgenTest(kernelName: string, inputs: TensorArrayData[], expectedOutputs: TestArrayData[], expectedGrads: TestArrayData[]|null, gradError: boolean): void {
    const inputTensors = inputs.map(input => new Tensor(input));
    let outputTensors: Tensor[];
    let inputGrads: Tensor[];
    if (kernelName.endsWith("_")) {
        const args = inputTensors.slice(1);
        const input = inputTensors[0];
        // console.log(kernelName, input.shape);
        const op: Function = (input as any)[kernelName];
        op.apply(input, args);
        outputTensors = [input];
    }
    else {
        // outputTensors = (ops as any)[kernelName](inputTensors);
    }
}
