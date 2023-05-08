import * as torch from "./ops_opgen";
import { tensor } from "./ops_artisanal";

import { TensorArrayData } from "./storage";
import { Tensor } from "./tensor";

type TestArrayData = (number | "NaN" | "+Inf" | "-Inf" | TestArrayData)[];

function compareArrays(x: TensorArrayData, expected: TestArrayData): void {
    expect(x.length).toEqual(expected.length);
    for (let i = 0; i < x.length; i++) {
        const actual = x[i];
        const expectedValue = expected[i];
        if (typeof expectedValue == "number") {
            expect(typeof actual).toEqual("number");
            expect(actual).not.toBeNaN();            
            expect(actual).toBeCloseTo(expectedValue, 6);
        }
        else if (typeof expectedValue == "string") {
            if (expectedValue == "NaN") {
                expect(actual).toBeNaN();
            }
            else if (expectedValue == "+Inf") {
                expect(actual).toBe(Infinity);
            }
            else if (expectedValue == "-Inf") {
                expect(actual).toBe(-Infinity);
            }
            else {
                throw new Error(`Unexpected string value: ${expectedValue}`);
            }
        }
        else if (actual instanceof Array) {
            compareArrays(actual, expectedValue);
        }
        else {
            throw new Error(`Unexpected value: ${actual}`);
        }
    }
}

export async function runOpgenTest(kernelName: string, inputs: TensorArrayData[], expectedOutputs: TestArrayData[], expectedGrads: TestArrayData[], gradError: boolean): Promise<void> {
    let outputTensor: Tensor;
    let inputGrads: Tensor[];
    if (kernelName.endsWith("_")) {
        const inputTensors = inputs.map(input => tensor(input));
        const input = inputTensors[0];
        const op: Function = (input as any)[kernelName];
        op.apply(input, inputTensors.slice(1));
        outputTensor = input;
        inputGrads = [];
    }
    else {
        const inputTensors = inputs.map(input => tensor({data: input, requiresGrad: true}));
        const op: Function = (torch as any)[kernelName];
        outputTensor = op.apply(null, inputTensors);
        outputTensor.backward();
        inputGrads = inputTensors.map(input => input.grad!);
    }
    const expectedOutput = expectedOutputs[0];
    compareArrays(await outputTensor.toArrayAsync(), expectedOutput);
    expect(inputGrads).toHaveLength(expectedGrads.length);
    for (let i = 0; i < inputGrads.length; i++) {
        const inputGrad = inputGrads[i];
        const expectedGrad = expectedGrads[i];
        compareArrays(await inputGrad.toArrayAsync(), expectedGrad);
    }
}
