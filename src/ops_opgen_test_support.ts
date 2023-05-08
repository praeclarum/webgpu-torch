import * as torch from "./ops_opgen";
import { tensor } from "./ops_artisanal";

import { TensorArrayData } from "./storage";
import { Tensor } from "./tensor";
import { DeviceType } from "./device";

type TestArrayData = (number | "NaN" | "+Inf" | "-Inf" | TestArrayData)[];

function compareArrays(x: TensorArrayData, expected: TestArrayData, deviceType: DeviceType): void {
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
                if (deviceType == "cpu") {
                    expect(actual).toBeNaN();
                }
                else {
                    if (Number.isNaN(actual)) {
                        // OK
                    }
                    else {
                        expect(actual).not.toBeNaN();
                        expect(actual).toEqual(0);
                    }
                }
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
            compareArrays(actual, expectedValue, deviceType);
        }
        else {
            throw new Error(`Unexpected value: ${actual}`);
        }
    }
}

async function runOpgenTest(kernelName: string, inputs: TensorArrayData[], expectedOutputs: TestArrayData[], expectedGrads: TestArrayData[], gradError: boolean, runBackward: boolean): Promise<void> {
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
        if (runBackward) {
            outputTensor.backward();
            inputGrads = inputTensors.map(input => input.grad!);
        }
        else {
            inputGrads = [];
        }
    }
    const expectedOutput = expectedOutputs[0];
    if (runBackward) {
        expect(inputGrads).toHaveLength(expectedGrads.length);
        for (let i = 0; i < inputGrads.length; i++) {
            const inputGrad = inputGrads[i];
            const expectedGrad = expectedGrads[i];
            compareArrays(await inputGrad.toArrayAsync(), expectedGrad, inputGrad.device.type);
        }
    }
    else {
        compareArrays(await outputTensor.toArrayAsync(), expectedOutput, outputTensor.device.type);
    }
}

export async function runOpgenTestForward(kernelName: string, inputs: TensorArrayData[], expectedOutputs: TestArrayData[]): Promise<void> {
    await runOpgenTest(kernelName, inputs, expectedOutputs, [], false, false);
}

export async function runOpgenTestBackward(kernelName: string, inputs: TensorArrayData[], expectedGrads: TestArrayData[], gradError: boolean): Promise<void> {
    await runOpgenTest(kernelName, inputs, [], expectedGrads, gradError, true);
}
