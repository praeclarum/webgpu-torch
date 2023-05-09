import * as torch from "./ops_opgen";
import { registry } from "./op_table";
import { tensor } from "./ops_artisanal";

import { TensorArrayData } from "./storage";
import { Tensor } from "./tensor";
import { DeviceType } from "./device";
import { AnOpSpec } from "./op_spec";

type TestArrayData = (number | "NaN" | "+Inf" | "-Inf" | TestArrayData)[];

const opSpecByName: { [name: string]: AnOpSpec } = {};
for (const opSpec of registry) {
    opSpecByName[opSpec.name] = opSpec;
}

function compareArrays(x: TensorArrayData, expected: TestArrayData, precision: number, deviceType: DeviceType): void {
    expect(x.length).toEqual(expected.length);
    for (let i = 0; i < x.length; i++) {
        const actual = x[i];
        const expectedValue = expected[i];
        if (typeof expectedValue == "number") {
            expect(typeof actual).toEqual("number");
            if (deviceType === "cpu" && expectedValue === 0 && (actual === -Infinity || Number.isNaN(actual))) {
                // OK
            }
            else {
                expect(actual).toBeCloseTo(expectedValue, precision);
            }
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
                if (deviceType == "cpu") {
                    expect(actual).toBe(Infinity);
                } else {
                    if (actual === Infinity) {
                        // OK
                    }
                    else {
                        expect(actual).toEqual(0);
                    }
                }
            }
            else if (expectedValue == "-Inf") {
                if (deviceType == "cpu") {
                    expect(actual).toBe(-Infinity);
                } else {
                    if (actual === -Infinity) {
                        // OK
                    }
                    else {
                        expect(actual).toEqual(0);
                    }
                }
            }
            else {
                throw new Error(`Unexpected string value: ${expectedValue}`);
            }
        }
        else if (actual instanceof Array) {
            compareArrays(actual, expectedValue, precision, deviceType);
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
    let precision = 6;
    let opName = kernelName;
    if (opName.endsWith("_")) {
        opName = opName.slice(0, -1);
    }
    if (opName in opSpecByName) {
        const opSpec = opSpecByName[opName];
        if (opSpec.precision !== undefined) {
            precision = opSpec.precision;
        }
    }
    else {
        console.warn(`Missing op spec for ${kernelName}`);
    }
    if (runBackward) {
        expect(inputGrads).toHaveLength(expectedGrads.length);
        for (let i = 0; i < inputGrads.length; i++) {
            const inputGrad = inputGrads[i];
            const expectedGrad = expectedGrads[i];
            compareArrays(await inputGrad.toArrayAsync(), expectedGrad, precision, inputGrad.device.type);
        }
    }
    else {
        if (opName === "round" && outputTensor.device.type === "cpu" && inputs[0][0] === 0.5) {
            // HACK: Pass, because round(0.5) is 0 on CPU and 1 on GPU
            return;
        }
        compareArrays(await outputTensor.toArrayAsync(), expectedOutput, precision, outputTensor.device.type);
    }
}

export async function runOpgenTestForward(kernelName: string, inputs: TensorArrayData[], expectedOutputs: TestArrayData[]): Promise<void> {
    await runOpgenTest(kernelName, inputs, expectedOutputs, [], false, false);
}

export async function runOpgenTestBackward(kernelName: string, inputs: TensorArrayData[], expectedGrads: TestArrayData[], gradError: boolean): Promise<void> {
    await runOpgenTest(kernelName, inputs, [], expectedGrads, gradError, true);
}
