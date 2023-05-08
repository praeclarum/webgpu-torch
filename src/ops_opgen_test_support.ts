export type TestArrayData = Array<number | "NaN" | "+Inf" | "-Inf" | TestArrayData>;
export function runOpgenTest(kernelName: string, inputs: TestArrayData[], outputs: TestArrayData[], grads: TestArrayData[]|null, gradError: boolean): void {
}
