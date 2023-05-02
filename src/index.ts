export function testExport(input: any): any {
    return input;
}

export function hasWebGPU() {
    const anavigator = navigator as any;
    if (!anavigator.gpu) { return false; }
    return true;
}

export * from "./shape";
export * from "./tensor";
export * from "./factories";
export * from "./ops";

export const foo = 42;
