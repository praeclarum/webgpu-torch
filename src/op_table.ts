import { OpSpec } from "./op_spec";

export const registry: OpSpec[] = [
    {
        name: "abs",
        aliases: ["absolute"],
        type: "unary",
        forward: "out = abs(input)",
        backward: "inputGrad = input > 0 ? outGrad : -outGrad",
    },
    {
        name: "acos",
        aliases: ["arccos"],
        type: "unary",
        forward: "out = acos(input)",
        backward: "inputGrad = -outGrad / sqrt(1 - input * input)",
    },
    {
        name: "acosh",
        aliases: ["arccosh"],
        type: "unary",
        forward: "out = acosh(input)",
        backward: "inputGrad = outGrad / sqrt(input * input - 1)",
    },
    {
        name: "add",
        type: "binary",
        alpha: true,
        forward: "out = input + other",
        backward: "inputGrad = outGrad; otherGrad = outGrad",
    },
    {
        name: "asin",
        aliases: ["arcsin"],
        type: "unary",
        forward: "out = asin(input)",
        backward: "inputGrad = outGrad / sqrt(1 - input * input)",
    },
    {
        name: "asinh",
        aliases: ["arcsinh"],
        type: "unary",
        forward: "out = asinh(input)",
        backward: "inputGrad = outGrad / sqrt(input * input + 1)",
    },
    {
        name: "atan",
        aliases: ["arctan"],
        type: "unary",
        forward: "out = atan(input)",
        backward: "inputGrad = outGrad / (input * input + 1)",
    },
    {
        name: "atan2",
        aliases: ["arctan2"],
        type: "binary",
        forward: "out = atan2(input, other)",
        backward: "inputGrad = outGrad * other / (input * input + other * other); otherGrad = -outGrad * input / (input * input + other * other)",
    },
    {
        name: "mul",
        aliases: ["multiply"],
        type: "binary",
        alpha: true,
        forward: "out = input * other",
        backward: "inputGrad = outGrad * other; otherGrad = outGrad * input",
    },
    {
        name: "sub",
        aliases: ["subtract"],
        type: "binary",
        alpha: true,
        forward: "out = input - other",
        backward: "inputGrad = outGrad; otherGrad = -outGrad",
    },
];

export const opSpecs = registry.reduce((map, spec) => {
    map[spec.name] = spec;
    return map;
}, {} as {[name: string]: any});
