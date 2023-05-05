import { OpSpec } from "./op_spec";

export const registry: OpSpec[] = [
    {
        name: "abs",
        aliases: ["absolute"],
        type: "unary",
        forward: "output = abs(input)",
        backward: "inputGrad = input > 0 ? outputGrad : -outputGrad",
    },
    {
        name: "acos",
        aliases: ["arccos"],
        type: "unary",
        forward: "output = acos(input)",
        backward: "inputGrad = -outputGrad / sqrt(1 - input * input)",
    },
    {
        name: "acosh",
        aliases: ["arccosh"],
        type: "unary",
        forward: "output = acosh(input)",
        backward: "inputGrad = outputGrad / sqrt(input * input - 1)",
    },
    {
        name: "add",
        type: "binary",
        alpha: true,
        forward: "output = input + other",
        backward: "inputGrad = outputGrad; otherGrad = outputGrad",
    },
    {
        name: "asin",
        aliases: ["arcsin"],
        type: "unary",
        forward: "output = asin(input)",
        backward: "inputGrad = outputGrad / sqrt(1 - input * input)",
    },
    {
        name: "asinh",
        aliases: ["arcsinh"],
        type: "unary",
        forward: "output = asinh(input)",
        backward: "inputGrad = outputGrad / sqrt(input * input + 1)",
    },
    {
        name: "atan",
        aliases: ["arctan"],
        type: "unary",
        forward: "output = atan(input)",
        backward: "inputGrad = outputGrad / (input * input + 1)",
    },
    {
        name: "atan2",
        aliases: ["arctan2"],
        type: "binary",
        forward: "output = atan2(input, other)",
        backward: "inputGrad = outputGrad * other / (input * input + other * other); otherGrad = -outputGrad * input / (input * input + other * other)",
    },
    {
        name: "mul",
        aliases: ["multiply"],
        type: "binary",
        alpha: true,
        forward: "output = input * other",
        backward: "inputGrad = outputGrad * other; otherGrad = outputGrad * input",
    },
    {
        name: "sub",
        aliases: ["subtract"],
        type: "binary",
        alpha: true,
        forward: "output = input - other",
        backward: "inputGrad = outputGrad; otherGrad = -outputGrad",
    },
];

export const opSpecs = registry.reduce((map, spec) => {
    map[spec.name] = spec;
    return map;
}, {} as {[name: string]: any});
