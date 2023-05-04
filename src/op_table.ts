import { OpSpec } from "./op_spec";

export const registry: OpSpec[] = [
    {
        name: "abs",
        aliases: ["absolute"],
        type: "unary",
        webGPU: "abs(input)",
        javaScript: "Math.abs(input)",
    },
    {
        name: "acos",
        aliases: ["arccos"],
        type: "unary",
        webGPU: "acos(input)",
        javaScript: "Math.acos(input)",
    },
    {
        name: "acosh",
        aliases: ["arccosh"],
        type: "unary",
        webGPU: "acosh(input)",
        javaScript: "Math.acosh(input)",
    },
    {
        name: "add",
        type: "binary",
        alpha: true,
        webGPU: "input + other",
        javaScript: "input + other",
    },
    {
        name: "asin",
        aliases: ["arcsin"],
        type: "unary",
        webGPU: "asin(input)",
        javaScript: "Math.asin(input)",
    },
    {
        name: "asinh",
        aliases: ["arcsinh"],
        type: "unary",
        webGPU: "asinh(input)",
        javaScript: "Math.asinh(input)",
    },
    {
        name: "atan",
        aliases: ["arctan"],
        type: "unary",
        webGPU: "atan(input)",
        javaScript: "Math.atan(input)",
    },
    {
        name: "atan2",
        aliases: ["arctan2"],
        type: "binary",
        webGPU: "atan2(input, other)",
        javaScript: "Math.atan2(input, other)",
    },
    {
        name: "sub",
        aliases: ["subtract"],
        type: "binary",
        alpha: true,
        webGPU: "input - other",
        javaScript: "input - other",
    },
];

export const opSpecs = registry.reduce((map, spec) => {
    map[spec.name] = spec;
    return map;
}, {} as {[name: string]: any});
