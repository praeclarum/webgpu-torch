import { OpSpec } from "./op_spec";

export const registry: OpSpec[] = [
    {
        name: "abs",
        aliases: ["absolute"],
        type: "unary",
        forward: "out = abs(input)",
    },
    {
        name: "acos",
        aliases: ["arccos"],
        type: "unary",
        forward: "out = acos(input)",
    },
    {
        name: "acosh",
        aliases: ["arccosh"],
        type: "unary",
        forward: "out = acosh(input)",
    },
    {
        name: "add",
        type: "binary",
        alpha: true,
        forward: "out = input + other",
    },
    {
        name: "asin",
        aliases: ["arcsin"],
        type: "unary",
        forward: "out = asin(input)",
    },
    {
        name: "asinh",
        aliases: ["arcsinh"],
        type: "unary",
        forward: "out = asinh(input)",
    },
    {
        name: "atan",
        aliases: ["arctan"],
        type: "unary",
        forward: "out = atan(input)",
    },
    {
        name: "atan2",
        aliases: ["arctan2"],
        type: "binary",
        forward: "out = atan2(input, other)",
    },
    {
        name: "sub",
        aliases: ["subtract"],
        type: "binary",
        alpha: true,
        forward: "out = input - other",
    },
];

export const opSpecs = registry.reduce((map, spec) => {
    map[spec.name] = spec;
    return map;
}, {} as {[name: string]: any});
