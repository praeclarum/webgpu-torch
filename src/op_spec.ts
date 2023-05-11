import type { ExprCode } from "./expr";

export type OpType = "unary" | "binary" | "reduction";

export type OpSpec = {
    name: string;
    nnName?: string;
    torchName?: string;
    nnOp?: boolean;
    aliases?: string[];
    alpha?: boolean;
    type: OpType;
    precision?: number;
    forward: ExprCode;
    backward?: ExprCode;
}

export type UnaryOpSpec = OpSpec & {
    type: "unary";
}

export type BinaryOpSpec = OpSpec & {
    type: "binary";
}

export type ReductionOpSpec = OpSpec & {
    type: "reduction";
    init: ExprCode;
    combineOp: "+" | "*" | "&&" | "||";
    reduce?: ExprCode;
}

export type AnOpSpec = UnaryOpSpec | BinaryOpSpec | ReductionOpSpec;
