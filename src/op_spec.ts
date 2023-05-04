import { KernelParamSpec, KernelSpec } from "./kernel";
import { ExprCode, exprNodeToString, parseCode, substituteIdentifiers } from "./expr";

export type OpType = "unary" | "binary";

export type OpSpec = {
    name: string;
    aliases?: string[];
    alpha?: boolean;
    type: OpType;
    gradients?: ExprCode[];
    webGPU: ExprCode;
    javaScript: ExprCode;    
}

export type UnaryOpSpec = OpSpec & {
    type: "unary";
    gradients?: [ExprCode];
}

export type BinaryOpSpec = OpSpec & {
    type: "binary";
    gradients?: [ExprCode, ExprCode];
}
