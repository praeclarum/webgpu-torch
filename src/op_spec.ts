
export type OpType = "unary" | "binary";

export type OpSpec = {
    name: string;
    aliases?: string[];
    alpha?: boolean;
    type: OpType;
    webGPU: string;
    javaScript: string;    
}

export type UnaryOpSpec = OpSpec & {
    type: "unary";
}

export type BinaryOpSpec = OpSpec & {
    type: "binary";
}

