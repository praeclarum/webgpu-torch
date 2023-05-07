export type Dtype = "float32" | "int32" | "uint32" | "uint8";

export type ATypedArray = Uint8Array | Int32Array | Uint32Array | Float32Array;

export const dtypeArrayCtors = {
    "uint8": Uint8Array,
    "int32": Int32Array,
    "uint32": Uint32Array,
    "float32": Float32Array,
};

export function dtypeByteSize(dtype: Dtype): number {
    switch (dtype) {
        case "uint8":
            return 1;
        case "int32":
            return 4;
        case "uint32":
            return 4;
        case "float32":
            return 4;
        default:
            throw new Error(`Invalid dtype ${dtype}`);
    }
}

export function getDtype(dtype?: Dtype | ATypedArray | null, defaultType?: Dtype): Dtype {
    if (dtype === null || dtype === undefined) {
        return defaultType || "float32";
    } else if (dtype === "uint8") {
        return "uint8";
    } else if (dtype === "int32") {
        return "int32";
    } else if (dtype === "uint32") {
        return "uint32";
    } else if (dtype === "float32") {
        return "float32";
    } else if (dtype instanceof Uint8Array) {
        return "uint8";
    } else if (dtype instanceof Int32Array) {
        return "int32";
    } else if (dtype instanceof Uint32Array) {
        return "uint32";
    } else if (dtype instanceof Float32Array) {
        return "float32";
    } else {
        throw new Error(`Invalid dtype ${JSON.stringify(dtype)}`);
    }
}
