export type Dtype = "float32" | "int32" | "uint8";
export type Dtypeish = Dtype;

export type ArrayType = Uint8Array | Int32Array | Float32Array;

const dtypeArrayCtors = {
    "uint8": Uint8Array,
    "int32": Int32Array,
    "float32": Float32Array,
};

export function getDtype(dtype: Dtype | ArrayType | null): Dtype {
    if (dtype === null) {
        return "float32";
    } else if (dtype instanceof String && (dtype as Dtype) in dtypeArrayCtors) {
        return dtype as Dtype;
    } else if (dtype instanceof Uint8Array) {
        return "uint8";
    } else if (dtype instanceof Int32Array) {
        return "int32";
    } else if (dtype instanceof Float32Array) {
        return "float32";
    } else {
        throw new Error(`Invalid dtype ${dtype}`);
    }
}

export function newArrayForDtype(length: number, dtype: Dtype) {
    switch (dtype) {
        case "uint8":
            return new Uint8Array(length);
        case "int32":
            return new Int32Array(length);
        case "float32":
            return new Float32Array(length);
        default:
            throw new Error(`Unsupported dtype: ${dtype}`);
    }
}
