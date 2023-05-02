export type Dtype = "float32" | "int32" | "uint8";
export type Dtypeish = Dtype;

export type ATypedArray = Uint8Array | Int32Array | Float32Array;

const dtypeArrayCtors = {
    "uint8": Uint8Array,
    "int32": Int32Array,
    "float32": Float32Array,
};

export function getDtype(dtype: Dtype | ATypedArray | null): Dtype {
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
