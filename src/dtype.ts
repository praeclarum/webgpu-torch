export type Dtype = "int8" | "uint8" | "int32" | "uint32" | "float32" | "int64";

export type ATypedArray = Int8Array | Uint8Array | Int32Array | Uint32Array | Float32Array;

const dtypeArrayCtors = {
    "int8": Int8Array,
    "uint8": Uint8Array,
    "int32": Int32Array,
    "uint32": Uint32Array,
    "float32": Float32Array,
};

export function newTypedArrayForDtype(length: number, dtype: Dtype) {
    switch (dtype) {
        case "int8":
            return new Int8Array(length);
        case "uint8":
            return new Uint8Array(length);
        case "int32":
            return new Int32Array(length);
        case "uint32":
            return new Uint32Array(length);
        case "float32":
            return new Float32Array(length);
        default:
            throw new Error(`Unsupported dtype: ${dtype}`);
    }
}

export function dtypedBufferToTypedArray(dtype: Dtype, buffer: ArrayBuffer, byteOffset?: number, byteLength?: number): ATypedArray {
    if (dtype === "int64") {
        throw new Error("int64 not supported");
    }
    const ctor = dtypeArrayCtors[dtype];
    if (ctor === undefined) {
        throw new Error(`Invalid dtype \"${dtype}\"`);
    }
    if (byteOffset === undefined) {
        byteOffset = 0;
    }
    if (byteLength === undefined) {
        byteLength = buffer.byteLength - byteOffset;
    }
    const length = byteLength / ctor.BYTES_PER_ELEMENT;
    return new ctor(buffer, byteOffset, length);
}

export function dtypeByteSize(dtype: Dtype): number {
    switch (dtype) {
        case "int8":
            return 1;
        case "uint8":
            return 1;
        case "int32":
            return 4;
        case "uint32":
            return 4;
        case "float32":
            return 4;
        case "int64":
            return 8;
        default:
            throw new Error(`Invalid dtype ${dtype}`);
    }
}

export function getDtype(dtype?: Dtype | ATypedArray | null, defaultType?: Dtype): Dtype {
    if (dtype === null || dtype === undefined) {
        return defaultType || "float32";
    } else if (dtype === "int8") {
        return "int8";
    } else if (dtype === "uint8") {
        return "uint8";
    } else if (dtype === "int32") {
        return "int32";
    } else if (dtype === "uint32") {
        return "uint32";
    } else if (dtype === "float32") {
        return "float32";
    } else if (dtype instanceof Int8Array) {
        return "int8";
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
