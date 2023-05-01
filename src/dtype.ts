export type Dtype = "float32" | "int32" | "boolean" | "string";

export function dtypeishToDtype(dtype: Dtype | null): Dtype {
    if (dtype === null) {
        return "float32";
    } else {
        return dtype;
    }
}
