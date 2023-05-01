export type Dtype = "float32" | "int32" | "boolean";
export type Dtypeish = Dtype;

export function getDtype(dtype: Dtype | null): Dtype {
    if (dtype === null) {
        return "float32";
    } else {
        return dtype;
    }
}
