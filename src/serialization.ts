import { StateDict } from "./nn_module";
import { shapeSize } from "./shape";
import { Dtype, dtypeByteSize } from "./dtype";

function dtypeToSafetensorDtype(dtype: Dtype): string {
    switch (dtype) {
        case "int8":
            return "I8";
        case "uint8":
            return "U8";
        case "int32":
            return "I32";
        case "uint32":
            return "U32";
        case "float32":
            return "F32";
        case "int64":
            return "I64";
        default:
            throw new Error(`Invalid dtype ${dtype}`);
    }
}

export async function saveSafetensorsAsync(
    stateDict: StateDict
): Promise<ArrayBuffer> {
    const headerJson: { [tensorName: string]: any } = {};
    let begin = 0;
    for (const key in stateDict) {
        const tensor = stateDict[key];
        const shape = tensor.shape;
        const byteSize = shapeSize(shape) * dtypeByteSize(tensor.dtype);
        const end = begin + byteSize;
        headerJson[key] = {
            dtype: dtypeToSafetensorDtype(tensor.dtype),
            shape,
            data_offsets: [begin, end],
        };
        begin = end;
    }
    const header = JSON.stringify(headerJson);
    const headerBytes = new TextEncoder().encode(header);
    const result = new ArrayBuffer(8 + headerBytes.byteLength + begin);
    const view = new DataView(result);
    view.setBigUint64(0, BigInt(headerBytes.byteLength), true);
    new Uint8Array(result, 8, headerBytes.byteLength).set(headerBytes);
    begin = 8 + headerBytes.byteLength;
    for (const key in stateDict) {
        const tensor = stateDict[key];
        const data = (await tensor.storage.toTypedArrayAsync(tensor.dtype)).buffer;
        const byteSize = shapeSize(tensor.shape) * dtypeByteSize(tensor.dtype);
        new Uint8Array(result, begin, byteSize).set(new Uint8Array(data));
        begin += byteSize;
    }
    return result;
}
