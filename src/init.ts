import { Tensor } from "./tensor";

export function calculateFanInAndFanOut(tensor: Tensor): [number, number] {
    const dimensions = tensor.ndim;
    if (dimensions < 2) {
        throw new Error("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions");
    }
    const numInputFmaps = tensor.shape[1];
    const numOutputFmaps = tensor.shape[0];
    const receptiveFieldSize = dimensions > 2 ? tensor.shape.slice(2).reduce((a, b) => a * b, 1) : 1;
    const fanIn = numInputFmaps * receptiveFieldSize;
    const fanOut = numOutputFmaps * receptiveFieldSize;
    return [fanIn, fanOut];
}

export function kaimingUniform_(tensor: Tensor): void {
    throw new Error("kaimingUniform_ not implemented");
}

export function uniform_(tensor: Tensor, lowerBound: number, upperBound: number): void {
    throw new Error("uniform_ not implemented");
}
