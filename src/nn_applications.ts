import { Linear, Module, ReLU, Sequential } from "./nn";
import type { Tensor } from "./tensor";

export class DeepSDF extends Module {
    pointEncoder: Linear;
    block1: Sequential;
    distanceHead: Linear;
    constructor(config?: {depth?: number, width?: number}) {
        super();
        const depth = config?.depth ?? 8;
        const width = config?.width ?? 128;
        this.pointEncoder = new Linear(3, width);
        this.block1 = new Sequential();
        for (let i = 0; i < depth; i++) {
            this.block1.push(new Sequential(
                new Linear(width, width),
                new ReLU(),
            ));
        }
        this.distanceHead = new Linear(width, 1);
    }
    forward(points: Tensor): Tensor {
        let h = this.pointEncoder.forward(points);
        h = this.block1.forward(h);
        const distance = this.distanceHead.forward(h);
        return distance;
    }
}
