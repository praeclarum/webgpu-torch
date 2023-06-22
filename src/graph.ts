import type { Device } from "./device";
import type { Dtype } from "./dtype";
import type { Kernel, KernelParamsInput } from "./kernel";
import type { Shape, Strides } from "./shape";
import type { UntypedStorage } from "./storage";

export abstract class GraphNode {
    readonly id: number;
    private static nextId = 0;
    abstract get inputs(): GraphNode[];
    abstract get device(): Device;
    abstract get shape(): Shape;
    abstract get storage(): UntypedStorage;
    abstract get isSource(): boolean;
    constructor() {
        this.id = GraphNode.nextId++;
    }
}

export class SourceNode extends GraphNode {
    private readonly _shape: Shape;
    private readonly _storage: UntypedStorage;
    get isSource(): boolean {
        return true;
    }
    get inputs(): GraphNode[] {
        return [];
    }
    get device(): Device {
        return this._storage.device;
    }
    get shape(): Shape {
        return this._shape;
    }
    get storage(): UntypedStorage {
        return this._storage;
    }
    constructor(storage: UntypedStorage, dtype: Dtype, shape: Shape, strides: Strides) {
        super();
        this._shape = shape;
        this._storage = storage;
    }
}

export class ComputedNode extends GraphNode {
    readonly kernel: Kernel;
    readonly params: KernelParamsInput;
    readonly inputs: GraphNode[];
    private readonly _shape: Shape;
    private _storage: UntypedStorage | null = null;
    get isSource(): boolean {
        return false;
    }
    get device(): Device {
        return this.kernel.device;
    }
    get storage(): UntypedStorage {
        if (this._storage === null) {
            this._storage = this.run();
        }
        return this._storage;
    }
    get shape(): Shape {
        return this._shape;
    }
    constructor(kernel: Kernel,
        params: KernelParamsInput,
        inputs: GraphNode[],
        outputShape: Shape) {
        super();
        this.kernel = kernel;
        this.params = params;
        this.inputs = inputs;
        this._shape = outputShape;
    }
    private run(): UntypedStorage {
        const inputs = this.inputs.map((input) => input.storage);
        const outputs = this.kernel.run(inputs, this.params);
        return outputs[0];
    }
}
