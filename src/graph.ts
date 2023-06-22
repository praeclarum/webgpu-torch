import type { Device } from "./device";
import { dtypeByteSize, type Dtype } from "./dtype";
import { shaderTypeToDtype, type Kernel, type KernelParamsInput } from "./kernel";
import { shapeSize, type Shape, type Strides } from "./shape";
import type { UntypedStorage } from "./storage";

export abstract class GraphNode {
    readonly id: number;
    private static nextId = 0;
    abstract get inputs(): GraphNode[];
    abstract get device(): Device;
    abstract get shape(): Shape;
    abstract get storageAvailable(): boolean;
    abstract get storage(): UntypedStorage;
    abstract get dtype(): Dtype;
    abstract get isSource(): boolean;
    get isComputed(): boolean {
        return !this.isSource;
    }
    constructor() {
        this.id = GraphNode.nextId++;
    }
}

export class SourceNode extends GraphNode {
    private readonly _shape: Shape;
    private readonly _storage: UntypedStorage;
    private readonly _dtype: Dtype;
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
    get storageAvailable(): boolean {
        return true;
    }
    get storage(): UntypedStorage {
        return this._storage;
    }
    get dtype(): Dtype {
        return this._dtype;
    }
    constructor(
        storage: UntypedStorage,
        dtype: Dtype,
        shape: Shape,
        strides: Strides
    ) {
        super();
        this._shape = shape;
        this._storage = storage;
        this._dtype = dtype;
    }
}

export class ComputedNode extends GraphNode {
    readonly kernel: Kernel;
    readonly params: KernelParamsInput;
    readonly inputs: GraphNode[];
    private readonly _shape: Shape;
    private readonly _dtype: Dtype;
    private _storage: UntypedStorage | null = null;
    get isSource(): boolean {
        return false;
    }
    get device(): Device {
        return this.kernel.device;
    }
    get storageAvailable(): boolean {
        return this._storage !== null;
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
    get dtype(): Dtype {
        return this._dtype;
    }
    constructor(
        kernel: Kernel,
        inputs: GraphNode[],
        params: KernelParamsInput,
        outputShape: Shape
    ) {
        super();
        this.kernel = kernel;
        this._dtype = shaderTypeToDtype(this.kernel.spec.outputs[0].shaderType);
        this.params = params;
        this.inputs = inputs;
        this._shape = outputShape;
    }
    private run(): UntypedStorage {
        // const inputs = this.inputs.map((input) => input.storage);
        // const outputs = this.kernel.run(inputs, this.params);
        const device = this.device;
        const [nodesToEval, nodesWithStorage] = this.createExecutionPlan();
        const temporaryStoragePool: {[byteSize: number]: UntypedStorage[]} = {};
        const alloc = (byteSize: number) => {
            if (!(byteSize in temporaryStoragePool)) {
                temporaryStoragePool[byteSize] = [];
            }
            const storage = temporaryStoragePool[byteSize].pop();
            if (storage !== undefined) {
                console.log("reuse temp", byteSize);
                return storage;
            }
            console.log("alloc temp", byteSize);
            return device.alloc(byteSize);
        };
        const free = (storage: UntypedStorage) => {
            const byteSize = storage.byteSize;
            console.log("free temp", byteSize);
            if (!(byteSize in temporaryStoragePool)) {
                temporaryStoragePool[byteSize] = [storage];
            } else {
                temporaryStoragePool[byteSize].push(storage);
            }
        };
        const computedStorages: {[nodeId: number]: UntypedStorage} = {};
        for (let node of nodesToEval) {
            const nodeId = node.id;
            if (nodeId in nodesWithStorage) {
                computedStorages[nodeId] = nodesWithStorage[nodeId].storage;
                continue;
            }
            const inputs = node.inputs.map((input) => computedStorages[input.id]);
            const output = alloc(dtypeByteSize(node.dtype) * shapeSize(node.shape));
            this.kernel.run(inputs, this.params, [output]);
            computedStorages[nodeId] = output;
        }
        return computedStorages[this.id];
    }
    private createExecutionPlan(): [GraphNode[], { [nodeId: number]: GraphNode }] {
        const sortedNodes: GraphNode[] = [];
        const visitedNodes = new Set<number>();
        const nodesWithStorage: { [nodeId: number]: GraphNode } = {};
        function topoSort(node: GraphNode) {
            if (visitedNodes.has(node.id)) {
                return;
            }
            visitedNodes.add(node.id);
            if (node.storageAvailable) {
                nodesWithStorage[node.id] = node;
            } else {
                for (let input of node.inputs) {
                    topoSort(input);
                }
            }
            sortedNodes.push(node);
        }
        topoSort(this);
        // console.log("Sorted nodes", sortedNodes.length);
        return [sortedNodes, nodesWithStorage];
    }
}
