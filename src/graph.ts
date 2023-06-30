import type { Device } from "./device";
import { dtypeByteSize, type Dtype } from "./dtype";
import {
    shaderTypeToDtype,
    type Kernel,
    type KernelParamsInput,
} from "./kernel";
import { shapeSize, type Shape, type Strides, defaultStrides } from "./shape";
import type { UntypedStorage } from "./storage";

type NodeId = number;

export type GraphNodeOutputSpec = {
    dtype: Dtype;
    shape: Shape;
    strides: Strides;
};

export class GraphNodeOutputRef {
    readonly node: GraphNode;
    readonly outputIndex: number;
    get dtype(): Dtype {
        return this.node.outputs[this.outputIndex].dtype;
    }
    constructor(node: GraphNode, outputIndex: number) {
        this.node = node;
        this.outputIndex = outputIndex;
    }
    addRef(): GraphNodeOutputRef {
        this.node.addRef();
        return this;
    }
}

export abstract class GraphNode {
    private static nextId = 0;
    readonly id: NodeId;
    private readonly _outputRefs: GraphNodeOutputRef[] = [];
    private _refCount = 0;
    get refCount(): number {
        return this._refCount;
    }
    abstract get device(): Device;
    abstract get inputs(): GraphNodeOutputRef[];
    abstract get outputs(): GraphNodeOutputSpec[];
    abstract get storageAvailable(): boolean;
    abstract get storages(): UntypedStorage[];
    abstract get isSource(): boolean;
    get isComputed(): boolean {
        return !this.isSource;
    }
    constructor() {
        this.id = GraphNode.nextId++;
    }
    abstract eager(): void;
    getOutputRef(outputIndex: number): GraphNodeOutputRef {
        if (this._outputRefs[outputIndex] === undefined) {
            this._outputRefs[outputIndex] = new GraphNodeOutputRef(
                this,
                outputIndex
            );
        }
        return this._outputRefs[outputIndex];
    }
    addRef(): void {
        this._refCount++;
    }
}

export class SourceNode extends GraphNode {
    private readonly _outputs: GraphNodeOutputSpec[];
    private readonly _storages: UntypedStorage[];
    get isSource(): boolean {
        return true;
    }
    get device(): Device {
        return this._storages[0].device;
    }
    get inputs(): GraphNodeOutputRef[] {
        return [];
    }
    get outputs(): GraphNodeOutputSpec[] {
        return this._outputs;
    }
    get storageAvailable(): boolean {
        return true;
    }
    get storages(): UntypedStorage[] {
        return this._storages;
    }
    constructor(
        storage: UntypedStorage,
        dtype: Dtype,
        shape: Shape,
        strides: Strides
    ) {
        super();
        this._outputs = [{ dtype, shape, strides }];
        this._storages = [storage];
    }
    eager(): void {
        // Nothing to do
    }
}

export abstract class ComputedNode extends GraphNode {
    private _storages: UntypedStorage[] | null = null;
    readonly inputs: GraphNodeOutputRef[];
    readonly outputs: GraphNodeOutputSpec[];
    get isSource(): boolean {
        return false;
    }
    get storageAvailable(): boolean {
        return this._storages !== null;
    }
    get storages(): UntypedStorage[] {
        if (this._storages === null) {
            run([this]);
        }
        return this._storages!;
    }
    constructor(inputs: GraphNodeOutputRef[], outputs: GraphNodeOutputSpec[]) {
        super();
        this.inputs = inputs;
        this.outputs = outputs;
    }
    eager(): void {
        if (this._storages === null) {
            run([this]);
        }
    }
    setStorages(storages: UntypedStorage[]) {
        this._storages = storages;
    }
    abstract runNode(
        inputs: UntypedStorage[],
        alloc: (byteSize: number) => UntypedStorage
    ): UntypedStorage[];
}

export class ViewNode extends ComputedNode {
    get device(): Device {
        return this.inputs[0].node.device;
    }
    get storageAvailable(): boolean {
        return this.inputs[0].node.storageAvailable;
    }
    get storages(): UntypedStorage[] {
        return [this.inputs[0].node.storages[this.inputs[0].outputIndex]];
    }
    constructor(input: GraphNodeOutputRef, shape: Shape, strides: Strides) {
        super([input], [{ shape, strides, dtype: input.dtype }]);
    }
    runNode(
        inputs: UntypedStorage[],
        alloc: (byteSize: number) => UntypedStorage
    ): UntypedStorage[] {
        return inputs;
    }
}

export class KernelNode extends ComputedNode {
    readonly kernel: Kernel;
    readonly params: KernelParamsInput;
    get device(): Device {
        return this.kernel.device;
    }
    constructor(
        kernel: Kernel,
        inputs: GraphNodeOutputRef[],
        params: KernelParamsInput,
        outputs: GraphNodeOutputSpec[]
    ) {
        super(inputs, outputs);
        if (inputs.length !== kernel.spec.inputs.length) {
            throw new Error(
                `Kernel \"${kernel.spec.name}\" expects ${kernel.spec.inputs.length} inputs, but ${inputs.length} were provided`
            );
        }
        this.kernel = kernel;
        this.params = params;
    }
    runNode(
        inputs: UntypedStorage[],
        alloc: (byteSize: number) => UntypedStorage
    ): UntypedStorage[] {
        const kernel = this.kernel;
        const [nodeRunEnv, paramValues] = kernel.getRunEnv(this.params);
        const outputs = this.outputs.map((output, i) => {
            const outputNumElements = kernel.spec.outputs[i].size(nodeRunEnv);
            const outputByteSize =
                outputNumElements * dtypeByteSize(output.dtype);
            return alloc(outputByteSize);
        });
        kernel.run(inputs, this.params, outputs);
        return outputs;
    }
}

function run(outputNodes: GraphNode[]): void {
    const device = outputNodes[0].device;
    const [depthFirstNodes, nodesWithStorage, liveness, retainNodes] =
        createExecutionPlan(outputNodes);
    const temporaryStoragePool: { [byteSize: number]: UntypedStorage[] } = {};
    const refCounts: Map<UntypedStorage, number> = new Map();
    const addRef = (storage: UntypedStorage) => {
        const refCount = refCounts.get(storage);
        if (refCount === undefined) {
            refCounts.set(storage, 1);
        } else {
            refCounts.set(storage, refCount + 1);
        }
    };
    const alloc = (byteSize: number) => {
        if (!(byteSize in temporaryStoragePool)) {
            temporaryStoragePool[byteSize] = [];
        }
        let storage = temporaryStoragePool[byteSize].pop();
        if (storage === undefined) {
            storage = device.alloc(byteSize);
        }
        return storage;
    };
    const remRef = (storage: UntypedStorage) => {
        // Decrement ref count
        const refCount = refCounts.get(storage);
        if (refCount === undefined) {
            throw new Error("Storage is not ref counted");
        }
        if (refCount === 1) {
            // This is the last ref, free the storage by putting it back in the pool
            refCounts.delete(storage);
            const byteSize = storage.byteSize;
            if (!(byteSize in temporaryStoragePool)) {
                temporaryStoragePool[byteSize] = [storage];
            } else {
                temporaryStoragePool[byteSize].push(storage);
            }
        } else {
            refCounts.set(storage, refCount - 1);
        }
    };
    const computedStorages: { [nodeId: number]: UntypedStorage[] } = {};
    const n = depthFirstNodes.length;
    for (let i = 0; i < n; i++) {
        const node = depthFirstNodes[i];
        const nodeId = node.id;
        // Easy case, we already have the storage for this node
        if (nodeId in nodesWithStorage) {
            computedStorages[nodeId] = nodesWithStorage[nodeId].storages;
            continue;
        }
        // Guess we have to compute it
        if (!(node instanceof ComputedNode)) {
            throw new Error(
                `Node ${nodeId} is not a ComputedNode, but it is not in nodesWithStorage`
            );
        }
        const inputs = node.inputs.map((input, j) => {
            const inputS = computedStorages[input.node.id][input.outputIndex];
            if (inputS === undefined) {
                throw new Error(
                    `Input #${j} of node ${node.id} not computed yet`
                );
            }
            return inputS;
        });
        const outputs = node.runNode(inputs, alloc);
        outputs.forEach(addRef);
        computedStorages[nodeId] = outputs;
        // Free any nodes' storages that are not live anymore
        for (let inLiveId of liveness.ins[i]) {
            if (liveness.outs[i].has(inLiveId)) {
                continue;
            }
            const storagesToFree = computedStorages[inLiveId];
            delete computedStorages[inLiveId];
            if (!(inLiveId in nodesWithStorage)) {
                for (let storage of storagesToFree) {
                    remRef(storage);
                }
            }
        }
    }
    // Set the storages of the retained nodes
    for (let node of retainNodes) {
        if (node instanceof ComputedNode) {
            const nodeId = node.id;
            const storages = computedStorages[nodeId];
            if (storages === undefined) {
                throw new Error(`Node ${nodeId} not computed`);
            }
            node.setStorages(storages);
        }
    }
}

function createExecutionPlan(
    outputNodes: GraphNode[]
): [
    GraphNode[],
    { [nodeId: number]: GraphNode },
    { ins: Set<NodeId>[]; outs: Set<NodeId>[] },
    GraphNode[]
] {
    const depthFirstNodes: GraphNode[] = [];
    const visitedNodes = new Set<number>();
    const retainNodes: GraphNode[] = outputNodes.slice();
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
                topoSort(input.node);
            }
            if (node.refCount > 1) {
                retainNodes.push(node);
            }
        }
        depthFirstNodes.push(node);
    }
    for (let outputNode of outputNodes) {
        topoSort(outputNode);
    }
    const liveness = getLiveness(depthFirstNodes, retainNodes);
    // console.log(`Liveness for node#${this.id}`, liveness);
    return [depthFirstNodes, nodesWithStorage, liveness, retainNodes];
}

function getLiveness(
    depthFirstNodes: GraphNode[],
    outputNodes: GraphNode[]
): {
    ins: Set<NodeId>[];
    outs: Set<NodeId>[];
} {
    const n = depthFirstNodes.length;
    const ins: Set<NodeId>[] = [];
    const outs: Set<NodeId>[] = [];
    for (let i = 0; i < n; i++) {
        ins.push(new Set<NodeId>());
        outs.push(new Set<NodeId>());
    }
    let changesOccurred = true;
    while (changesOccurred) {
        changesOccurred = false;
        for (let i = n - 1; i >= 0; i--) {
            const node = depthFirstNodes[i];
            const nouts = new Set<NodeId>();
            if (i + 1 < n) {
                for (let inn of ins[i + 1]) {
                    nouts.add(inn);
                }
            } else {
                for (let outputNode of outputNodes) {
                    nouts.add(outputNode.id);
                }
            }
            changesOccurred = changesOccurred || !setsAreEqual(nouts, outs[i]);
            const nins = new Set<NodeId>(nouts);
            nins.delete(node.id);
            for (let input of node.inputs) {
                nins.add(input.node.id);
            }
            changesOccurred = changesOccurred || !setsAreEqual(nins, ins[i]);
            ins[i] = nins;
            outs[i] = nouts;
        }
    }
    return { ins, outs };
}

function setsAreEqual<T>(a: Set<T>, b: Set<T>) {
    if (a.size !== b.size) {
        return false;
    }
    for (let item of a) {
        if (!b.has(item)) {
            return false;
        }
    }
    return true;
}
