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
    private _referenceCount = 0;
    get referenceCount(): number {
        return this._referenceCount;
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
    getOutputRef(outputIndex: number): GraphNodeOutputRef {
        if (this._outputRefs[outputIndex] === undefined) {
            this._outputRefs[outputIndex] = new GraphNodeOutputRef(this, outputIndex);
        }
        return this._outputRefs[outputIndex];
    }
    addRef(): void {
        this._referenceCount++;
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

export class ComputedNode extends GraphNode {
    readonly kernel: Kernel;
    readonly params: KernelParamsInput;
    readonly inputs: GraphNodeOutputRef[];
    private readonly _outputs: GraphNodeOutputSpec[];
    private _storages: UntypedStorage[] | null = null;
    get isSource(): boolean {
        return false;
    }
    get device(): Device {
        return this.kernel.device;
    }
    get storageAvailable(): boolean {
        return this._storages !== null;
    }
    get storages(): UntypedStorage[] {
        if (this._storages === null) {
            this._storages = this.run();
        }
        return this._storages;
    }
    get outputs(): GraphNodeOutputSpec[] {
        return this._outputs;
    }
    constructor(
        kernel: Kernel,
        inputs: GraphNodeOutputRef[],
        params: KernelParamsInput,
        outputs: GraphNodeOutputSpec[]
    ) {
        super();
        if (inputs.length !== kernel.spec.inputs.length) {
            throw new Error(
                `Kernel \"${kernel.spec.name}\" expects ${kernel.spec.inputs.length} inputs, but ${inputs.length} were provided`
            );
        }
        this.kernel = kernel;
        this.params = params;
        this.inputs = inputs;
        this._outputs = outputs;
    }
    private run(): UntypedStorage[] {
        const device = this.device;
        const [depthFirstNodes, nodesWithStorage, liveness] =
            this.createExecutionPlan();
        const temporaryStoragePool: { [byteSize: number]: UntypedStorage[] } =
            {};
        const alloc = (byteSize: number) => {
            if (!(byteSize in temporaryStoragePool)) {
                temporaryStoragePool[byteSize] = [];
            }
            const storage = temporaryStoragePool[byteSize].pop();
            if (storage !== undefined) {
                // console.log("reuse temp", byteSize);
                return storage;
            }
            // console.log("alloc temp", byteSize);
            return device.alloc(byteSize);
        };
        const free = (storage: UntypedStorage) => {
            const byteSize = storage.byteSize;
            // console.log("free temp", byteSize);
            if (!(byteSize in temporaryStoragePool)) {
                temporaryStoragePool[byteSize] = [storage];
            } else {
                temporaryStoragePool[byteSize].push(storage);
            }
        };
        const computedStorages: { [nodeId: number]: UntypedStorage[] } = {};
        const n = depthFirstNodes.length;
        for (let i = 0; i < n; i++) {
            const node = depthFirstNodes[i];
            const nodeId = node.id;
            // Easy case, we already have the storage for this node.
            if (nodeId in nodesWithStorage) {
                computedStorages[nodeId] = nodesWithStorage[nodeId].storages;
                continue;
            }
            if (!(node instanceof ComputedNode)) {
                throw new Error(
                    `Node ${nodeId} is not a ComputedNode, but it is not in nodesWithStorage`
                );
            }
            const inputs = node.inputs.map((input, j) => {
                const inputS =
                    computedStorages[input.node.id][input.outputIndex];
                if (inputS === undefined) {
                    throw new Error(
                        `Input #${j} of node ${this.id} with kernel \"${this.kernel.spec.name}\" not computed yet`
                    );
                }
                return inputS;
            });
            const [nodeRunEnv, paramValues] = node.kernel.getRunEnv(node.params);
            const outputs = node.outputs.map((output, i) => {
                const outputByteSize =
                    node.kernel.spec.outputs[i].size(nodeRunEnv) *
                    dtypeByteSize(output.dtype);
                return alloc(outputByteSize);
            });
            node.kernel.run(inputs, node.params, outputs);
            computedStorages[nodeId] = outputs;
            // Free any nodes that are not live anymore.
            for (let inLiveId of liveness.ins[i]) {
                if (inLiveId in nodesWithStorage) {
                    continue;
                }
                if (liveness.outs[i].has(inLiveId)) {
                    continue;
                }
                const storagesToFree = computedStorages[inLiveId];
                delete computedStorages[inLiveId];
                for (let storage of storagesToFree) {
                    free(storage);
                }
            }
        }
        return computedStorages[this.id];
    }
    private createExecutionPlan(): [
        GraphNode[],
        { [nodeId: number]: GraphNode },
        { ins: Set<NodeId>[]; outs: Set<NodeId>[] }
    ] {
        const depthFirstNodes: GraphNode[] = [];
        const visitedNodes = new Set<number>();
        const retainNodes: GraphNode[] = [this];
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
                if (node.referenceCount > 1) {
                    retainNodes.push(node);
                }
            }
            depthFirstNodes.push(node);
        }
        topoSort(this);
        const liveness = this.getLiveness(depthFirstNodes, retainNodes);
        // console.log(`Liveness for node#${this.id}`, liveness);
        return [depthFirstNodes, nodesWithStorage, liveness];
    }
    private getLiveness(depthFirstNodes: GraphNode[], outputNodes: GraphNode[]): {
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
                changesOccurred =
                    changesOccurred || !setsAreEqual(nouts, outs[i]);
                const nins = new Set<NodeId>(nouts);
                nins.delete(node.id);
                for (let input of node.inputs) {
                    nins.add(input.node.id);
                }
                changesOccurred =
                    changesOccurred || !setsAreEqual(nins, ins[i]);
                ins[i] = nins;
                outs[i] = nouts;
            }
        }
        return { ins, outs };
    }
}
