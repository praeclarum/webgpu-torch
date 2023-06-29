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
    eager(): void {
        // Nothing to do
    }
}

export abstract class ComputedNode extends GraphNode {
    private readonly _outputs: GraphNodeOutputSpec[];
    get outputs(): GraphNodeOutputSpec[] {
        return this._outputs;
    }
    get isSource(): boolean {
        return false;
    }
    constructor(outputs: GraphNodeOutputSpec[]) {
        super();
        this._outputs = outputs;
    }
}

export class ViewNode extends ComputedNode {
    readonly _input: GraphNodeOutputRef;
    get device(): Device {
        return this._input.node.device;
    }
    get inputs(): GraphNodeOutputRef[] {
        return [this._input];
    }
    get storageAvailable(): boolean {
        return this._input.node.storageAvailable;
    }
    get storages(): UntypedStorage[] {
        return [this._input.node.storages[this._input.outputIndex]];
    }
    eager(): void {
        this._input.node.eager();
    }
    constructor(input: GraphNodeOutputRef, shape: Shape, strides: Strides) {
        super([{shape,strides,dtype:input.dtype}]);
        this._input = input;
    }
}

export class KernelNode extends ComputedNode {
    readonly kernel: Kernel;
    readonly params: KernelParamsInput;
    readonly inputs: GraphNodeOutputRef[];
    private _storages: UntypedStorage[] | null = null;
    get device(): Device {
        return this.kernel.device;
    }
    get storageAvailable(): boolean {
        return this._storages !== null;
    }
    get storages(): UntypedStorage[] {
        if (this._storages === null) {
            KernelNode.run([this]);
        }
        return this._storages!;
    }
    constructor(
        kernel: Kernel,
        inputs: GraphNodeOutputRef[],
        params: KernelParamsInput,
        outputs: GraphNodeOutputSpec[]
    ) {
        super(outputs);
        if (inputs.length !== kernel.spec.inputs.length) {
            throw new Error(
                `Kernel \"${kernel.spec.name}\" expects ${kernel.spec.inputs.length} inputs, but ${inputs.length} were provided`
            );
        }
        this.kernel = kernel;
        this.params = params;
        this.inputs = inputs;
    }
    eager(): void {
        if (this._storages === null) {
            KernelNode.run([this]);
        }
    }
    private static run(outputNodes: GraphNode[]): void {
        const device = outputNodes[0].device;
        const [depthFirstNodes, nodesWithStorage, liveness, retainNodes] =
            this.createExecutionPlan(outputNodes);
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
            // Easy case, we already have the storage for this node
            if (nodeId in nodesWithStorage) {
                computedStorages[nodeId] = nodesWithStorage[nodeId].storages;
                continue;
            }
            if (!(node instanceof ComputedNode)) {
                throw new Error(
                    `Node ${nodeId} is not a ComputedNode, but it is not in nodesWithStorage`
                );
            }
            if (node instanceof KernelNode) {
                node as ComputedNode;
                const inputs = node.inputs.map((input, j) => {
                    const inputS =
                        computedStorages[input.node.id][input.outputIndex];
                    if (inputS === undefined) {
                        throw new Error(
                            `Input #${j} of node ${node.id} with kernel \"${node.kernel.spec.name}\" not computed yet`
                        );
                    }
                    return inputS;
                });
                const [nodeRunEnv, paramValues] = node.kernel.getRunEnv(
                    node.params
                );
                const outputs = node.outputs.map((output, i) => {
                    const outputNumElements =
                        node.kernel.spec.outputs[i].size(nodeRunEnv);
                    const outputByteSize =
                        outputNumElements * dtypeByteSize(output.dtype);
                    return alloc(outputByteSize);
                });
                node.kernel.run(inputs, node.params, outputs);
                computedStorages[nodeId] = outputs;
            } else if (node instanceof ViewNode) {
                throw new Error("ViewNodes are not supported yet");
            } else {
                throw new Error(`Unknown node type ${node.constructor.name}`);
            }
            // Free any nodes that are not live anymore
            for (let inLiveId of liveness.ins[i]) {
                if (liveness.outs[i].has(inLiveId)) {
                    continue;
                }
                const storagesToFree = computedStorages[inLiveId];
                delete computedStorages[inLiveId];
                if (!(inLiveId in nodesWithStorage)) {
                    for (let storage of storagesToFree) {
                        free(storage);
                    }
                }
            }
        }
        // Set the storages of the retained nodes
        for (let node of retainNodes) {
            if (node instanceof KernelNode) {
                const nodeId = node.id;
                const storages = computedStorages[nodeId];
                if (storages === undefined) {
                    throw new Error(
                        `Node ${nodeId} with kernel \"${node.kernel.spec.name}\" not computed`
                    );
                }
                node._storages = storages;
            }
        }
    }
    private static createExecutionPlan(
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
                if (node.referenceCount > 1) {
                    retainNodes.push(node);
                }
            }
            depthFirstNodes.push(node);
        }
        for (let outputNode of outputNodes) {
            topoSort(outputNode);
        }
        const liveness = KernelNode.getLiveness(depthFirstNodes, retainNodes);
        // console.log(`Liveness for node#${this.id}`, liveness);
        return [depthFirstNodes, nodesWithStorage, liveness, retainNodes];
    }
    private static getLiveness(
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
