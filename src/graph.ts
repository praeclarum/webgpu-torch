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

export type GraphNodeOutputRef = { node: GraphNode; outputIndex: number };
export type GraphNodeOutputSpec = {
    dtype: Dtype;
    shape: Shape;
    strides: Strides;
};

export abstract class GraphNode {
    readonly id: NodeId;
    private static nextId = 0;
    abstract get device(): Device;
    abstract get inputs(): GraphNodeOutputRef[];
    abstract get outputs(): GraphNodeOutputSpec[];
    // abstract get shape(): Shape;
    // abstract get strides(): Shape;
    // abstract get dtype(): Dtype;
    abstract get storageAvailable(): boolean;
    abstract get storages(): UntypedStorage[];
    abstract get isSource(): boolean;
    get isComputed(): boolean {
        return !this.isSource;
    }
    constructor() {
        this.id = GraphNode.nextId++;
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
        // this._dtype = shaderTypeToDtype(this.kernel.spec.outputs[0].shaderType);
        // this._shapes = outputShapes;
        // this._strides = defaultStrides(outputShape);
    }
    private run(): UntypedStorage[] {
        // const inputs = this.inputs.map((input) => input.storage);
        // const outputs = this.kernel.run(inputs, this.params);
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
            const outputs = this.outputs.map((output, i) => {
                // output size = dtypeByteSize(output.dtype) * shapeSize(output.shape)
                const outputByteSize =
                    this.kernel.eval(
                        this.kernel.spec.outputs[i].size,
                        this.params
                    ) * dtypeByteSize(output.dtype);
                return alloc(outputByteSize);
            });
            node.kernel.run(inputs, this.params, outputs);
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
            }
            depthFirstNodes.push(node);
        }
        topoSort(this);
        const liveness = this.getLiveness(depthFirstNodes);
        // console.log(`Liveness for node#${this.id}`, liveness);
        return [depthFirstNodes, nodesWithStorage, liveness];
    }
    private getLiveness(depthFirstNodes: GraphNode[]): {
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
                    nouts.add(node.id);
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
