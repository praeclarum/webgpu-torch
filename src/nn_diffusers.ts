import type { Dtype } from "./dtype";
import { Linear } from "./nn_basic";
import { GroupNorm } from "./nn_norm";
import { Conv2d } from "./nn_2d";
import { Module, ModuleList, Sequential } from "./nn_module";
import { SiLU } from "./nn_opgen";
import { cat } from "./ops_artisanal";
import { cos, div, exp, mul, sin } from "./ops_opgen";
import type { Tensor } from ".";

export interface UNetModelConfig {
    /** channels of input tensor */
    inChannels: number;
    /** base channels in model */
    modelChannels: number;
    /** channels of output tensor */
    outChannels: number;
    /** number of residual blocks per down/up sampling stage */
    numResBlocks?: number;
    /**
     * a collection of downsample rates at which attention is applied.
     * For example, if this contains 4, then at 4x downsampling, attention is applied.
     */
    attentionResolutions?: number[];
    /** the dropout probability */
    dropout?: number;
    /** channel multiples per down/up sampling stage */
    channelMult?: number[];
    /** if `true` use learnable convolutional upsampling/downsampling */
    convResample?: boolean;
    /** determines whether to use 1D, 2D, or 3D convolutions */
    dims?: number;
    /** if specified, then this model will be class-conditioned with `numClasses` classes */
    numClasses?: number;
    useCheckpoint?: boolean;
    dtype?: Dtype;
    /** the number of attention heads in each attention layer */
    numHeads?: number;
    /** if specified, ignore numHeads and instead use this number of channels in each attention head */
    numHeadChannels?: number;
    numHeadsUpSample?: number;
    /** use a FiLM-like conditioning mechanism */
    useScaleShiftNorm?: boolean;
    /** use residual blocks for up/down sampling */
    resblockUpdown?: boolean;
    useNewAttentionOrder?: boolean;
    useSpatialTransformer?: boolean;
    transformerDepth?: number;
    contextDim?: number;
}

/**
 * Full UNet model with attention and timestep embedding.
 *
 * ** This is still a work in progress **
 */
export class UNetModel extends Module {
    inChannels: number;
    modelChannels: number;
    outChannels: number;
    numResBlocks: number;
    attentionResolutions: number[];
    dropout: number;
    channelMult: number[];
    convResample: boolean;
    numClasses: number | null;
    useCheckpoint: boolean;
    dtype: Dtype;
    numHeadChannels: number;

    private _featureSize: number;

    timeEmbed: Sequential;
    inputBlocks: ModuleList;
    middleBlock: TimestepEmbedSequential;
    outputBlocks: ModuleList;
    out: Sequential;

    /**
     * Constructs a new UNet model with a given number of input and output channels along with
     * a variety of other options.
     */
    constructor(config: UNetModelConfig) {
        super();

        if (config.numHeads === -1) {
            if (config.numHeadChannels === -1) {
                throw new Error(
                    `Must specify either numHeads or numHeadChannels`
                );
            }
        }
        if (config.numHeadChannels === -1) {
            if (config.numHeads === -1) {
                throw new Error(
                    `Must specify either numHeads or numHeadChannels`
                );
            }
        }

        this.inChannels = config.inChannels;
        this.modelChannels = config.modelChannels;
        this.outChannels = config.outChannels;
        this.numResBlocks = config.numResBlocks || 2;
        this.attentionResolutions = config.attentionResolutions || [];
        this.dropout = config.dropout || 0.0;
        this.channelMult = config.channelMult || [1, 2, 4, 8];
        this.convResample =
            config.convResample === undefined ? true : config.convResample;
        this.numClasses = config.numClasses || null;
        this.useCheckpoint = config.useCheckpoint || false;
        this.dtype = config.dtype || "float32";
        this.numHeadChannels = config.numHeadChannels || -1;

        const contextDim = config.contextDim || null;
        const dims = config.dims || 2;
        const resblockDownup = config.resblockUpdown || false;
        const transformerDepth = config.transformerDepth || 1;
        const useCheckpoint = config.useCheckpoint || false;
        const useNewAttentionOrder = config.useNewAttentionOrder || false;

        const timeEmbedDim = config.modelChannels * 4;
        this.timeEmbed = new Sequential(
            linear(this.modelChannels, timeEmbedDim),
            new SiLU(),
            linear(timeEmbedDim, timeEmbedDim),
        );

        this.inputBlocks = new ModuleList([
            new TimestepEmbedSequential(
                conv_nd(
                    dims,
                    this.inChannels,
                    this.modelChannels,
                    3,
                    1,
                    1,
                    this.dtype
                )
            ),
        ]);
        this._featureSize = this.modelChannels;
        let numHeads = config.numHeads || -1;
        const numHeadsUpSample = config.numHeadsUpSample || -1;
        let dimHead: number = 0;

        // Create the backbone
        const inputBlockChans = [this.modelChannels];
        let ch = this.modelChannels;
        let ds = 1;
        for (const [level, mult] of this.channelMult.entries()) {
            for (let i = 0; i < this.numResBlocks; i++) {
                const layers = [
                    new ResBlock(
                        ch,
                        timeEmbedDim,
                        mult * this.modelChannels,
                        dims,
                        false,
                        false,
                        this.dropout,
                        this.useCheckpoint,
                        undefined,
                        config.useScaleShiftNorm
                    ),
                ];
                ch = mult * this.modelChannels;
                if (this.attentionResolutions.includes(ds)) {
                    if (this.numHeadChannels === -1) {
                        dimHead = ch / numHeads;
                    } else {
                        numHeads = ch / this.numHeadChannels;
                        dimHead = this.numHeadChannels;
                    }
                    if (config.useSpatialTransformer) {
                        layers.push(
                            new SpatialTransformer(
                                ch,
                                numHeads,
                                dimHead,
                                transformerDepth,
                                0.0,
                                contextDim
                            )
                        );
                    } else {
                        layers.push(
                            new AttentionBlock(
                                ch,
                                numHeads,
                                dimHead,
                                useCheckpoint,
                                useNewAttentionOrder
                            )
                        );
                    }
                }
            }
            if (level != this.channelMult.length - 1) {
                const outCh = ch;
                this.inputBlocks.push(
                    new TimestepEmbedSequential(
                        resblockDownup
                            ? new ResBlock(
                                  ch,
                                  timeEmbedDim,
                                  outCh,
                                  dims,
                                  false,
                                  true,
                                  this.dropout,
                                  useCheckpoint,
                                  undefined,
                                  config.useScaleShiftNorm
                              )
                            : new Downsample(ch, this.convResample, dims, outCh)
                    )
                );
                ch = outCh;
                inputBlockChans.push(ch);
                ds *= 2;
                this._featureSize += ch;
            }
        }

        // Create the bottleneck
        if (this.numHeadChannels === -1) {
            dimHead = ch / numHeads;
        } else {
            numHeads = ch / this.numHeadChannels;
            dimHead = this.numHeadChannels;
        }
        this.middleBlock = new TimestepEmbedSequential(
            new ResBlock(
                ch,
                timeEmbedDim,
                ch,
                dims,
                false,
                false,
                this.dropout,
                this.useCheckpoint,
                undefined,
                config.useScaleShiftNorm
            ),
            config.useSpatialTransformer
                ? new SpatialTransformer(
                      ch,
                      numHeads,
                      dimHead,
                      transformerDepth,
                      0.0,
                      contextDim
                  )
                : new AttentionBlock(
                      ch,
                      numHeads,
                      dimHead,
                      useCheckpoint,
                      useNewAttentionOrder
                  ),
            new ResBlock(
                ch,
                timeEmbedDim,
                ch,
                dims,
                false,
                false,
                this.dropout,
                this.useCheckpoint,
                undefined,
                config.useScaleShiftNorm
            )
        );
        this._featureSize += ch;

        // Output blocks
        this.outputBlocks = new ModuleList([]);
        for (const [level, mult] of this.channelMult.entries()) {
            for (let i = 0; i < this.numResBlocks + 1; i++) {
                const ich = inputBlockChans.pop() ?? 0;
                const layers = [
                    new ResBlock(
                        ch + ich,
                        timeEmbedDim,
                        this.modelChannels * mult,
                        dims,
                        false,
                        false,
                        this.dropout,
                        this.useCheckpoint,
                        undefined,
                        config.useScaleShiftNorm
                    ),
                ];
                ch = this.modelChannels * mult;
                if (this.attentionResolutions.includes(ds)) {
                    if (this.numHeadChannels === -1) {
                        dimHead = ch / numHeads;
                    } else {
                        numHeads = ch / this.numHeadChannels;
                        dimHead = this.numHeadChannels;
                    }
                    layers.push(
                        config.useSpatialTransformer
                            ? new SpatialTransformer(
                                  ch,
                                  numHeads,
                                  dimHead,
                                  transformerDepth,
                                  0.0,
                                  contextDim
                              )
                            : new AttentionBlock(
                                  ch,
                                  numHeadsUpSample,
                                  dimHead,
                                  useCheckpoint,
                                  useNewAttentionOrder
                              )
                    );
                }
                if (level > 0 && i == this.numResBlocks) {
                    const outCh = ch;
                    layers.push(
                        resblockDownup
                            ? new ResBlock(
                                  ch,
                                  timeEmbedDim,
                                  outCh,
                                  dims,
                                  true,
                                  false,
                                  this.dropout,
                                  this.useCheckpoint,
                                  undefined,
                                  config.useScaleShiftNorm
                              )
                            : new Upsample(ch, this.convResample, dims, outCh)
                    );
                    ds /= 2;
                }
                this.outputBlocks.push(new TimestepEmbedSequential(...layers));
                this._featureSize += ch;
            }
        }

        this.out = new Sequential(
            normalization(ch),
            new SiLU(),
            zeroModule(
                conv_nd(
                    dims,
                    this.modelChannels,
                    this.outChannels,
                    3,
                    1,
                    1,
                    this.dtype
                )
            ),
        );
    }

    /**
     * Apply the model to the input batch.
     * @param x a [B, C, ...] Tensor of inputs.
     * @param timesteps a 1-D batch of timesteps.
     * @param context conditioning from cross-attention.
     * @param y a [B] Tensor of labels, if cross-conditional.
     * @returns a [B, C, ...] Tensor of outputs.
     */
    forward(x: Tensor, timesteps: Tensor, context?: Tensor, y?: Tensor) {
        if ((y !== undefined) != (this.numClasses !== null)) {
            throw new Error(
                "Must specify y if and only if numClasses is set (the model is cross-conditional)"
            );
        }
        const hs: Tensor[] = [];
        const tEmb = timestepEmbedding(timesteps, this.modelChannels, this.dtype, false);
        const emb = this.timeEmbed.forward(tEmb);

        if (this.numClasses !== null) {
            if (y?.shape.length != 1 || y!.shape[0] != x.shape[0]) {
                throw new Error(
                    "y must be a 1-D batch of labels whose batch size matches x"
                );
            }
        }

        let h: Tensor = x;
        for (const module of this.inputBlocks) {
            h = (module as any).forward(h, emb, context);
            hs.push(h);
        }
        h = this.middleBlock.forward(h, emb, context);
        for (const module of this.outputBlocks) {
            h = cat([h, hs.pop()!], 1);
            h = (module as any).forward(h, emb, context);
        }
        return this.out.forward(h);
    }
}

function conv_nd(
    dims: number,
    inChannels: number,
    outChannels: number,
    kernelSize: number,
    stride: number,
    padding: number | [number, number] | "valid" | "same",
    dtype: Dtype
): Conv2d {
    if (dims === 2) {
        return new Conv2d(
            inChannels,
            outChannels,
            kernelSize,
            stride,
            padding,
            dtype
        );
    } else {
        throw new Error(`conv_nd: dims ${dims} not implemented`);
    }
}

function linear(inChannels: number, outChannels: number): Linear {
    return new Linear(inChannels, outChannels);
}

function normalization(channels: number): GroupNorm32 {
    return new GroupNorm32(32, channels);
}

function timestepEmbedding(timesteps: Tensor, dim: number, dtype: Dtype, repeatOnly: boolean = false, maxPeriod: number = 10000): Tensor {
    let embedding: Tensor;
    if (!repeatOnly) {
        // const half = dim / 2;
        // const freqs =
        //     exp(div(mul(-Math.log(maxPeriod), arange(0, half, dtype)),
        //             half));
        // const args = timesteps[:, None].float() * freqs[None];
        // embedding = cat([cos(args), sin(args)], -1);
        // if (dim % 2 == 1) {
        //     embedding = cat([embedding, zerosLike([embedding.shape[0], 1], dtype)], -1);
        // }
        throw new Error("Not implemented due to missing arange, zerosLike, slicing");
    }
    else {
        // embedding = repeat(timesteps, "b -> b d", {d: dim});
        throw new Error("Not implemented due to missing repeat");
    }
}

function zeroModule<T extends Module>(module: T): T {
    for (const p of module.parameters()) {
        p.detach().zero_();
    }
    return module;
}

class AttentionBlock extends Module {
    constructor(
        channels: number,
        numHeads: number,
        numHeadChannels: number,
        useCheckpoint: boolean,
        useNewAttentionOrder: boolean
    ) {
        super();
    }
}

class Downsample extends Module {
    constructor(
        channels: number,
        useConv: boolean,
        dims: number = 2,
        outChannels?: number,
        padding: number = 1
    ) {
        super();
    }
}

class Upsample extends Module {
    constructor(
        channels: number,
        useConv: boolean,
        dims: number = 2,
        outChannels?: number,
        padding: number = 1
    ) {
        super();
    }
}

class GroupNorm32 extends GroupNorm {
    constructor(numGroups: number, channels: number) {
        super(numGroups, channels);
    }
}

class SpatialTransformer extends Module {
    constructor(
        inChannels: number,
        nHeads: number,
        dHead: number,
        depth: number = 1,
        dropout: number = 0,
        contextDim: number | null = null
    ) {
        super();
    }
}

class TimestepBlock extends Module {}

/**
 * A sequential module that passes timestep embeddings to the children that
 * support it as an extra input.
 */
export class TimestepEmbedSequential extends TimestepBlock {
    constructor(...modules: Module[]) {
        super();
        for (const [i, module] of modules.entries()) {
            this.addModule(i, module);
        }
    }
    forward(x: Tensor, emb: Tensor, context?: Tensor) {
        for (const layer of this.children) {
            if (layer instanceof TimestepBlock) {
                x = (layer as any).forward(x, emb);
            }
            // else if (layer instanceof SpatialTransformer) {
            //     x = (layer as any)(x, context);
            // }
            else {
                x = (layer as any)(x);
            }
        }
        return x;
    }
}

class ResBlock extends TimestepBlock {
    constructor(
        channels: number,
        embChannels: number,
        outChannels: number,
        dims = 2,
        up = false,
        down = false,
        dropout: number = 0.0,
        useCheckpoint = false,
        useConv = false,
        useScaleShiftNorm = false
    ) {
        super();
    }
}
