import { Dtype } from "./dtype";
import { Conv2d, Linear } from "./nn_cnn";
import { Module, ModuleList, Sequential } from "./nn_module";
import { SiLU } from "./nn_opgen";
import { Tensor } from "./tensor";

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
        this.timeEmbed = new Sequential([
            linear(this.modelChannels, timeEmbedDim),
            new SiLU(),
            linear(timeEmbedDim, timeEmbedDim),
        ]);

        this.inputBlocks = new ModuleList([
            new TimestepEmbedSequential(
                conv_nd(
                    dims,
                    this.inChannels,
                    this.modelChannels,
                    3,
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
    }
}

function conv_nd(
    dims: number,
    inChannels: number,
    outChannels: number,
    kernelSize: number,
    stride: number,
    dtype: Dtype
): Conv2d {
    if (dims === 2) {
        return new Conv2d(
            inChannels,
            outChannels,
            kernelSize,
            stride,
            "same",
            dtype
        );
    } else {
        throw new Error(`conv_nd: dims ${dims} not implemented`);
    }
}

function linear(inChannels: number, outChannels: number): Linear {
    return new Linear(inChannels, outChannels);
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
class TimestepEmbedSequential extends TimestepBlock {
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
