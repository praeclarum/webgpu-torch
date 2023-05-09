import { Dtype } from "./dtype";
import { Conv2d, Linear } from "./nn_cnn";
import { Module, ModuleList, Sequential } from "./nn_module";
import { SiLU } from "./nn_opgen";
import { Tensor } from "./tensor";

/**
 * Full UNet model with attention and timestep embedding.
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
    numHeads: number;
    numHeadChannels: number;
    numHeadsUpSample: number;

    private _featureSize: number;

    timeEmbed: Sequential;
    inputBlocks: ModuleList;

    /**
     * Constructs a new UNet model with a given number of input and output channels along with
     * a variety of other options.
     * @param inChannels channels of input tensor
     * @param modelChannels base channels in model
     * @param outChannels channels of output tensor
     * @param numResBlocks number of residual blocks per down/up sampling stage
     * @param attentionResolutions a collection of downsample rates at which attention is applied.
     *                           For example, if this contains 4, then at 4x downsampling, attention is applied.
     * @param dropout the dropout probability
     * @param channelMult channel multiples per down/up sampling stage
     * @param convResample if `true` use learnable convolutional upsampling/downsampling
     * @param dims determines whether to use 1D, 2D, or 3D convolutions
     * @param numClasses if specified, then this model will be class-conditioned with `numClasses` classes
     * @param numHeads the number of attention heads in each attention layer
     * @param numHeadChannels if specified, ignore numHeads and instead use this number of channels in each attention head
     * @param useScaleShiftNorm use a FiLM-like conditioning mechanism
     * @param resblockUpdown use residual blocks for up/down sampling
     */
    constructor(
        inChannels: number,
        modelChannels: number,
        outChannels: number,
        numResBlocks: number,
        attentionResolutions: number[],
        dropout: number = 0,
        channelMult: number[] = [1, 2, 4, 8],
        convResample: boolean = true,
        dims: number = 2,
        numClasses: number | null = null,
        useCheckpoint: boolean = false,
        dtype: Dtype = "float32",
        numHeads: number = -1,
        numHeadChannels: number = -1,
        numHeadsUpSample: number = -1,
        useScaleShiftNorm: boolean = false,
        resblockUpdown: boolean = false
    ) {
        super();

        if (numHeads === -1) {
            if (numHeadChannels === -1) {
                throw new Error(
                    `Must specify either numHeads or numHeadChannels`
                );
            }
        }
        if (numHeadChannels === -1) {
            if (numHeads === -1) {
                throw new Error(
                    `Must specify either numHeads or numHeadChannels`
                );
            }
        }

        this.inChannels = inChannels;
        this.modelChannels = modelChannels;
        this.outChannels = outChannels;
        this.numResBlocks = numResBlocks;
        this.attentionResolutions = attentionResolutions;
        this.dropout = dropout;
        this.channelMult = channelMult;
        this.convResample = convResample;
        this.numClasses = numClasses;
        this.useCheckpoint = useCheckpoint;
        this.dtype = dtype;
        this.numHeads = numHeads;
        this.numHeadChannels = numHeadChannels;
        this.numHeadsUpSample = numHeadsUpSample;

        const timeEmbedDim = modelChannels * 4;
        this.timeEmbed = new Sequential([
            linear(modelChannels, timeEmbedDim),
            new SiLU(),
            linear(timeEmbedDim, timeEmbedDim),
        ]);

        this.inputBlocks = new ModuleList([
            new TimestepEmbedSequential([
                conv_nd(dims, inChannels, modelChannels, 3, 1, this.dtype),
            ]),
        ]);
        this._featureSize = modelChannels;
        const inputBlockChans = [modelChannels];
        let ch = modelChannels;
        let ds = 1;
        for (const [level, mult] of channelMult.entries()) {
            for (let i = 0; i < numResBlocks; i++) {
                const layers = [
                    // new ResBlock(
                    //     ch,
                    //     timeEmbedDim,
                    //     dropout,
                    //     mult * modelChannels,
                    //     dims,
                    //     useCheckpoint,
                    //     useScaleShiftNorm)
                ];
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

class TimestepBlock extends Module {}

/**
 * A sequential module that passes timestep embeddings to the children that
 * support it as an extra input.
 */
class TimestepEmbedSequential extends TimestepBlock {
    constructor(modules: Module[]) {
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
