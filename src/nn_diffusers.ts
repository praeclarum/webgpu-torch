import { Dtype } from "./dtype";
import { Module } from "./nn_module";

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
    // dims: number;
    numClasses: number | null;
    useCheckpoint: boolean;
    dtype: Dtype;
    numHeads: number;
    numHeadChannels: number;
    numHeadsUpSample: number;
    // useScaleShiftNorm: boolean;
    // resblockUpdown: boolean;

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
    }
}
