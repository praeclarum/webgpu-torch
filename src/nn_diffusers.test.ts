import { UNetModel } from "./nn_diffusers";

test("can build unet without spatial", () => {
    const m = new UNetModel({
        inChannels: 3,
        modelChannels: 256,
        outChannels: 3,
        numHeads: 1,
        attentionResolutions: [2],
    });
    expect(m.outChannels).toEqual(3);
});

test("can build unet with spatial", () => {
    const m = new UNetModel({
        inChannels: 3,
        modelChannels: 256,
        outChannels: 3,
        numHeads: 1,
        attentionResolutions: [2],
        useSpatialTransformer: true,
    });
    expect(m.outChannels).toEqual(3);
});
