import * as nn from './nn';

test("Conv2d can set inChannels and outChannels", () => {
    const conv2d = new nn.Conv2d(3, 4);
    expect(conv2d.inChannels).toBe(3);
    expect(conv2d.outChannels).toBe(4);
});

test("Conv2d can set inChannels and outChannels", () => {
    const conv2d = new nn.Conv2d(3, 4);
    expect(conv2d.inChannels).toBe(3);
    expect(conv2d.outChannels).toBe(4);
});
