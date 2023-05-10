import { CodeWriter } from "../src/opgen";
import { OpSpec, ReductionOpSpec, UnaryOpSpec } from "../src/op_spec";
import { registry as opRegistry } from "../src/op_table";
import { tensor } from "../src/ops_artisanal";
import { Tensor } from "../src/tensor";

// import fs
import * as fs from "fs";

console.log("Running plot generator...");

let plotsDir = __dirname + "/../web/plots";
if (!fs.existsSync(plotsDir)) {
    fs.mkdirSync(plotsDir);
}
plotsDir = fs.realpathSync(plotsDir);
console.log("plots dir:", plotsDir);

type AxisBounds = {min: number, max: number};
type PlotBounds = {h: AxisBounds, v: AxisBounds};
type Samples = number[];
type OpSamples = { input: Samples, output: Samples, inputGrad: Samples };

function writePlotSvg(op: OpSpec, samples: OpSamples, bounds: PlotBounds): void {
    const plotWidth = bounds.h.max - bounds.h.min;
    const plotHeight = bounds.v.max - bounds.v.min;
    const plotAspect = plotWidth / plotHeight;
    const imageHeight = 720;
    const imageWidth = imageHeight * plotAspect;
    const plotFrameWidth = imageWidth * 0.9;
    const plotFrameHeight = imageHeight * 0.9;
    const plotFrameX = (imageWidth - plotFrameWidth) / 2;
    const plotFrameY = (imageHeight - plotFrameHeight) / 2;
    const w = new CodeWriter();
    w.writeLine(`<?xml version="1.0" encoding="UTF-8" standalone="no"?>`);
    w.writeLine(`<svg width="${imageWidth}" height="${imageHeight}" viewBox="0 0 ${imageWidth} ${imageHeight}" xmlns="http://www.w3.org/2000/svg">`);
    w.writeLine(`<rect x="${plotFrameX}" y="${plotFrameY}" width="${plotFrameWidth}" height="${plotFrameHeight}" fill="white" stroke="black" stroke-width="1"/>`);
    w.writeLine(`<path d="M ${plotFrameX} ${plotFrameY + plotFrameHeight} L ${plotFrameX + plotFrameWidth} ${plotFrameY + plotFrameHeight}" stroke="black" stroke-width="1"/>`);
    w.writeLine(`<path d="M ${plotFrameX} ${plotFrameY} L ${plotFrameX} ${plotFrameY + plotFrameHeight}" stroke="black" stroke-width="1"/>`);
    w.writeLine(`<text x="${plotFrameX + plotFrameWidth / 2}" y="${plotFrameY + plotFrameHeight + 20}" text-anchor="middle" font-family="Verdana" font-size="20">${op.name}</text>`);
    w.writeLine(`<text x="${plotFrameX - 20}" y="${plotFrameY + plotFrameHeight / 2}" text-anchor="middle" font-family="Verdana" font-size="20" transform="rotate(-90, ${plotFrameX - 20}, ${plotFrameY + plotFrameHeight / 2})">${op.name}</text>`);
    const dx = plotFrameWidth / (samples.input.length - 1);
    const dy = plotFrameHeight / (bounds.v.max - bounds.v.min);
    function projectPoint(sampleIndex: number, value: number): [number, number] {
        const x = plotFrameX + sampleIndex * dx;
        const y = plotFrameY + plotFrameHeight - (value - bounds.v.min) * dy;
        return [x, y];
    }
    function makePath(samples: Samples, color: string): string {
        let path = "<path d=\"";
        for (let sampleIndex = 0; sampleIndex < samples.length; sampleIndex++) {
            const [x, y] = projectPoint(sampleIndex, samples[sampleIndex]);
            if (sampleIndex == 0) {
                path += `M ${x} ${y}`;
            } else {
                path += `L ${x} ${y}`;
            }
        }
        path += `" stroke="${color}" stroke-width="1" fill="none"/>`;
        return path;
    }
    w.writeLine(`${makePath(samples.input, "blue")}`);
    w.writeLine(`${makePath(samples.output, "red")}`);
    w.writeLine(`${makePath(samples.inputGrad, "green")}`);
    w.writeLine(`</svg>`);

    const plotFileName = `${plotsDir}/${op.name}.svg`;
    fs.writeFileSync(plotFileName, w.toString());
}

async function sampleUnaryOp(op: UnaryOpSpec, numSamples: number, bounds: PlotBounds): Promise<OpSamples> {
    const inputSamples: Samples = []
    const otherSamples: Samples = []
    const outputSamples: Samples = []
    const inputGradSamples: Samples = []
    const otherGradSamples: Samples = []
    const dinput = (bounds.h.max - bounds.h.min)/(numSamples - 1);
    for (let sampleIndex = 0; sampleIndex < numSamples; sampleIndex++) {
        const inputValue = bounds.h.min + sampleIndex * dinput;
        inputSamples.push(inputValue);
        const inputTensor = tensor({data:[inputValue], requiresGrad: true});
        // console.log("input:", inputTensor);
        const outputTensor = (inputTensor as any)[op.name]() as Tensor;
        outputTensor.backward();
        const outputArray = await outputTensor.toArrayAsync();
        outputSamples.push(outputArray[0] as number);
        const inputGradArray = await inputTensor.grad!.toArrayAsync();
        inputGradSamples.push(inputGradArray[0] as number);
    }
    const results: OpSamples = { input: inputSamples, output: outputSamples, inputGrad: inputGradSamples };
    return results;
}

async function writePlots() {
    const bounds = { h: { min: -5, max: 5 }, v: { min: -5, max: 5 } };
    for (const op of opRegistry) {
        try {
            if (op.type === "unary") {
                const samples = await sampleUnaryOp(op as UnaryOpSpec, 100, bounds);
                writePlotSvg(op, samples, bounds);
            }
        } catch (e) {
            console.log("error:", e);
        }
    }
}
(async () => {
    try {
      await writePlots();
    } catch (error) {
      console.error("Error caught:", error);
      process.exit(1);
    }
  })();
