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
    const imageHeight = 480;
    const imageWidth = imageHeight * plotAspect;
    const plotLineWidth = imageHeight * 0.01;
    const fontSize = imageHeight * 0.04;
    const plotFrameWidth = imageWidth - 4 * fontSize;
    const plotFrameHeight = imageHeight - 4 * fontSize;
    const plotFrameX = (imageWidth - plotFrameWidth) / 2;
    const plotFrameY = (imageHeight - plotFrameHeight) / 2;
    const ds = plotFrameWidth / (samples.input.length - 1);
    const dx = plotFrameWidth / (bounds.h.max - bounds.h.min);
    const dy = plotFrameHeight / (bounds.v.max - bounds.v.min);
    function projectSamplePoint(sampleIndex: number, value: number): [number, number] {
        const x = plotFrameX + sampleIndex * ds;
        let y = plotFrameY + plotFrameHeight - (value - bounds.v.min) * dy;
        if (Number.isNaN(y))
            y = 0.0;
        return [x, y];
    }
    function projectPoint(x: number, y: number): [number, number] {
        return [plotFrameX + (x - bounds.h.min) * dx, plotFrameY + plotFrameHeight - (y - bounds.v.min) * dy];
    }
    const w = new CodeWriter();
    const plotFillColor = "rgba(128, 128, 128, 0.1)";
    const textColor = "rgba(128, 128, 128, 0.75)";
    const gridColor = "rgba(128, 128, 128, 0.25)";
    w.writeLine(`<?xml version="1.0" encoding="UTF-8" standalone="no"?>`);
    w.writeLine(`<svg width="${imageWidth}" height="${imageHeight}" viewBox="0 0 ${imageWidth} ${imageHeight}" xmlns="http://www.w3.org/2000/svg">`);
    w.writeLine(`<rect x="${plotFrameX}" y="${plotFrameY}" width="${plotFrameWidth}" height="${plotFrameHeight}" fill="${plotFillColor}" stroke="${textColor}" stroke-width="${plotLineWidth/4}"/>`);
    w.writeLine(`<text x="${plotFrameX + plotFrameWidth / 2}" y="${plotFrameY + plotFrameHeight + fontSize}" text-anchor="middle" font-family="sans-serif" font-size="${fontSize}" fill="${textColor}">input</text>`);
    w.writeLine(`<text x="${plotFrameX - fontSize/2}" y="${plotFrameY + plotFrameHeight / 2}" text-anchor="middle" font-family="sans-serif" font-size="${fontSize}" transform="rotate(-90, ${plotFrameX - fontSize/2}, ${plotFrameY + plotFrameHeight / 2})" fill="${textColor}">${op.name}</text>`);
    w.writeLine(`<clipPath id="plotClip">`);
    w.writeLine(`<rect x="${plotFrameX}" y="${plotFrameY}" width="${plotFrameWidth}" height="${plotFrameHeight}"/>`);
    w.writeLine(`</clipPath>`);
    // Draw the grid
    const gridStep = 10 ** Math.ceil(Math.log10(plotWidth / 10))/2;
    const gridStepX = plotFrameWidth / plotWidth * gridStep;
    const gridStepY = plotFrameHeight / plotHeight * gridStep;
    w.writeLine(`<g clip-path=\"url(#plotClip)\" stroke="${gridColor}" stroke-width="${plotLineWidth / 4}" stroke-dasharray="${plotLineWidth / 2} ${plotLineWidth / 2}">`);
    for (let y = Math.ceil(bounds.v.min / gridStep) * gridStep; y <= bounds.v.max; y += gridStep) {
        const [x1, y1] = projectPoint(bounds.h.min, y);
        const [x2, y2] = projectPoint(bounds.h.max, y);
        const value = y.toFixed(2);
        if (Math.abs(y) < 1e-6) {
            w.writeLine(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke-width="${plotLineWidth / 2}" stroke-dasharray="none"/>`);
        }
        else {
            w.writeLine(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`);
        }
        w.writeLine(`<text x="${plotFrameX + fontSize/6}" y="${y1 + fontSize/2}" text-anchor="left" font-family="sans-serif" font-size="${fontSize/2}" fill="${gridColor}" stroke-dasharray="none">${value}</text>`);
    }
    for (let x = Math.ceil(bounds.h.min / gridStep) * gridStep; x <= bounds.h.max; x += gridStep) {
        const [x1, y1] = projectPoint(x, bounds.v.min);
        const [x2, y2] = projectPoint(x, bounds.v.max);
        const value = x.toFixed(2);
        if (Math.abs(x) < 1e-6) {
            w.writeLine(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke-width="${plotLineWidth / 2}" stroke-dasharray="none"/>`);
        }
        else {
            w.writeLine(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`);
        }
        w.writeLine(`<text x="${x1 + fontSize/6}" y="${plotFrameY + plotFrameHeight - fontSize/6}" text-anchor="start" font-family="sans-serif" font-size="${fontSize/2}" fill="${gridColor}" stroke-dasharray="none">${value}</text>`);
    }
    w.writeLine(`</g>`);
    // Draw the plots
    w.writeLine(`<g>`);
    function makePath(samples: Samples, color: string): string {
        let path = "<path clip-path=\"url(#plotClip)\" d=\"";
        let needsMove = true;
        let lastPoint = [0, 0];
        for (let sampleIndex = 0; sampleIndex < samples.length; sampleIndex++) {
            const [x, y] = projectSamplePoint(sampleIndex, samples[sampleIndex]);
            if (needsMove) {
                path += `M ${x} ${y}`;
                needsMove = false;
            } else {
                path += `L ${x} ${y}`;
            }
        }
        path += `" stroke="${color}" stroke-width="${plotLineWidth}" fill="none"/>`;
        return path;
    }
    // w.writeLine(`${makePath(samples.input, "blue")}`);
    w.writeLine(`${makePath(samples.inputGrad, "red")}`);
    w.writeLine(`${makePath(samples.output, "green")}`);
    w.writeLine(`</g>`);
    w.writeLine(`</svg>`);

    const plotFileName = `${plotsDir}/${op.name}.svg`;
    writeFile(plotFileName, w.toString());
}

function writeFile(path: string, code: string) {
    const oldCode = fs.existsSync(path) ? fs.readFileSync(path, { encoding: "utf8" }) : null;
    if (oldCode === code) {
        // console.log("OK", path);
    }
    else {
        console.log("Writing", path);
        fs.writeFileSync(path, code, { encoding: "utf8" });
    }
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
        const outputArray = await outputTensor.toArrayAsync() as number[];
        outputSamples.push(outputArray[0]);
        const inputGradArray = await inputTensor.grad!.toArrayAsync() as number[];
        inputGradSamples.push(inputGradArray[0]);
    }
    const results: OpSamples = { input: inputSamples, output: outputSamples, inputGrad: inputGradSamples };
    return results;
}

async function writePlots() {
    const boundsVertical = 4.0;
    const boundsHorizontal = boundsVertical * 4 / 3;
    const bounds = { h: { min: -boundsHorizontal/2, max: boundsHorizontal/2 }, v: { min: -boundsVertical/2, max: boundsVertical/2 } };
    for (const op of opRegistry) {
        try {
            if (op.type === "unary") {
                const samples = await sampleUnaryOp(op as UnaryOpSpec, 128, bounds);
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
