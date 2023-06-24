import { Module } from "./nn_module";

export class GroupNorm extends Module {
    numGroups: number;
    numChannels: number;
    constructor(numGroups: number, numChannels: number) {
        super();
        this.numGroups = numGroups;
        this.numChannels = numChannels;
    }
}
