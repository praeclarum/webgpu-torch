export type DeviceType = "cpu" | "webgpu";
export type DeviceId = string;

export type Deviceish = DeviceType | Device | DeviceId;

export class Device {
    private _id: DeviceId;
    private _type: DeviceType;
    get id(): DeviceId {
        return this._id;
    }
    get type(): DeviceType {
        return this._type;
    }
    constructor(id: DeviceId, type: DeviceType) {
        this._id = id;
        this._type = type;
    }
}
