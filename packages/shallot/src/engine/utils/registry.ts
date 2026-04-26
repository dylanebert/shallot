export interface Registry<T> {
    add(data: T, name?: string): number;
    set(id: number, data: T): void;
    get(id: number): T | undefined;
    getByName(name: string): number | undefined;
    getName(id: number): string | undefined;
    all(): T[];
    count(): number;
    clear(): void;
    version: number;
}

export function registry<T>(capacity: number): Registry<T> {
    const items: T[] = [];
    const names = new Map<string, number>();
    const reverseNames = new Map<number, string>();
    let version = 0;

    return {
        add(data: T, name?: string): number {
            if (name) {
                const existing = names.get(name);
                if (existing !== undefined) {
                    items[existing] = data;
                    version++;
                    return existing;
                }
            }
            if (items.length >= capacity) throw new Error(`registry limit reached (${capacity})`);
            const id = items.length;
            items.push(data);
            if (name) {
                names.set(name, id);
                reverseNames.set(id, name);
            }
            version++;
            return id;
        },

        set(id: number, data: T): void {
            if (id < 0 || id >= items.length) throw new Error(`registry set out of bounds: ${id}`);
            items[id] = data;
            version++;
        },

        get(id: number): T | undefined {
            return items[id];
        },

        getByName(name: string): number | undefined {
            return names.get(name);
        },

        getName(id: number): string | undefined {
            return reverseNames.get(id);
        },

        all(): T[] {
            return items;
        },

        count(): number {
            return items.length;
        },

        clear(): void {
            items.length = 0;
            names.clear();
            reverseNames.clear();
            version++;
        },

        get version(): number {
            return version;
        },
    };
}
