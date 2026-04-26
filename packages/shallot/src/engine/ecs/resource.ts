export interface ResourceContainer {
    getResource<T>(key: Resource<T>): T | undefined;
}

/** typed key for global state */
export type Resource<T> = symbol & {
    __type?: T;
    from(state: ResourceContainer): T | undefined;
};

/**
 * create a typed resource key
 * @example
 * const Volume = resource<number>("volume");
 * state.setResource(Volume, 0.8);
 */
export function resource<T>(name: string): Resource<T> {
    const key = Symbol(name) as symbol;
    const resourceKey = Object.assign(key, {
        from(state: ResourceContainer): T | undefined {
            return state.getResource(resourceKey);
        },
    });
    return resourceKey as Resource<T>;
}
