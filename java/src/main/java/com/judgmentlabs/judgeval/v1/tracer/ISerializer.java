package com.judgmentlabs.judgeval.v1.tracer;

import java.lang.reflect.Type;

/**
 * Serializes objects to string representations.
 */
public interface ISerializer {
    /**
     * Serializes an object to a string.
     *
     * @param obj
     *            the object to serialize
     * @return the serialized string
     */
    String serialize(Object obj);

    /**
     * Serializes an object to a string using the specified type.
     *
     * @param obj
     *            the object to serialize
     * @param type
     *            the type to use for serialization
     * @return the serialized string
     */
    default String serialize(Object obj, Type type) {
        return serialize(obj);
    }
}
