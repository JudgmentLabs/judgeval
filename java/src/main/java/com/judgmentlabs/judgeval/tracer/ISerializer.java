package com.judgmentlabs.judgeval.tracer;

import java.lang.reflect.Type;

/**
 * @deprecated Use {@link com.judgmentlabs.judgeval.v1.tracer.ISerializer}
 *             instead.
 */
@Deprecated
public interface ISerializer {
    String serialize(Object obj);

    default String serialize(Object obj, Type type) {
        return serialize(obj);
    }
}
