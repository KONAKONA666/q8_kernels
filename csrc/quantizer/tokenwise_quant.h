#pragma once


struct QuantizerParamsBase {
    using index_t = int64_t;

    int batch, dim;

    index_t x_batch_stride;
    index_t out_batch_stride;
    index_t out_scales_batch_stride;
    // Common data pointers.
    void *__restrict__ x_ptr;
    void *__restrict__ out_ptr;
    void *__restrict__ out_scales_ptr;
};

