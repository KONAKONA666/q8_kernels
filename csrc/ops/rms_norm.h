#pragma once


struct RMSNormsParamsBase {
    using index_t = int64_t;

    int batch, dim;
    float eps;

    index_t x_batch_stride;
    index_t weights_stride;
    index_t out_batch_stride;
    index_t out_scales_stride;
    // Common data pointers.
    void *__restrict__ x_ptr;
    void *__restrict__ weights_ptr;
    void *__restrict__ out_ptr;
    void *__restrict__ out_scales_ptr;
};

struct RMSNormsBackwardParamsBase {
    using index_t = int64_t;

    int batch, dim;
    float eps;

    index_t x_normed_batch_stride;
    index_t x_norms_batch_stride;

    index_t weights_stride;
    index_t grad_out_batch_stride;

    index_t out_batch_stride;
    // Common data pointers.
    
    void *__restrict__ x_normed_ptr;
    void *__restrict__ x_norms_ptr;
    void *__restrict__ weights_ptr;
    
    void *__restrict__ grad_out_ptr;

    void *__restrict__ out_ptr;
};

