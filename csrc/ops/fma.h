#pragma once

struct FMABaseParams {
    using index_t = int64_t;

    int batch;

    index_t x_batch_stride;

    index_t y_batch_stride;
    index_t z_batch_stride;
    
    index_t out_batch_stride;
    

    index_t y_change_batch_every;
    index_t z_change_batch_every;
    
    int dim;

    float scale_add;
    
    void *__restrict__ x_ptr;
    void *__restrict__ y_ptr;
    void *__restrict__ z_ptr;
    void *__restrict__ out_ptr;
    
};

