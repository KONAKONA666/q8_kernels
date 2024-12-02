// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once


#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()



#define BATCH_SWITCH(BATCH_A, BATCH_B, ...)     \
  [&] {                                         \
    if (BATCH_A == 1) {                         \
        constexpr static bool BA_ = false;       \
        constexpr static bool BB_ = true;        \
        return __VA_ARGS__();                   \
    } else if(BATCH_B == 1) {                   \
        constexpr static bool BA_ = true;        \
        constexpr static bool BB_ = false;       \
        return __VA_ARGS__();                   \
    } else {                                    \
        constexpr static bool BA_ = true;       \
        constexpr static bool BB_ = true;       \
        return __VA_ARGS__();                   \
    }                                           \
  }()                                           


