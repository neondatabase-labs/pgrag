#pragma once

// get the C _bool_ type
#include <stdbool.h>

// only declare the function you need
#ifdef __cplusplus
extern "C" {
#endif

extern bool PostmasterIsAliveInternal();

#ifdef __cplusplus
}
#endif

