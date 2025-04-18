// signature_check.c

#include <stdbool.h>
#include <c.h>
#include <storage/pmsignal.h>

// If the real signature ever changes, the assignment here will error:
static bool (*_check_sig)() = &PostmasterIsAliveInternal;

