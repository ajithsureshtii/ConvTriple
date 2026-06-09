#ifndef FC_GPU_CUH_
#define FC_GPU_CUH_

#include <cstdint>
#include <vector>

#include "io/net_io_channel.hpp"

using std::vector;

namespace TROY {

using INT_TYPE = uint32_t;

// AB2 protocol: ALICE encrypts input, BOB holds weight (standard direction)
void fc_ab2(IO::NetIO** ios, int party, const INT_TYPE* x, const INT_TYPE* w, INT_TYPE* c,
            size_t batch, size_t in_dim, size_t out_dim, int device_id = 0);

// AB2 reverse: BOB encrypts weight, ALICE holds input
void fc_ab2_reverse(IO::NetIO** ios, int party, const INT_TYPE* x, const INT_TYPE* w, INT_TYPE* c,
                    size_t batch, size_t in_dim, size_t out_dim, int device_id = 0);

// AB protocol: both parties hold additive shares of both input and weight
void fc_ab(IO::NetIO** ios, int party, const INT_TYPE* x, const INT_TYPE* w, INT_TYPE* c,
           size_t batch, size_t in_dim, size_t out_dim, int device_id = 0);

// AB reverse: both parties hold additive shares, weight-side encrypts
void fc_ab_reverse(IO::NetIO** ios, int party, const INT_TYPE* x, const INT_TYPE* w, INT_TYPE* c,
                   size_t batch, size_t in_dim, size_t out_dim, int device_id = 0);

// Top-level dispatcher — called from hpmpc_interface.cpp
void fc(IO::NetIO** ios, int party, const INT_TYPE* x, const INT_TYPE* w, INT_TYPE* c,
        size_t batch, size_t in_dim, size_t out_dim,
        int factor = 1, bool is_ab = false, int device_id = 0);

// Ideal (plaintext) FC for VERIFY mode: c[b*out_dim + j] = sum_i x[b*in_dim+i] * w[j*in_dim+i] mod t
vector<INT_TYPE> ideal_fc(const INT_TYPE* x, const INT_TYPE* w, size_t t,
                           size_t batch, size_t in_dim, size_t out_dim);

} // namespace TROY

#endif
