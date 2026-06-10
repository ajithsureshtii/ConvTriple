#include <chrono>
#include <sstream>

#include "constants.hpp"
#include "fc_gpu.cuh"
#include "conv2d_gpu.cuh"  // reuse setup(), send(), recv(), random_polynomial(), add_inplace()

#include <troy/troy.h>
#include <cuda_runtime.h>

namespace TROY {

static unsigned char FC_ALICE = 1;
static unsigned char FC_BOB   = 2;

// ─────────────────────────────────────────────────────────────────────────────
// AB2 — ALICE encrypts input (EncryptLeft), BOB holds weight
// ─────────────────────────────────────────────────────────────────────────────
void fc_ab2(IO::NetIO** ios, int party, const INT_TYPE* x, const INT_TYPE* w, INT_TYPE* c,
            size_t batch, size_t in_dim, size_t out_dim, int device_id) {
    using namespace troy;
    {
    auto he = setup(device_id);
    linear::PolynomialEncoderRing2k<INT_TYPE> encoder(he, BIT_LEN);
    if (utils::device_count() > 0) {
        he->to_device_inplace();
        encoder.to_device_inplace();
    } else {
        std::cerr << RED << "FC: Couldn't find a GPU" << NC << "\n";
    }

    linear::MatmulHelper helper(batch, in_dim, out_dim, POLY_MOD,
                                linear::MatmulObjective::EncryptLeft, /*pack_lwe=*/false);

    KeyGenerator keygen(he);
    Encryptor encryptor(he);
    encryptor.set_secret_key(keygen.secret_key());
    Evaluator evaluator(he);
    Decryptor decryptor(he, keygen.secret_key());

    // BOB pre-encodes its weight matrix once (NTT on GPU)
    linear::Plain2d w_encoded;
    if (party == FC_BOB) {
        auto ntt = measure::now();
        w_encoded = helper.encode_weights_ring2k(encoder, w, std::nullopt, false);
        double ntt_time = std::chrono::duration<double, std::milli>(measure::now() - ntt).count();
        std::cerr << "P" << party - 1 << ": FC NTT preprocessing time[s]: " << ntt_time / 1000.0
                  << "\n";
    }

    vector<INT_TYPE> R = random_polynomial(batch * out_dim);

    if (party == FC_ALICE) {
        // Encrypt input, send, receive result, decrypt
        linear::Cipher2d x_encrypted
            = helper.encrypt_inputs_ring2k(encryptor, encoder, x, std::nullopt);
        std::stringstream x_serialized;
        x_encrypted.save(x_serialized, he);
        send(ios, x_serialized);

        auto y_serialized = recv(ios);
        auto y_encrypted  = helper.deserialize_outputs(evaluator, y_serialized);
        vector<INT_TYPE> y_decrypted
            = helper.decrypt_outputs_ring2k(encoder, decryptor, y_encrypted);

        for (size_t i = 0; i < batch * out_dim; ++i)
            c[i] = y_decrypted[i];

    } else { // FC_BOB
        // Receive encrypted input, optionally add own share, matmul, sub R, send
        linear::Plain2d R_encoded
            = helper.encode_outputs_ring2k(encoder, R.data(), std::nullopt);

        auto stream      = recv(ios);
        auto x_encrypted = linear::Cipher2d::load_new(stream, he);
        // Received ciphertexts are in seed (compressed) form; matmul calls
        // transform_to_ntt_batched internally which requires the seed expanded.
        x_encrypted.expand_seed(he);

        // If x (BOB's additive input share) is non-null, add it as plaintext
        if (x) {
            linear::Plain2d x_encoded
                = helper.encode_inputs_ring2k(encoder, x, std::nullopt, true);
            x_encrypted.add_plain_inplace(evaluator, x_encoded);
        }

        linear::Cipher2d y_encrypted = helper.matmul(evaluator, x_encrypted, w_encoded);
        y_encrypted.sub_plain_inplace(evaluator, R_encoded);

        std::stringstream y_serialized;
        helper.serialize_outputs(evaluator, y_encrypted, y_serialized);
        send(ios, y_serialized);

        for (size_t i = 0; i < batch * out_dim; ++i)
            c[i] = R[i];
    }
    } // all troy objects destroyed here
    troy::MemoryPool::ReleaseUnused();
}

// ─────────────────────────────────────────────────────────────────────────────
// AB2 reverse — BOB encrypts weight (EncryptRight), ALICE holds input
// ─────────────────────────────────────────────────────────────────────────────
void fc_ab2_reverse(IO::NetIO** ios, int party, const INT_TYPE* x, const INT_TYPE* w, INT_TYPE* c,
                    size_t batch, size_t in_dim, size_t out_dim, int device_id) {
    using namespace troy;
    {
    auto he = setup(device_id);
    linear::PolynomialEncoderRing2k<INT_TYPE> encoder(he, BIT_LEN);
    auto parmsid = encoder.context()->first_context_data_pointer()->parms_id();
    if (utils::device_count() > 0) {
        he->to_device_inplace();
        encoder.to_device_inplace();
    } else {
        std::cerr << RED << "FC: Couldn't find a GPU" << NC << "\n";
    }

    linear::MatmulHelper helper(batch, in_dim, out_dim, POLY_MOD,
                                linear::MatmulObjective::EncryptRight, /*pack_lwe=*/false);

    KeyGenerator keygen(he);
    Encryptor encryptor(he);
    encryptor.set_secret_key(keygen.secret_key());
    Evaluator evaluator(he);
    Decryptor decryptor(he, keygen.secret_key());

    // BOB encrypts weight and sends first
    linear::Cipher2d w_encrypted;
    if (party == FC_BOB) {
        w_encrypted
            = helper.encrypt_weights_ring2k(encryptor, encoder, w, std::nullopt);
        std::stringstream w_serialized;
        w_encrypted.save(w_serialized, he);
        send(ios, w_serialized);
    } else {
        auto stream = recv(ios);
        w_encrypted = linear::Cipher2d::load_new(stream, he);
        w_encrypted.expand_seed(he);
    }

    vector<INT_TYPE> R = random_polynomial(batch * out_dim);

    if (party == FC_BOB) {
        // Receive result from ALICE and decrypt
        auto y_serialized = recv(ios);
        auto y_encrypted  = helper.deserialize_outputs(evaluator, y_serialized);
        vector<INT_TYPE> y_decrypted
            = helper.decrypt_outputs_ring2k(encoder, decryptor, y_encrypted);

        for (size_t i = 0; i < batch * out_dim; ++i)
            c[i] = y_decrypted[i];

    } else { // FC_ALICE
        // Encode own input, matmul_reverse, sub R, send
        linear::Plain2d x_encoded
            = helper.encode_inputs_ring2k(encoder, x, parmsid, false);
        linear::Plain2d R_encoded
            = helper.encode_outputs_ring2k(encoder, R.data(), parmsid);

        linear::Cipher2d y_encrypted
            = helper.matmul_reverse(evaluator, x_encoded, w_encrypted);
        y_encrypted.sub_plain_inplace(evaluator, R_encoded);

        std::stringstream y_serialized;
        helper.serialize_outputs(evaluator, y_encrypted, y_serialized);
        send(ios, y_serialized);

        for (size_t i = 0; i < batch * out_dim; ++i)
            c[i] = R[i];
    }
    } // all troy objects destroyed here
    troy::MemoryPool::ReleaseUnused();
}

// ─────────────────────────────────────────────────────────────────────────────
// AB — both parties hold additive shares of both x and w (EncryptLeft)
// Each encrypts its own x-share, sends, receives other's, computes cross-term
// ─────────────────────────────────────────────────────────────────────────────
void fc_ab(IO::NetIO** ios, int party, const INT_TYPE* x, const INT_TYPE* w, INT_TYPE* c,
           size_t batch, size_t in_dim, size_t out_dim, int device_id) {
    using namespace troy;
    {
    auto he = setup(device_id);
    linear::PolynomialEncoderRing2k<INT_TYPE> encoder(he, BIT_LEN);
    if (utils::device_count() > 0) {
        he->to_device_inplace();
        encoder.to_device_inplace();
    } else {
        std::cerr << RED << "FC: Couldn't find a GPU" << NC << "\n";
    }

    linear::MatmulHelper helper(batch, in_dim, out_dim, POLY_MOD,
                                linear::MatmulObjective::EncryptLeft, /*pack_lwe=*/false);

    KeyGenerator keygen(he);
    Encryptor encryptor(he);
    encryptor.set_secret_key(keygen.secret_key());
    Evaluator evaluator(he);
    Decryptor decryptor(he, keygen.secret_key());

    // Each party pre-encodes its own weight share (NTT on GPU)
    auto ntt      = measure::now();
    linear::Plain2d w_encoded = helper.encode_weights_ring2k(encoder, w, std::nullopt, false);
    double ntt_time = std::chrono::duration<double, std::milli>(measure::now() - ntt).count();
    std::cerr << "P" << party - 1 << ": FC AB NTT preprocessing time[s]: " << ntt_time / 1000.0
              << "\n";

    // Each party encrypts its own x-share
    linear::Cipher2d x_encrypted
        = helper.encrypt_inputs_ring2k(encryptor, encoder, x, std::nullopt);

    vector<INT_TYPE> R = random_polynomial(batch * out_dim);
    linear::Plain2d R_encoded = helper.encode_outputs_ring2k(encoder, R.data(), std::nullopt);

    // Exchange encrypted x-shares
    std::stringstream x_serialized;
    x_encrypted.save(x_serialized, he);
    std::stringstream received_x_serialized;
    if (party == FC_ALICE) {
        send(ios, x_serialized);
        received_x_serialized = recv(ios);
    } else {
        received_x_serialized = recv(ios);
        send(ios, x_serialized);
    }
    auto other_x_encrypted = linear::Cipher2d::load_new(received_x_serialized, he);
    other_x_encrypted.expand_seed(he);

    // Compute: matmul(other_x, own_w) - R
    linear::Cipher2d y_encrypted = helper.matmul(evaluator, other_x_encrypted, w_encoded);
    y_encrypted.sub_plain_inplace(evaluator, R_encoded);

    std::stringstream y_serialized;
    helper.serialize_outputs(evaluator, y_encrypted, y_serialized);

    // Exchange results
    std::stringstream received_y_serialized;
    if (party == FC_ALICE) {
        send(ios, y_serialized);
        received_y_serialized = recv(ios);
    } else {
        received_y_serialized = recv(ios);
        send(ios, y_serialized);
    }

    auto other_y_encrypted = helper.deserialize_outputs(evaluator, received_y_serialized);
    vector<INT_TYPE> y_decrypted
        = helper.decrypt_outputs_ring2k(encoder, decryptor, other_y_encrypted);

    // c = decrypt(other's result) + own R
    add_inplace(y_decrypted, R, PLAIN_MOD);

    for (size_t i = 0; i < batch * out_dim; ++i)
        c[i] = y_decrypted[i];

#ifdef VERIFY
    if (party == FC_ALICE) {
        std::cout << PURPLE << "Verifying FC AB" << NC << "\n";
        std::vector<INT_TYPE> x2(batch * in_dim);
        std::vector<INT_TYPE> w2(out_dim * in_dim);
        std::vector<INT_TYPE> c2(batch * out_dim);
        ios[0]->recv_data(x2.data(), x2.size() * sizeof(INT_TYPE));
        ios[0]->recv_data(w2.data(), w2.size() * sizeof(INT_TYPE));
        ios[0]->recv_data(c2.data(), c2.size() * sizeof(INT_TYPE));
        add_inplace(x2, x, PLAIN_MOD);
        add_inplace(c2, c, PLAIN_MOD);
        add_inplace(w2, w, PLAIN_MOD);
        vector<INT_TYPE> ideal = ideal_fc(x2.data(), w2.data(), PLAIN_MOD, batch, in_dim, out_dim);
        if (vector_equal(c2, ideal))
            std::cout << GREEN << "GPU-FC AB: PASSED" << NC << "\n";
        else
            std::cout << RED << "GPU-FC AB: FAILED" << NC << "\n";
    } else {
        ios[0]->send_data(x, batch * in_dim * sizeof(INT_TYPE));
        ios[0]->send_data(w, out_dim * in_dim * sizeof(INT_TYPE));
        ios[0]->send_data(c, batch * out_dim * sizeof(INT_TYPE));
        ios[0]->flush();
    }
#endif
    } // all troy objects destroyed here
    troy::MemoryPool::ReleaseUnused();
}

// ─────────────────────────────────────────────────────────────────────────────
// AB reverse — both parties hold shares; each encrypts own weight (EncryptRight)
// ─────────────────────────────────────────────────────────────────────────────
void fc_ab_reverse(IO::NetIO** ios, int party, const INT_TYPE* x, const INT_TYPE* w, INT_TYPE* c,
                   size_t batch, size_t in_dim, size_t out_dim, int device_id) {
    using namespace troy;
    {
    auto he = setup(device_id);
    linear::PolynomialEncoderRing2k<INT_TYPE> encoder(he, BIT_LEN);
    auto parmsid = encoder.context()->first_context_data_pointer()->parms_id();
    if (utils::device_count() > 0) {
        he->to_device_inplace();
        encoder.to_device_inplace();
    } else {
        std::cerr << RED << "FC: Couldn't find a GPU" << NC << "\n";
    }

    linear::MatmulHelper helper(batch, in_dim, out_dim, POLY_MOD,
                                linear::MatmulObjective::EncryptRight, /*pack_lwe=*/false);

    KeyGenerator keygen(he);
    Encryptor encryptor(he);
    encryptor.set_secret_key(keygen.secret_key());
    Evaluator evaluator(he);
    Decryptor decryptor(he, keygen.secret_key());

    // Each party encrypts its own weight share
    linear::Cipher2d w_encrypted
        = helper.encrypt_weights_ring2k(encryptor, encoder, w, std::nullopt);

    std::stringstream w_serialized;
    w_encrypted.save(w_serialized, he);

    // Exchange encrypted weight shares
    std::stringstream received_w_serialized;
    if (party == FC_ALICE) {
        send(ios, w_serialized);
        received_w_serialized = recv(ios);
    } else {
        received_w_serialized = recv(ios);
        send(ios, w_serialized);
    }
    auto other_w_encrypted = linear::Cipher2d::load_new(received_w_serialized, he);
    other_w_encrypted.expand_seed(he);

    vector<INT_TYPE> R = random_polynomial(batch * out_dim);
    linear::Plain2d x_encoded
        = helper.encode_inputs_ring2k(encoder, x, parmsid, false);
    linear::Plain2d R_encoded
        = helper.encode_outputs_ring2k(encoder, R.data(), parmsid);

    // Compute: matmul_reverse(own_x, other_w) - R
    linear::Cipher2d y_encrypted
        = helper.matmul_reverse(evaluator, x_encoded, other_w_encrypted);
    y_encrypted.sub_plain_inplace(evaluator, R_encoded);

    std::stringstream y_serialized;
    helper.serialize_outputs(evaluator, y_encrypted, y_serialized);

    // Exchange results
    std::stringstream received_y_serialized;
    if (party == FC_ALICE) {
        received_y_serialized = recv(ios);
        send(ios, y_serialized);
    } else {
        send(ios, y_serialized);
        received_y_serialized = recv(ios);
    }

    auto other_y_encrypted = helper.deserialize_outputs(evaluator, received_y_serialized);
    vector<INT_TYPE> y_decrypted
        = helper.decrypt_outputs_ring2k(encoder, decryptor, other_y_encrypted);

    add_inplace(y_decrypted, R, PLAIN_MOD);

    for (size_t i = 0; i < batch * out_dim; ++i)
        c[i] = y_decrypted[i];

#ifdef VERIFY
    if (party == FC_BOB) {
        std::cout << PURPLE << "Verifying FC AB REVERSED" << NC << "\n";
        std::vector<INT_TYPE> x2(batch * in_dim);
        std::vector<INT_TYPE> w2(out_dim * in_dim);
        std::vector<INT_TYPE> c2(batch * out_dim);
        ios[0]->recv_data(x2.data(), x2.size() * sizeof(INT_TYPE));
        ios[0]->recv_data(w2.data(), w2.size() * sizeof(INT_TYPE));
        ios[0]->recv_data(c2.data(), c2.size() * sizeof(INT_TYPE));
        add_inplace(x2, x, PLAIN_MOD);
        add_inplace(c2, c, PLAIN_MOD);
        add_inplace(w2, w, PLAIN_MOD);
        vector<INT_TYPE> ideal = ideal_fc(x2.data(), w2.data(), PLAIN_MOD, batch, in_dim, out_dim);
        if (vector_equal(c2, ideal))
            std::cout << GREEN << "GPU-FC AB REV: PASSED" << NC << "\n";
        else
            std::cout << RED << "GPU-FC AB REV: FAILED" << NC << "\n";
    } else {
        ios[0]->send_data(x, batch * in_dim * sizeof(INT_TYPE));
        ios[0]->send_data(w, out_dim * in_dim * sizeof(INT_TYPE));
        ios[0]->send_data(c, batch * out_dim * sizeof(INT_TYPE));
        ios[0]->flush();
    }
#endif
    } // all troy objects destroyed here
    troy::MemoryPool::ReleaseUnused();
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level dispatcher
// ─────────────────────────────────────────────────────────────────────────────
void fc(IO::NetIO** ios, int party, const INT_TYPE* x, const INT_TYPE* w, INT_TYPE* c,
        size_t batch, size_t in_dim, size_t out_dim,
        int factor, bool is_ab, int device_id) {
    auto start = measure::now();

    size_t ac_batch = batch / factor;
    size_t x_slice  = ac_batch * in_dim;
    size_t w_slice  = out_dim * in_dim;
    size_t c_slice  = ac_batch * out_dim;

    for (int i = 0; i < factor; ++i) {
        const INT_TYPE* xi = x ? x + x_slice * i : nullptr;
        const INT_TYPE* wi = w ? w + w_slice * i : nullptr;
        INT_TYPE*       ci = c + c_slice * i;

#if defined(REVERSE_GPU) && REVERSE_GPU == 1
        if (is_ab)
            fc_ab_reverse(ios, party, xi, wi, ci, ac_batch, in_dim, out_dim, device_id);
        else
            fc_ab2_reverse(ios, party, xi, wi, ci, ac_batch, in_dim, out_dim, device_id);
#else
        if (is_ab)
            fc_ab(ios, party, xi, wi, ci, ac_batch, in_dim, out_dim, device_id);
        else
            fc_ab2(ios, party, xi, wi, ci, ac_batch, in_dim, out_dim, device_id);
#endif
    }

    double time = std::chrono::duration<double, std::milli>(measure::now() - start).count();
    std::cerr << "P" << party - 1 << ": FC triple time + NTT[s]: " << time / 1000.0 << "\n";
    std::cerr << "P" << party - 1
              << ": FC triple data[MiB]: " << (1.0 * ios[0]->counter) / (1 << 20) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Ideal (plaintext) FC for correctness verification
// Weight layout: w[j * in_dim + i]  (out_dim rows × in_dim cols)
// Output layout: c[b * out_dim + j]
// ─────────────────────────────────────────────────────────────────────────────
vector<INT_TYPE> ideal_fc(const INT_TYPE* x, const INT_TYPE* w, size_t t,
                           size_t batch, size_t in_dim, size_t out_dim) {
    vector<INT_TYPE> y(batch * out_dim, 0);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t j = 0; j < out_dim; ++j) {
            for (size_t i = 0; i < in_dim; ++i) {
                add_mod_inplace(y[b * out_dim + j],
                    multiply_mod(x[b * in_dim + i], w[j * in_dim + i], t), t);
            }
        }
    }
    return y;
}

} // namespace TROY
