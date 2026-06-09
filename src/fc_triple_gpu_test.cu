// fc_triple_gpu_test.cu
//
// Standalone multi-process GPU FC triple test.
// Forks num_processes children; each calls TROY::fc directly for a
// representative ResNet50 FC layer, bypassing all SEAL/OT/mux setup.
// Use this to verify that each process lands on the correct GPU and to
// benchmark FC triple generation with vs without GPU.
//
// Usage:
//   P0 / H100 (model owner): ./fc_triple_gpu_test 0 <P1_IP> <base_port> <num_processes> <num_gpus>
//   P1 / A100 (data  owner): ./fc_triple_gpu_test 1 <P0_IP> <base_port> <num_processes> <num_gpus>
//
// PID 0 = H100 = BOB  (holds weights, internal party 2)
// PID 1 = A100 = ALICE (encrypts input, internal party 1)
// Example with 24 processes, H100 on $IPA, A100 on $IPB:
//   H100:  ./fc_triple_gpu_test 0 $IPB 7000 24 8
//   A100:  ./fc_triple_gpu_test 1 $IPA 7000 24 4

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include <cuda_runtime.h>

#include "constants.hpp"
#include "io/net_io_channel.hpp"
#include "troy/fc_gpu.cuh"

// ── Test FC layer: ResNet50 first FC layer (25088 → 4096) ────────────────────
// Large enough to stress the GPU pipeline and show a meaningful speedup.
static constexpr size_t TEST_BATCH   = 1;
static constexpr size_t TEST_IN_DIM  = 25088;
static constexpr size_t TEST_OUT_DIM = 4096;

// ── port spacing: 2 sockets per 2PC process ──────────────────────────────────
static constexpr int SOCKETS_PER_PROC = 2;

static void run_child(int party, int party_id, const char* peer_ip, int base_port,
                      int process_id, int num_gpus) {
    int gpu_id = process_id % num_gpus;
    int port   = base_port + SOCKETS_PER_PROC * process_id;

    // First CUDA call in this child — initialises the runtime cleanly.
    // Never call any CUDA API in the parent before fork().
    cudaSetDevice(gpu_id);
    int actual_gpu = -1;
    cudaGetDevice(&actual_gpu);
    int num_cuda_devs = 0;
    cudaGetDeviceCount(&num_cuda_devs);

    // ALICE (party 1) listens; BOB (party 2) connects.
    const char* addr = (party == 1 /* ALICE */) ? nullptr : peer_ip;
    IO::NetIO*  io   = new IO::NetIO(addr, port);
    IO::NetIO*  ios[1] = {io};

    std::vector<TROY::INT_TYPE> x(TEST_BATCH * TEST_IN_DIM,  1);
    std::vector<TROY::INT_TYPE> w(TEST_OUT_DIM * TEST_IN_DIM, 1);
    std::vector<TROY::INT_TYPE> c(TEST_BATCH * TEST_OUT_DIM,  0);

    auto t0 = std::chrono::high_resolution_clock::now();

    TROY::fc(ios, party, x.data(), w.data(), c.data(),
             TEST_BATCH, TEST_IN_DIM, TEST_OUT_DIM,
             /*factor=*/1, /*is_ab=*/false, gpu_id);

    double elapsed = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();

    double mb = static_cast<double>(io->counter) / (1 << 20);

    char buf[256];
    snprintf(buf, sizeof(buf),
             "[P%d pid=%d proc=%02d] gpu_requested=%d gpu_actual=%d/%d  time=%.2fs  sent=%.1fMiB\n",
             party_id, getpid(), process_id, gpu_id, actual_gpu, num_cuda_devs, elapsed, mb);
    fputs(buf, stderr);

    delete io;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        fprintf(stderr,
            "Usage: %s <pid 0|1> <peer_ip> <base_port> <num_processes> <num_gpus>\n"
            "  pid 0 = H100 / BOB  (holds weights)\n"
            "  pid 1 = A100 / ALICE (encrypts input)\n"
            "  Run on both machines with the same base_port.\n", argv[0]);
        return 1;
    }

    int         party_id      = atoi(argv[1]);   // 0 = H100/BOB, 1 = A100/ALICE
    const char* peer_ip       = argv[2];
    int         base_port     = atoi(argv[3]);
    int         num_processes = atoi(argv[4]);
    int         num_gpus      = atoi(argv[5]);

    if (party_id != 0 && party_id != 1) {
        fprintf(stderr, "PID must be 0 (H100/BOB) or 1 (A100/ALICE)\n");
        return 1;
    }

    // PID 0 (H100) → internal party 2 (BOB,  holds weights)
    // PID 1 (A100) → internal party 1 (ALICE, encrypts input)
    int party = (party_id == 0) ? 2 : 1;

    // NOTE: do NOT call any CUDA API here — initialising the runtime in the
    // parent process breaks all forked children with "initialization error".
    fprintf(stderr, "P%d (%s): using %d GPU(s) across %d processes\n",
            party_id, (party == 1 ? "ALICE/A100" : "BOB/H100"), num_gpus, num_processes);
    fprintf(stderr, "P%d: FC layer: batch=%zu in=%zu out=%zu\n",
            party_id, TEST_BATCH, TEST_IN_DIM, TEST_OUT_DIM);

    auto wall_start = std::chrono::high_resolution_clock::now();

    std::vector<pid_t> children(num_processes);
    for (int id = 0; id < num_processes; ++id) {
        pid_t cpid = fork();
        if (cpid == 0) {
            run_child(party, party_id, peer_ip, base_port, id, num_gpus);
            exit(0);
        }
        children[id] = cpid;
    }

    int failed = 0;
    for (int id = 0; id < num_processes; ++id) {
        int status = 0;
        waitpid(children[id], &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
            ++failed;
    }

    double wall = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - wall_start).count();

    fprintf(stderr, "\nP%d: all %d processes done in %.2fs (%d failed)\n",
            party_id, num_processes, wall, failed);
    return failed ? 1 : 0;
}
