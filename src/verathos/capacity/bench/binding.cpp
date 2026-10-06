#include <torch/extension.h>
#include <tuple>

double run_workspace_round(
    at::Tensor a,
    at::Tensor b,
    at::Tensor q,
    at::Tensor partial,
    at::Tensor root,
    uint64_t seed,
    int64_t pass_index,
    int round_index,
    int device_index
);

double run_workspace_pass_committed(
    at::Tensor a,
    at::Tensor b,
    at::Tensor q,
    at::Tensor partial,
    at::Tensor root,
    at::Tensor round_roots,
    uint64_t seed,
    int64_t pass_index,
    int rounds,
    int64_t block_size,
    int device_index
);

double run_workspace_pass_transition_committed(
    at::Tensor a,
    at::Tensor b,
    at::Tensor q,
    at::Tensor partial,
    at::Tensor root,
    at::Tensor state_roots,
    at::Tensor round_roots,
    uint64_t seed,
    int64_t pass_index,
    int rounds,
    int64_t block_size,
    int transition_mix_rounds,
    int transition_fanout,
    int device_index
);

double run_workspace_round_transition_committed(
    at::Tensor a,
    at::Tensor b,
    at::Tensor q,
    at::Tensor partial,
    at::Tensor root,
    at::Tensor state_roots,
    at::Tensor round_roots,
    uint64_t seed,
    int64_t pass_index,
    int round_index,
    int64_t block_size,
    int transition_mix_rounds,
    int transition_fanout,
    int device_index
);

double run_workspace_round_transition_state(
    at::Tensor a,
    at::Tensor b,
    at::Tensor q,
    uint64_t seed,
    int64_t pass_index,
    int round_index,
    int transition_mix_rounds,
    int transition_fanout,
    int state_index,
    int device_index
);

std::tuple<at::Tensor, at::Tensor>
q_merkle_commit_i32(at::Tensor q, int64_t block_size, int device_index);
std::tuple<at::Tensor, at::Tensor>
state_merkle_commit_i8(at::Tensor a, at::Tensor b, int64_t block_size, int device_index);

double run_fp64_identity_round_committed(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    at::Tensor round_roots,
    uint64_t seed,
    int64_t pass_index,
    int round_index,
    int64_t block_size,
    int device_index
);

double run_fp64_identity_chain_committed(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    at::Tensor round_roots,
    uint64_t seed,
    int64_t start_pass,
    int64_t passes,
    int rounds,
    int64_t block_size,
    int device_index
);

std::tuple<at::Tensor, at::Tensor>
fp64_c_merkle_commit_f64(at::Tensor c, int64_t block_size, int device_index);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run_workspace_round",
        &run_workspace_round,
        "Run one round of one fixed-workspace pass, leaving A/B/Q resident"
    );
    m.def(
        "run_workspace_pass_committed",
        &run_workspace_pass_committed,
        "Run one fixed-workspace pass and store verifier-openable round commitments"
    );
    m.def(
        "run_workspace_pass_transition_committed",
        &run_workspace_pass_transition_committed,
        "Run one transition-mix pass and store A/B transition roots plus Q round roots"
    );
    m.def(
        "run_workspace_round_transition_committed",
        &run_workspace_round_transition_committed,
        "Run one transition-mix round and store A/B transition roots plus Q root"
    );
    m.def(
        "run_workspace_round_transition_state",
        &run_workspace_round_transition_state,
        "Run one transition-mix round up to a selected state index, leaving A/B resident"
    );
    m.def(
        "q_merkle_commit_i32",
        &q_merkle_commit_i32,
        "Compute the Merkle root and flat tree of an int32 Q matrix"
    );
    m.def(
        "state_merkle_commit_i8",
        &state_merkle_commit_i8,
        "Compute the Merkle root and flat tree of int8 A/B state matrices"
    );
    m.def(
        "run_fp64_identity_round_committed",
        &run_fp64_identity_round_committed,
        "Run one FP64 identity round and store a verifier-openable C root"
    );
    m.def(
        "run_fp64_identity_chain_committed",
        &run_fp64_identity_chain_committed,
        "Run an FP64 identity pass chain and store verifier-openable C roots"
    );
    m.def(
        "fp64_c_merkle_commit_f64",
        &fp64_c_merkle_commit_f64,
        "Compute the Merkle root and flat tree of an FP64 C matrix"
    );
}
