#include <torch/extension.h>
#include <tuple>
#include <vector>

double run_workspace_pass(
    at::Tensor a,
    at::Tensor b,
    at::Tensor q,
    at::Tensor partial,
    at::Tensor root,
    uint64_t seed,
    int64_t pass_index,
    int rounds,
    int device_index
);

double run_workspace_chain(
    at::Tensor a,
    at::Tensor b,
    at::Tensor q,
    at::Tensor partial,
    at::Tensor root,
    at::Tensor roots,
    uint64_t seed,
    int64_t start_pass,
    int64_t passes,
    int rounds,
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

double run_workspace_pass_committed_mixed(
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
    int memory_mix_rounds,
    int device_index
);

double run_workspace_chain_committed(
    at::Tensor a,
    at::Tensor b,
    at::Tensor q,
    at::Tensor partial,
    at::Tensor root,
    at::Tensor roots,
    at::Tensor round_roots,
    uint64_t seed,
    int64_t start_pass,
    int64_t passes,
    int rounds,
    int64_t block_size,
    int device_index
);

double run_workspace_chain_committed_mixed(
    at::Tensor a,
    at::Tensor b,
    at::Tensor q,
    at::Tensor partial,
    at::Tensor root,
    at::Tensor roots,
    at::Tensor round_roots,
    uint64_t seed,
    int64_t start_pass,
    int64_t passes,
    int rounds,
    int64_t block_size,
    int memory_mix_rounds,
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

double run_workspace_round_mixed(
    at::Tensor a,
    at::Tensor b,
    at::Tensor q,
    at::Tensor partial,
    at::Tensor root,
    uint64_t seed,
    int64_t pass_index,
    int round_index,
    int memory_mix_rounds,
    int device_index
);

at::Tensor q_merkle_root_i32(at::Tensor q, int64_t block_size, int device_index);
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

double run_fp64_identity_pass_committed(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    at::Tensor round_roots,
    uint64_t seed,
    int64_t pass_index,
    int rounds,
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

at::Tensor fp64_c_merkle_root_f64(at::Tensor c, int64_t block_size, int device_index);
std::tuple<at::Tensor, at::Tensor>
fp64_c_merkle_commit_f64(at::Tensor c, int64_t block_size, int device_index);

std::vector<double> profile_workspace_pass(
    at::Tensor a,
    at::Tensor b,
    at::Tensor q,
    at::Tensor partial,
    at::Tensor root,
    uint64_t seed,
    int64_t pass_index,
    int rounds,
    int device_index
);

std::vector<double> profile_workspace_pass_transition_mix(
    at::Tensor a,
    at::Tensor b,
    at::Tensor q,
    at::Tensor partial,
    at::Tensor root,
    uint64_t seed,
    int64_t pass_index,
    int rounds,
    int device_index,
    int transition_mix_rounds,
    bool commit_transitions
);

std::vector<double> profile_workspace_pass_algo(
    at::Tensor a,
    at::Tensor b,
    at::Tensor q,
    at::Tensor partial,
    at::Tensor root,
    uint64_t seed,
    int64_t pass_index,
    int rounds,
    int device_index,
    int gemm_algo_code
);

std::vector<double> profile_workspace_pass_lt(
    at::Tensor a,
    at::Tensor b,
    at::Tensor q,
    at::Tensor partial,
    at::Tensor root,
    uint64_t seed,
    int64_t pass_index,
    int rounds,
    int device_index,
    int heuristic_index
);

std::vector<double> profile_workspace_pass_reduce_blocks(
    at::Tensor a,
    at::Tensor b,
    at::Tensor q,
    at::Tensor partial,
    at::Tensor root,
    uint64_t seed,
    int64_t pass_index,
    int rounds,
    int device_index,
    int requested_reduce_blocks
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run_workspace_pass",
        &run_workspace_pass,
        "Run one fixed-workspace hot-capacity pass"
    );
    m.def(
        "run_workspace_chain",
        &run_workspace_chain,
        "Run a fixed-workspace hot-capacity pass chain"
    );
    m.def(
        "run_workspace_pass_committed",
        &run_workspace_pass_committed,
        "Run one fixed-workspace pass and store verifier-openable round commitments"
    );
    m.def(
        "run_workspace_pass_committed_mixed",
        &run_workspace_pass_committed_mixed,
        "Run one fixed-workspace v2 mixed pass and store verifier-openable round commitments"
    );
    m.def(
        "run_workspace_chain_committed",
        &run_workspace_chain_committed,
        "Run a fixed-workspace pass chain and store verifier-openable round commitments"
    );
    m.def(
        "run_workspace_chain_committed_mixed",
        &run_workspace_chain_committed_mixed,
        "Run a fixed-workspace v2 mixed pass chain and store verifier-openable round commitments"
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
        "run_workspace_round",
        &run_workspace_round,
        "Run one round of one fixed-workspace pass, leaving A/B/Q resident"
    );
    m.def(
        "run_workspace_round_mixed",
        &run_workspace_round_mixed,
        "Run one v2 mixed round of one fixed-workspace pass, leaving A/B/Q resident"
    );
    m.def(
        "q_merkle_root_i32",
        &q_merkle_root_i32,
        "Compute the verifier-openable Merkle root of an int32 Q matrix"
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
        "run_fp64_identity_pass_committed",
        &run_fp64_identity_pass_committed,
        "Run one FP64 identity pass and store verifier-openable C roots"
    );
    m.def(
        "run_fp64_identity_chain_committed",
        &run_fp64_identity_chain_committed,
        "Run an FP64 identity pass chain and store verifier-openable C roots"
    );
    m.def(
        "fp64_c_merkle_root_f64",
        &fp64_c_merkle_root_f64,
        "Compute the verifier-openable Merkle root of an FP64 C matrix"
    );
    m.def(
        "fp64_c_merkle_commit_f64",
        &fp64_c_merkle_commit_f64,
        "Compute the Merkle root and flat tree of an FP64 C matrix"
    );
    m.def(
        "profile_workspace_pass",
        &profile_workspace_pass,
        "Profile one fixed-workspace hot-capacity pass; returns [fill_ms, gemm_ms, reduce_ms, total_ms]"
    );
    m.def(
        "profile_workspace_pass_transition_mix",
        &profile_workspace_pass_transition_mix,
        "Profile a transition-dependent memory mix pass; returns [fill_ms, mix_ms, state_reduce_ms, gemm_ms, q_reduce_ms, total_ms]"
    );
    m.def(
        "profile_workspace_pass_algo",
        &profile_workspace_pass_algo,
        "Profile one fixed-workspace hot-capacity pass with a cuBLAS GEMM algorithm code"
    );
    m.def(
        "profile_workspace_pass_lt",
        &profile_workspace_pass_lt,
        "Profile one fixed-workspace hot-capacity pass with a zero-workspace cuBLASLt heuristic index"
    );
    m.def(
        "profile_workspace_pass_reduce_blocks",
        &profile_workspace_pass_reduce_blocks,
        "Profile one fixed-workspace hot-capacity pass with a requested reduction block count"
    );
}
