//! Verathos proof-protocol-v2 multilinear polynomial commitments.
//!
//! This crate intentionally exposes a small byte-oriented C ABI.  It uses
//! Nova's Pallas/Pedersen IPA evaluation engine with deterministic generators
//! and zero blinding. Signed i8 commitment-only calls use Nova's exact
//! small-integer MSM path. IPA openings retain the regular scalar path.

#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(not(target_endian = "little"))]
compile_error!("verathos-pcs-v2 currently requires a little-endian target");

use bincode::config;
use ff::{Field, PrimeField};
use halo2curves::group::GroupEncoding;
use nova_snark::{
    provider::{
        ipa_pc,
        keccak::Keccak256Transcript,
        pasta::pallas,
        pedersen,
        traits::{DlogGroup, DlogGroupExt},
        PallasEngine,
    },
    spartan::polys::{eq::EqPolynomial, multilinear::MultilinearPolynomial},
    traits::{
        commitment::CommitmentEngineTrait, evaluation::EvaluationEngineTrait, Engine,
        TranscriptEngineTrait, TranscriptReprTrait,
    },
};
use num_bigint::BigUint;
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sha3::{
    digest::{ExtendableOutput, XofReader},
    Shake256,
};
use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    env, fs,
    panic::{catch_unwind, AssertUnwindSafe},
    ptr,
    sync::{Arc, Condvar, Mutex, OnceLock, RwLock, RwLockReadGuard},
};

type E = PallasEngine;
type Fr = <E as Engine>::Scalar;
type CE = pedersen::CommitmentEngine<E>;
type EE = ipa_pc::EvaluationEngine<E>;
type Commitment = <CE as CommitmentEngineTrait<E>>::Commitment;
type CommitmentKey = <CE as CommitmentEngineTrait<E>>::CommitmentKey;
type ProverKey = <EE as EvaluationEngineTrait<E>>::ProverKey;
type VerifierKey = <EE as EvaluationEngineTrait<E>>::VerifierKey;
type Proof = <EE as EvaluationEngineTrait<E>>::EvaluationArgument;

/// Stable ABI version for the ctypes wrapper.
pub const ABI_VERSION: u32 = 10;
/// Protocol version absorbed into every opening transcript.
pub const PROTOCOL_VERSION: u64 = 2;
/// Largest accepted unpadded vector.  This covers current transformer axes
/// while bounding setup memory and attacker-controlled verifier work.
pub const MAX_VECTOR_LEN: usize = 1 << 17;
pub const COMMITMENT_BYTES: usize = 32;
pub const SCALAR_BYTES: usize = 32;
pub const MAX_PROOF_BYTES: usize = 2048;
/// Largest accepted linear combination of commitments. This covers the
/// protocol's maximum block width while bounding verifier-controlled work.
pub const MAX_COMBINE_TERMS: usize = 512;
/// Largest accepted catalog-wide commitment fold. This separate entry point
/// is used only for authenticated static catalogs; ordinary proof-controlled
/// combinations retain the much smaller `MAX_COMBINE_TERMS` bound.
pub const MAX_CATALOG_COMBINE_TERMS: usize = 1 << 20;
/// Maximum number of decompressed catalog points retained across models.
/// At 64 bytes per affine point this is approximately 256 MiB.
pub const MAX_CACHED_CATALOG_POINTS: usize = 1 << 22;
/// Maximum independent catalog folds in one transcript-bound batch.
pub const MAX_CATALOG_FOLDS: usize = 32;
/// Largest accepted direct batch of compressed commitment encodings.
pub const MAX_VALIDATE_COMMITMENTS: usize = 1 << 20;
/// Largest accepted commitment-only signed-i8 batch.
pub const MAX_BATCH_COMMITMENTS: usize = 4096;
/// Largest accepted packed signed-i8 batch input.
pub const MAX_BATCH_I8_BYTES: usize = 16 << 20;
/// Largest accepted packed signed-i64 aggregate-opening input.
pub const MAX_BATCH_I64_BYTES: usize = 64 << 20;
pub const MAX_GEMM_BATCH_BLOCKS: usize = 64;
pub const MAX_GEMM_BATCH_BYTES: usize = 16 << 20;
pub const MAX_LEAN_PROJECTION_VECTORS: usize = 2048;
pub const MAX_LEAN_PROJECTION_BYTES: usize = 512 << 20;
pub const MAX_MERKLE_BUILD_LEAVES: usize = 1 << 20;
const MAX_CONCURRENT_OPERATIONS: usize = 1;
const DEFAULT_WORKER_THREADS: usize = 8;
const MAX_WORKER_THREADS: usize = 64;

const GENERATOR_DOMAIN: &[u8] = b"VERATHOS_PCS_V2_PALLAS_PEDERSEN_GENS_V1";
const TRANSCRIPT_DOMAIN: &[u8] = b"VERATHOS_PCS_V2_PALLAS_IPA_TRANSCRIPT_V1";
const LINEAR_TRANSCRIPT_DOMAIN: &[u8] =
    b"VERATHOS_PCS_V3_PALLAS_IPA_LINEAR_TRANSCRIPT_V1";
const IPA_EVALUATION_GENERATOR_DOMAIN: &[u8] = b"ipa";
const GEMM_BATCH_TRANSCRIPT_DOMAIN: &[u8] = b"VERATHOS/GEMM_V2/BATCH_SUMCHECK/PALLAS_SCALAR/SHA256";
const LEAN_PROJECTION_BATCH_TRANSCRIPT_DOMAIN: &[u8] =
    b"VERATHOS/PROOF_V3/LEAN_PROJECTION_BATCH/U31X4/AGGREGATE_CATALOG/V2/SHA256";
const LEAN_PROJECTION_FOLD_TRANSCRIPT_DOMAIN: &[u8] =
    b"VERATHOS/PROOF_V3/LEAN_PROJECTION_FOLD/U31X4/V1/SHAKE256";
const LEAN_PROJECTION_NATIVE_MAGIC: &[u8; 4] = b"VLPN";
const LEAN_PROJECTION_NATIVE_VERSION: u16 = 1;
const LEAN_PROJECTION_FOLD_COUNT: usize = 4;
const MAX_LEAN_PROJECTION_MATERIAL_CLAIMS: usize = 512;
const GEMM_BATCH_PROOF_MAGIC: &[u8; 4] = b"VGB2";
const PCS_MERKLE_LEAF_DOMAIN: &[u8] = b"VERATHOS/PCS_V2/BLOCK_MERKLE/LEAF/SHA256";
const PCS_MERKLE_PADDING_DOMAIN: &[u8] = b"VERATHOS/PCS_V2/BLOCK_MERKLE/PADDING/SHA256";
const PCS_MERKLE_NODE_DOMAIN: &[u8] = b"VERATHOS/PCS_V2/BLOCK_MERKLE/NODE/SHA256";
const PCS_FP16_BLOCK_DOMAIN: &[u8] = b"VERATHOS/PCS_V2/FP16_BLOCK/SHA256";

const ERR_NULL: i32 = -1;
const ERR_LENGTH: i32 = -2;
const ERR_ENCODING: i32 = -3;
const ERR_FORMAT: i32 = -4;
const ERR_CRYPTO: i32 = -5;
const ERR_BUFFER: i32 = -6;
const ERR_INTERNAL: i32 = -99;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
enum Encoding {
    SignedI8 = 1,
    SignedI64 = 2,
    PallasScalar = 3,
}

impl Encoding {
    fn parse(raw: u32) -> Result<Self, PcsError> {
        match raw {
            1 => Ok(Self::SignedI8),
            2 => Ok(Self::SignedI64),
            3 => Ok(Self::PallasScalar),
            _ => Err(PcsError::new(ERR_ENCODING, "unsupported vector encoding")),
        }
    }
}

#[derive(Debug)]
struct PcsError {
    code: i32,
    message: String,
}

impl PcsError {
    fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

struct Setup {
    ck: CommitmentKey,
    // Retained only for the regression test that proves the optimized prover
    // is wire-identical to Nova's reference implementation.
    #[cfg_attr(not(test), allow(dead_code))]
    pk: ProverKey,
    vk: VerifierKey,
    ipa_evaluation_base: pallas::Affine,
    // For signed i8 commitments, s = (s + 128) - 128.  The second term only
    // depends on the key and the unpadded vector length, so cache it once per
    // process and reuse it for every static weight column of that length.
    i8_offset_adjustments: Mutex<HashMap<usize, Commitment>>,
}

static SETUP_CACHE: OnceLock<Mutex<HashMap<usize, Arc<Setup>>>> = OnceLock::new();
static CATALOG_POINT_CACHE: OnceLock<Mutex<CatalogPointCache>> = OnceLock::new();
// Nova generator derivation and curve operations are individually thread-safe,
// but generator derivation must not overlap an operation using a key for a
// different size. Cached operations take a shared guard and remain concurrent.
static OPERATION_GATE: OnceLock<RwLock<()>> = OnceLock::new();
// Nova's underlying curve backend uses process-global internals and can fault
// under simultaneous native calls. Serialize curve operations while retaining
// the read/write gate that excludes generator setup.
static OPERATION_LIMIT: OnceLock<(Mutex<usize>, Condvar)> = OnceLock::new();
static WORKER_POOL: OnceLock<Result<ThreadPool, String>> = OnceLock::new();

#[derive(Default)]
struct CatalogPointCache {
    entries: HashMap<[u8; 32], Arc<Vec<pallas::Affine>>>,
    least_recently_used: VecDeque<[u8; 32]>,
    total_points: usize,
}

impl CatalogPointCache {
    fn touch(&mut self, catalog_id: [u8; 32]) {
        if let Some(position) = self
            .least_recently_used
            .iter()
            .position(|candidate| *candidate == catalog_id)
        {
            self.least_recently_used.remove(position);
        }
        self.least_recently_used.push_back(catalog_id);
    }

    fn insert(
        &mut self,
        catalog_id: [u8; 32],
        bases: Arc<Vec<pallas::Affine>>,
    ) -> Result<(), PcsError> {
        let point_count = bases.len();
        if point_count > MAX_CACHED_CATALOG_POINTS {
            return Err(PcsError::new(
                ERR_LENGTH,
                "catalog is larger than the native point-cache bound",
            ));
        }
        if self.entries.contains_key(&catalog_id) {
            self.touch(catalog_id);
            return Ok(());
        }
        while self.total_points.saturating_add(point_count) > MAX_CACHED_CATALOG_POINTS {
            let oldest = self.least_recently_used.pop_front().ok_or_else(|| {
                PcsError::new(ERR_INTERNAL, "catalog point-cache accounting is corrupt")
            })?;
            if let Some(removed) = self.entries.remove(&oldest) {
                self.total_points = self.total_points.saturating_sub(removed.len());
            }
        }
        self.total_points = self
            .total_points
            .checked_add(point_count)
            .ok_or_else(|| PcsError::new(ERR_INTERNAL, "catalog point-cache size overflow"))?;
        self.entries.insert(catalog_id, bases);
        self.touch(catalog_id);
        Ok(())
    }

    fn get(&mut self, catalog_id: [u8; 32]) -> Option<Arc<Vec<pallas::Affine>>> {
        let bases = self.entries.get(&catalog_id).cloned()?;
        self.touch(catalog_id);
        Some(bases)
    }
}

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_last_error(message: impl Into<String>) {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = message.into());
}

fn clear_last_error() {
    LAST_ERROR.with(|slot| slot.borrow_mut().clear());
}

fn configured_worker_threads() -> usize {
    env::var("VERATHOS_PCS_V2_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| (1..=MAX_WORKER_THREADS).contains(value))
        .unwrap_or_else(|| {
            let available = std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1);
            linux_cgroup_cpu_capacity_ceil()
                .unwrap_or(available)
                .min(DEFAULT_WORKER_THREADS)
        })
}

fn parse_cpu_list_count(value: &str) -> Option<usize> {
    let mut count = 0usize;
    for part in value.trim().split(',') {
        if part.is_empty() {
            continue;
        }
        let mut bounds = part.split('-');
        let start = bounds.next()?.parse::<usize>().ok()?;
        let end = bounds
            .next()
            .map(str::parse::<usize>)
            .transpose()
            .ok()?
            .unwrap_or(start);
        if bounds.next().is_some() || end < start {
            return None;
        }
        count = count.checked_add(end - start + 1)?;
    }
    (count > 0).then_some(count)
}

#[cfg(target_os = "linux")]
fn linux_cgroup_cpu_capacity_ceil() -> Option<usize> {
    let cpu_max = fs::read_to_string("/sys/fs/cgroup/cpu.max").ok()?;
    let mut fields = cpu_max.split_whitespace();
    let quota = fields.next()?;
    let period = fields.next()?.parse::<usize>().ok()?;
    if quota == "max" || period == 0 || fields.next().is_some() {
        return None;
    }
    let quota = quota.parse::<usize>().ok()?;
    let mut capacity = quota.checked_add(period - 1)? / period;
    if let Ok(cpuset) = fs::read_to_string("/sys/fs/cgroup/cpuset.cpus.effective") {
        if let Some(count) = parse_cpu_list_count(&cpuset) {
            capacity = capacity.min(count);
        }
    }
    (capacity > 0).then_some(capacity)
}

#[cfg(not(target_os = "linux"))]
fn linux_cgroup_cpu_capacity_ceil() -> Option<usize> {
    None
}

fn worker_pool() -> Result<&'static ThreadPool, PcsError> {
    WORKER_POOL
        .get_or_init(|| {
            ThreadPoolBuilder::new()
                .num_threads(configured_worker_threads())
                .thread_name(|index| format!("verathos-pcs-v2-{index}"))
                .build()
                .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(|message| {
            PcsError::new(
                ERR_INTERNAL,
                format!("PCS worker pool initialization failed: {message}"),
            )
        })
}

fn in_worker_pool<F, T>(operation: F) -> Result<T, PcsError>
where
    F: FnOnce() -> T + Send,
    T: Send,
{
    Ok(worker_pool()?.install(operation))
}

fn shape(vector_len: usize) -> Result<(usize, usize), PcsError> {
    if vector_len == 0 || vector_len > MAX_VECTOR_LEN {
        return Err(PcsError::new(
            ERR_LENGTH,
            format!("vector length must be in 1..={MAX_VECTOR_LEN}"),
        ));
    }
    let padded_len = vector_len
        .checked_next_power_of_two()
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "vector length overflow"))?;
    let log_n = padded_len.trailing_zeros() as usize;
    Ok((padded_len, log_n))
}

fn proof_wire_len(log_n: usize) -> Result<usize, PcsError> {
    if log_n > 17 {
        return Err(PcsError::new(
            ERR_LENGTH,
            "unsupported padded vector length",
        ));
    }
    // Nova 0.73 + bincode 2.0.1 standard encoding: one-byte Vec lengths,
    // log_n compressed Pallas points in each of L/R, then one scalar.
    Ok(2 + 64 * log_n + SCALAR_BYTES)
}

struct OperationPermit;

impl Drop for OperationPermit {
    fn drop(&mut self) {
        let (counter, available) = OPERATION_LIMIT.get_or_init(|| (Mutex::new(0), Condvar::new()));
        if let Ok(mut active) = counter.lock() {
            *active = active.saturating_sub(1);
            available.notify_one();
        }
    }
}

struct OperationGuard {
    _setup_gate: RwLockReadGuard<'static, ()>,
    _permit: OperationPermit,
}

fn operation_permit() -> Result<OperationPermit, PcsError> {
    let (counter, available) = OPERATION_LIMIT.get_or_init(|| (Mutex::new(0), Condvar::new()));
    let mut active = counter
        .lock()
        .map_err(|_| PcsError::new(ERR_INTERNAL, "PCS operation limit is poisoned"))?;
    while *active >= MAX_CONCURRENT_OPERATIONS {
        active = available
            .wait(active)
            .map_err(|_| PcsError::new(ERR_INTERNAL, "PCS operation limit is poisoned"))?;
    }
    *active += 1;
    Ok(OperationPermit)
}

fn operation_guard() -> Result<OperationGuard, PcsError> {
    let permit = operation_permit()?;
    let setup_gate = OPERATION_GATE
        .get_or_init(|| RwLock::new(()))
        .read()
        .map_err(|_| PcsError::new(ERR_INTERNAL, "PCS operation gate is poisoned"))?;
    Ok(OperationGuard {
        _setup_gate: setup_gate,
        _permit: permit,
    })
}

fn setup_for(padded_len: usize) -> Result<Arc<Setup>, PcsError> {
    let cache = SETUP_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .map_err(|_| PcsError::new(ERR_INTERNAL, "PCS setup cache is poisoned"))?;
    if let Some(existing) = guard.get(&padded_len) {
        return Ok(Arc::clone(existing));
    }

    // Setup is serialized by the cache lock. The exclusive gate also prevents
    // generator derivation from overlapping operations on another cached key.
    let _exclusive = OPERATION_GATE
        .get_or_init(|| RwLock::new(()))
        .write()
        .map_err(|_| PcsError::new(ERR_INTERNAL, "PCS operation gate is poisoned"))?;
    let ck = in_worker_pool(|| CE::setup(GENERATOR_DOMAIN, padded_len))?
        .map_err(|_| PcsError::new(ERR_CRYPTO, "PCS generator setup failed"))?;
    let (pk, vk) = in_worker_pool(|| EE::setup(&ck))?
        .map_err(|_| PcsError::new(ERR_CRYPTO, "PCS evaluation setup failed"))?;
    // Nova derives this one-element key internally in `EE::setup` but keeps
    // it private. Re-deriving the deterministic key lets the optimized prover
    // reproduce Nova's exact IPA transcript and wire proof.
    let ipa_evaluation_key = in_worker_pool(|| CE::setup(IPA_EVALUATION_GENERATOR_DOMAIN, 1))?
        .map_err(|_| PcsError::new(ERR_CRYPTO, "IPA evaluation generator setup failed"))?;
    let ipa_evaluation_base = *ipa_evaluation_key
        .generators()
        .first()
        .ok_or_else(|| PcsError::new(ERR_INTERNAL, "IPA evaluation key is empty"))?;
    let created = Arc::new(Setup {
        ck,
        pk,
        vk,
        ipa_evaluation_base,
        i8_offset_adjustments: Mutex::new(HashMap::new()),
    });
    guard.insert(padded_len, Arc::clone(&created));
    Ok(created)
}

fn signed_i64_to_scalar(value: i64) -> Fr {
    if value >= 0 {
        Fr::from(value as u64)
    } else {
        -Fr::from(value.unsigned_abs())
    }
}

fn poly_from_i8(values: &[i8]) -> Result<(Vec<Fr>, usize), PcsError> {
    let (padded_len, _) = shape(values.len())?;
    let mut poly = Vec::with_capacity(padded_len);
    poly.extend(
        values
            .iter()
            .map(|value| signed_i64_to_scalar(*value as i64)),
    );
    poly.resize(padded_len, Fr::ZERO);
    Ok((poly, padded_len))
}

fn poly_from_i64(values: &[i64]) -> Result<(Vec<Fr>, usize), PcsError> {
    let (padded_len, _) = shape(values.len())?;
    let mut poly = Vec::with_capacity(padded_len);
    poly.extend(values.iter().map(|value| signed_i64_to_scalar(*value)));
    poly.resize(padded_len, Fr::ZERO);
    Ok((poly, padded_len))
}

fn poly_from_scalar_bytes(values: &[u8], vector_len: usize) -> Result<(Vec<Fr>, usize), PcsError> {
    let (padded_len, _) = shape(vector_len)?;
    let expected_len = vector_len
        .checked_mul(SCALAR_BYTES)
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "field-vector byte length overflow"))?;
    if values.len() != expected_len {
        return Err(PcsError::new(
            ERR_LENGTH,
            "field vector has the wrong byte length",
        ));
    }
    let mut poly = values
        .chunks_exact(SCALAR_BYTES)
        .map(decode_scalar)
        .collect::<Result<Vec<_>, _>>()?;
    poly.resize(padded_len, Fr::ZERO);
    Ok((poly, padded_len))
}

fn decode_pallas_affine(bytes: &[u8]) -> Option<pallas::Affine> {
    if bytes.len() != COMMITMENT_BYTES {
        return None;
    }
    let mut repr = <pallas::Affine as GroupEncoding>::Repr::default();
    repr.as_mut().copy_from_slice(bytes);
    let point = Option::<pallas::Affine>::from(pallas::Affine::from_bytes(&repr))?;
    (point.to_bytes().as_ref() == bytes).then_some(point)
}

fn decode_pallas_point(bytes: &[u8]) -> Option<pallas::Point> {
    decode_pallas_affine(bytes).map(pallas::Point::from)
}

fn canonical_point(bytes: &[u8]) -> bool {
    decode_pallas_point(bytes).is_some()
}

fn decode_scalar(bytes: &[u8]) -> Result<Fr, PcsError> {
    if bytes.len() != SCALAR_BYTES {
        return Err(PcsError::new(ERR_FORMAT, "scalar must be exactly 32 bytes"));
    }
    let mut repr = <Fr as PrimeField>::Repr::default();
    repr.as_mut().copy_from_slice(bytes);
    let Some(scalar) = Option::<Fr>::from(Fr::from_repr(repr)) else {
        return Err(PcsError::new(ERR_FORMAT, "non-canonical Pallas scalar"));
    };
    if scalar.to_repr().as_ref() != bytes {
        return Err(PcsError::new(ERR_FORMAT, "non-canonical Pallas scalar"));
    }
    Ok(scalar)
}

fn encode_scalar(scalar: &Fr) -> [u8; SCALAR_BYTES] {
    let repr = scalar.to_repr();
    let mut out = [0u8; SCALAR_BYTES];
    out.copy_from_slice(repr.as_ref());
    out
}

fn decode_point(point_bytes: &[u8], expected_log_n: usize) -> Result<Vec<Fr>, PcsError> {
    let expected = expected_log_n
        .checked_mul(SCALAR_BYTES)
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "point length overflow"))?;
    if point_bytes.len() != expected {
        return Err(PcsError::new(
            ERR_LENGTH,
            format!("MLE point must contain exactly {expected_log_n} scalars"),
        ));
    }
    point_bytes
        .chunks_exact(SCALAR_BYTES)
        .map(decode_scalar)
        .collect()
}

fn encode_commitment(commitment: &Commitment) -> Result<[u8; COMMITMENT_BYTES], PcsError> {
    let bytes = bincode::serde::encode_to_vec(commitment, config::standard())
        .map_err(|_| PcsError::new(ERR_INTERNAL, "commitment encoding failed"))?;
    if bytes.len() != COMMITMENT_BYTES || !canonical_point(&bytes) {
        return Err(PcsError::new(
            ERR_INTERNAL,
            "unexpected Nova commitment encoding",
        ));
    }
    let mut out = [0u8; COMMITMENT_BYTES];
    out.copy_from_slice(&bytes);
    Ok(out)
}

fn decode_commitment(bytes: &[u8]) -> Result<Commitment, PcsError> {
    if !canonical_point(bytes) {
        return Err(PcsError::new(ERR_FORMAT, "non-canonical Pallas commitment"));
    }
    let (commitment, consumed) = bincode::serde::decode_from_slice::<Commitment, _>(
        bytes,
        config::standard().with_limit::<64>(),
    )
    .map_err(|_| PcsError::new(ERR_FORMAT, "invalid Pallas commitment"))?;
    if consumed != bytes.len() || encode_commitment(&commitment)?.as_slice() != bytes {
        return Err(PcsError::new(ERR_FORMAT, "non-canonical Pallas commitment"));
    }
    Ok(commitment)
}

fn combine_commitments_bounded(
    commitment_bytes: &[u8],
    coefficient_bytes: &[u8],
    term_count: usize,
    maximum_terms: usize,
) -> Result<[u8; COMMITMENT_BYTES], PcsError> {
    if term_count == 0 || term_count > maximum_terms {
        return Err(PcsError::new(
            ERR_LENGTH,
            format!("commitment term count must be in 1..={maximum_terms}"),
        ));
    }
    let commitment_len = term_count
        .checked_mul(COMMITMENT_BYTES)
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "commitment input length overflow"))?;
    let coefficient_len = term_count
        .checked_mul(SCALAR_BYTES)
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "coefficient input length overflow"))?;
    if commitment_bytes.len() != commitment_len || coefficient_bytes.len() != coefficient_len {
        return Err(PcsError::new(
            ERR_LENGTH,
            "commitment and coefficient inputs have the wrong byte length",
        ));
    }

    // Curve operations share the read gate with commits, proofs, and
    // verification so generator setup cannot overlap them.
    let _operation = operation_guard()?;
    let mut bases = Vec::with_capacity(term_count);
    let mut coefficients = Vec::with_capacity(term_count);
    for (commitment_raw, coefficient_raw) in commitment_bytes
        .chunks_exact(COMMITMENT_BYTES)
        .zip(coefficient_bytes.chunks_exact(SCALAR_BYTES))
    {
        let point = decode_pallas_affine(commitment_raw)
            .ok_or_else(|| PcsError::new(ERR_FORMAT, "non-canonical Pallas commitment"))?;
        bases.push(point);
        coefficients.push(decode_scalar(coefficient_raw)?);
    }
    let combined = in_worker_pool(|| {
        <pallas::Point as DlogGroupExt>::vartime_multiscalar_mul(&coefficients, &bases)
    })?;
    let encoded = combined.to_bytes();
    let mut output = [0u8; COMMITMENT_BYTES];
    output.copy_from_slice(encoded.as_ref());
    Ok(output)
}

fn catalog_commitment_id(commitment_bytes: &[u8], term_count: usize) -> Result<[u8; 32], PcsError> {
    let term_count = u64::try_from(term_count)
        .map_err(|_| PcsError::new(ERR_LENGTH, "catalog term count exceeds u64"))?;
    let mut hasher = Sha256::new();
    hasher.update(b"VERATHOS/PCS_V2/CATALOG_POINT_CACHE/V1");
    hasher.update(term_count.to_le_bytes());
    hasher.update(commitment_bytes);
    let digest = hasher.finalize();
    let mut catalog_id = [0u8; 32];
    catalog_id.copy_from_slice(&digest);
    Ok(catalog_id)
}

fn register_catalog_commitments(
    commitment_bytes: &[u8],
    term_count: usize,
) -> Result<[u8; 32], PcsError> {
    if term_count == 0 || term_count > MAX_CATALOG_COMBINE_TERMS {
        return Err(PcsError::new(
            ERR_LENGTH,
            format!("catalog commitment term count must be in 1..={MAX_CATALOG_COMBINE_TERMS}"),
        ));
    }
    let commitment_len = term_count
        .checked_mul(COMMITMENT_BYTES)
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "commitment input length overflow"))?;
    if commitment_bytes.len() != commitment_len {
        return Err(PcsError::new(
            ERR_LENGTH,
            "catalog commitment input has the wrong byte length",
        ));
    }
    let catalog_id = catalog_commitment_id(commitment_bytes, term_count)?;
    {
        let mut cache = CATALOG_POINT_CACHE
            .get_or_init(|| Mutex::new(CatalogPointCache::default()))
            .lock()
            .map_err(|_| PcsError::new(ERR_INTERNAL, "catalog point cache is poisoned"))?;
        if cache.get(catalog_id).is_some() {
            return Ok(catalog_id);
        }
    }

    let _operation = operation_guard()?;
    let bases = Arc::new(in_worker_pool(|| {
        commitment_bytes
            .par_chunks_exact(COMMITMENT_BYTES)
            .map(|commitment_raw| {
                decode_pallas_affine(commitment_raw)
                    .ok_or_else(|| PcsError::new(ERR_FORMAT, "non-canonical Pallas commitment"))
            })
            .collect::<Result<Vec<_>, PcsError>>()
    })??);
    let mut cache = CATALOG_POINT_CACHE
        .get_or_init(|| Mutex::new(CatalogPointCache::default()))
        .lock()
        .map_err(|_| PcsError::new(ERR_INTERNAL, "catalog point cache is poisoned"))?;
    cache.insert(catalog_id, bases)?;
    Ok(catalog_id)
}

fn combine_registered_catalog_commitments_u63(
    catalog_id: [u8; 32],
    coefficient_bytes: &[u8],
    term_count: usize,
) -> Result<[u8; COMMITMENT_BYTES], PcsError> {
    if term_count == 0 || term_count > MAX_CATALOG_COMBINE_TERMS {
        return Err(PcsError::new(
            ERR_LENGTH,
            format!("catalog commitment term count must be in 1..={MAX_CATALOG_COMBINE_TERMS}"),
        ));
    }
    let coefficient_len = term_count
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "coefficient input length overflow"))?;
    if coefficient_bytes.len() != coefficient_len {
        return Err(PcsError::new(
            ERR_LENGTH,
            "catalog coefficient input has the wrong byte length",
        ));
    }
    let bases = CATALOG_POINT_CACHE
        .get_or_init(|| Mutex::new(CatalogPointCache::default()))
        .lock()
        .map_err(|_| PcsError::new(ERR_INTERNAL, "catalog point cache is poisoned"))?
        .get(catalog_id)
        .ok_or_else(|| PcsError::new(ERR_FORMAT, "catalog point cache entry is unavailable"))?;
    if bases.len() != term_count {
        return Err(PcsError::new(
            ERR_LENGTH,
            "registered catalog term count does not match",
        ));
    }
    let coefficients = coefficient_bytes
        .chunks_exact(std::mem::size_of::<u64>())
        .map(|coefficient_raw| {
            let coefficient = u64::from_le_bytes(
                coefficient_raw
                    .try_into()
                    .map_err(|_| PcsError::new(ERR_FORMAT, "malformed u63 coefficient"))?,
            );
            if coefficient >= 1u64 << 63 {
                return Err(PcsError::new(
                    ERR_FORMAT,
                    "catalog coefficient is not an unsigned 63-bit integer",
                ));
            }
            Ok(coefficient)
        })
        .collect::<Result<Vec<_>, PcsError>>()?;
    let _operation = operation_guard()?;
    let combined = in_worker_pool(|| {
        <pallas::Point as DlogGroupExt>::vartime_multiscalar_mul_small_with_max_num_bits(
            &coefficients,
            bases.as_slice(),
            63,
        )
    })?;
    let encoded = combined.to_bytes();
    let mut output = [0u8; COMMITMENT_BYTES];
    output.copy_from_slice(encoded.as_ref());
    Ok(output)
}

fn combine_registered_catalog_commitments_u7_batch(
    catalog_id: [u8; 32],
    coefficient_bytes: &[u8],
    term_count: usize,
    fold_count: usize,
) -> Result<Vec<u8>, PcsError> {
    if term_count == 0 || term_count > MAX_CATALOG_COMBINE_TERMS {
        return Err(PcsError::new(
            ERR_LENGTH,
            format!("catalog commitment term count must be in 1..={MAX_CATALOG_COMBINE_TERMS}"),
        ));
    }
    if fold_count == 0 || fold_count > MAX_CATALOG_FOLDS {
        return Err(PcsError::new(
            ERR_LENGTH,
            format!("catalog fold count must be in 1..={MAX_CATALOG_FOLDS}"),
        ));
    }
    let coefficient_len = term_count
        .checked_mul(fold_count)
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "coefficient input length overflow"))?;
    if coefficient_bytes.len() != coefficient_len {
        return Err(PcsError::new(
            ERR_LENGTH,
            "catalog coefficient input has the wrong byte length",
        ));
    }
    if coefficient_bytes.iter().any(|value| *value >= 1 << 7) {
        return Err(PcsError::new(
            ERR_FORMAT,
            "catalog coefficient is not an unsigned 7-bit integer",
        ));
    }
    let bases = CATALOG_POINT_CACHE
        .get_or_init(|| Mutex::new(CatalogPointCache::default()))
        .lock()
        .map_err(|_| PcsError::new(ERR_INTERNAL, "catalog point cache is poisoned"))?
        .get(catalog_id)
        .ok_or_else(|| PcsError::new(ERR_FORMAT, "catalog point cache entry is unavailable"))?;
    if bases.len() != term_count {
        return Err(PcsError::new(
            ERR_LENGTH,
            "registered catalog term count does not match",
        ));
    }
    let _operation = operation_guard()?;
    let combined = in_worker_pool(|| {
        coefficient_bytes
            .par_chunks_exact(term_count)
            .map(|coefficients| {
                <pallas::Point as DlogGroupExt>::vartime_multiscalar_mul_small_with_max_num_bits(
                    coefficients,
                    bases.as_slice(),
                    7,
                )
                .to_bytes()
            })
            .collect::<Vec<_>>()
    })?;
    let mut output = Vec::with_capacity(fold_count * COMMITMENT_BYTES);
    for encoded in combined {
        output.extend_from_slice(encoded.as_ref());
    }
    Ok(output)
}

fn combine_commitments(
    commitment_bytes: &[u8],
    coefficient_bytes: &[u8],
    term_count: usize,
) -> Result<[u8; COMMITMENT_BYTES], PcsError> {
    combine_commitments_bounded(
        commitment_bytes,
        coefficient_bytes,
        term_count,
        MAX_COMBINE_TERMS,
    )
}

fn validate_commitment_encodings(
    commitment_bytes: &[u8],
    commitment_count: usize,
) -> Result<(), PcsError> {
    if commitment_count == 0 || commitment_count > MAX_VALIDATE_COMMITMENTS {
        return Err(PcsError::new(
            ERR_LENGTH,
            format!("commitment validation count must be in 1..={MAX_VALIDATE_COMMITMENTS}"),
        ));
    }
    let expected_len = commitment_count
        .checked_mul(COMMITMENT_BYTES)
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "commitment input length overflow"))?;
    if commitment_bytes.len() != expected_len {
        return Err(PcsError::new(
            ERR_LENGTH,
            "commitment validation input has the wrong byte length",
        ));
    }
    let valid = in_worker_pool(|| {
        commitment_bytes
            .par_chunks_exact(COMMITMENT_BYTES)
            .all(canonical_point)
    })?;
    if !valid {
        return Err(PcsError::new(ERR_FORMAT, "non-canonical Pallas commitment"));
    }
    Ok(())
}

fn preparse_proof(bytes: &[u8], log_n: usize) -> Result<(), PcsError> {
    let expected = proof_wire_len(log_n)?;
    if bytes.len() != expected || bytes.len() > MAX_PROOF_BYTES {
        return Err(PcsError::new(
            ERR_FORMAT,
            format!("proof must be exactly {expected} bytes"),
        ));
    }
    if bytes[0] != log_n as u8 {
        return Err(PcsError::new(ERR_FORMAT, "wrong IPA L-vector length"));
    }
    let mut offset = 1usize;
    for _ in 0..log_n {
        if !canonical_point(&bytes[offset..offset + COMMITMENT_BYTES]) {
            return Err(PcsError::new(ERR_FORMAT, "non-canonical IPA L point"));
        }
        offset += COMMITMENT_BYTES;
    }
    if bytes[offset] != log_n as u8 {
        return Err(PcsError::new(ERR_FORMAT, "wrong IPA R-vector length"));
    }
    offset += 1;
    for _ in 0..log_n {
        if !canonical_point(&bytes[offset..offset + COMMITMENT_BYTES]) {
            return Err(PcsError::new(ERR_FORMAT, "non-canonical IPA R point"));
        }
        offset += COMMITMENT_BYTES;
    }
    decode_scalar(&bytes[offset..offset + SCALAR_BYTES])?;
    Ok(())
}

fn decode_proof(bytes: &[u8], log_n: usize) -> Result<Proof, PcsError> {
    preparse_proof(bytes, log_n)?;
    let (proof, consumed) = bincode::serde::decode_from_slice::<Proof, _>(
        bytes,
        config::standard().with_limit::<MAX_PROOF_BYTES>(),
    )
    .map_err(|_| PcsError::new(ERR_FORMAT, "invalid IPA proof encoding"))?;
    if consumed != bytes.len() {
        return Err(PcsError::new(ERR_FORMAT, "trailing IPA proof bytes"));
    }
    let encoded = bincode::serde::encode_to_vec(&proof, config::standard())
        .map_err(|_| PcsError::new(ERR_INTERNAL, "IPA proof re-encoding failed"))?;
    if encoded != bytes {
        return Err(PcsError::new(
            ERR_FORMAT,
            "non-canonical IPA proof encoding",
        ));
    }
    Ok(proof)
}

fn opening_transcript(
    outer_digest: &[u8; 32],
    vector_len: usize,
    padded_len: usize,
    encoding: Encoding,
    point: &[Fr],
) -> Keccak256Transcript<E> {
    let mut transcript = Keccak256Transcript::<E>::new(TRANSCRIPT_DOMAIN);
    let metadata = [
        Fr::from(PROTOCOL_VERSION),
        Fr::from(vector_len as u64),
        Fr::from(padded_len as u64),
        Fr::from(encoding as u32 as u64),
    ];
    transcript.absorb(b"statement_metadata", &metadata.as_slice());

    // Inject all 256 caller-supplied bits without reducing the digest modulo
    // the scalar field: four little-endian u64 limbs are each embedded exactly.
    let mut limbs = [Fr::ZERO; 4];
    for (index, chunk) in outer_digest.chunks_exact(8).enumerate() {
        limbs[index] = Fr::from(u64::from_le_bytes(chunk.try_into().expect("8-byte chunk")));
    }
    transcript.absorb(b"outer_digest_le64", &limbs.as_slice());

    // Nova's IPA does not absorb its public equality vector internally, so
    // the wrapper must bind the exact MLE point before entering IPA.
    transcript.absorb(b"mle_point", &point);
    transcript
}

fn linear_opening_transcript(
    outer_digest: &[u8; 32],
    vector_len: usize,
    padded_len: usize,
    encoding: Encoding,
    coefficient_bytes: &[u8],
) -> Keccak256Transcript<E> {
    let mut transcript = Keccak256Transcript::<E>::new(LINEAR_TRANSCRIPT_DOMAIN);
    let metadata = [
        Fr::from(PROTOCOL_VERSION),
        Fr::from(vector_len as u64),
        Fr::from(padded_len as u64),
        Fr::from(encoding as u32 as u64),
    ];
    transcript.absorb(b"statement_metadata", &metadata.as_slice());

    let mut outer_limbs = [Fr::ZERO; 4];
    for (index, chunk) in outer_digest.chunks_exact(8).enumerate() {
        outer_limbs[index] =
            Fr::from(u64::from_le_bytes(chunk.try_into().expect("8-byte chunk")));
    }
    transcript.absorb(b"outer_digest_le64", &outer_limbs.as_slice());

    // The public coefficient vector can be large. Bind its exact canonical
    // scalar encoding without making it part of the proof wire.
    let coefficient_digest = sha256_array(&[coefficient_bytes]);
    let mut coefficient_limbs = [Fr::ZERO; 4];
    for (index, chunk) in coefficient_digest.chunks_exact(8).enumerate() {
        coefficient_limbs[index] =
            Fr::from(u64::from_le_bytes(chunk.try_into().expect("8-byte chunk")));
    }
    transcript.absorb(
        b"linear_coefficients_sha256_le64",
        &coefficient_limbs.as_slice(),
    );
    transcript
}

fn commit_poly(poly: &[Fr], padded_len: usize) -> Result<[u8; COMMITMENT_BYTES], PcsError> {
    let setup = setup_for(padded_len)?;
    let _operation = operation_guard()?;
    let commitment = in_worker_pool(|| CE::commit(&setup.ck, poly, &Fr::ZERO))?;
    encode_commitment(&commitment)
}

fn i8_offset_adjustment(setup: &Setup, vector_len: usize) -> Result<Commitment, PcsError> {
    let mut adjustments = setup
        .i8_offset_adjustments
        .lock()
        .map_err(|_| PcsError::new(ERR_INTERNAL, "PCS i8 offset cache is poisoned"))?;
    if let Some(cached) = adjustments.get(&vector_len) {
        return Ok(*cached);
    }
    let offsets = vec![128u8; vector_len];
    // Cache the already-negated correction. Multiplication by -1 is paid
    // once per vector length, rather than once per column.
    let created = in_worker_pool(|| CE::commit_small(&setup.ck, &offsets, &Fr::ZERO))? * (-Fr::ONE);
    adjustments.insert(vector_len, created);
    Ok(created)
}

fn commit_i8_with_setup(
    setup: &Setup,
    values: &[i8],
    adjustment: Commitment,
    shifted: &mut Vec<u8>,
) -> Result<[u8; COMMITMENT_BYTES], PcsError> {
    // Nova's small-scalar MSM only accepts unsigned integers. Shift each
    // signed coefficient into u8 and subtract the matching fixed commitment:
    //
    //   sum((s_i + 128) G_i) - sum(128 G_i) = sum(s_i G_i).
    //
    // This is exactly the regular field-scalar commitment with zero blinding.
    // Trailing power-of-two padding is omitted because every padded
    // coefficient is zero.
    shifted.clear();
    shifted.extend(values.iter().map(|value| ((*value as i16) + 128) as u8));
    let shifted_commitment = in_worker_pool(|| CE::commit_small(&setup.ck, shifted, &Fr::ZERO))?;
    encode_commitment(&(shifted_commitment + adjustment))
}

fn commit_i8_values(values: &[i8]) -> Result<[u8; COMMITMENT_BYTES], PcsError> {
    let (padded_len, _) = shape(values.len())?;
    let setup = setup_for(padded_len)?;
    let _operation = operation_guard()?;
    let adjustment = i8_offset_adjustment(&setup, values.len())?;
    let mut shifted = Vec::with_capacity(values.len());
    commit_i8_with_setup(&setup, values, adjustment, &mut shifted)
}

fn commit_i8_batch_values(
    values: &[i8],
    vector_len: usize,
    batch_count: usize,
    output: &mut [u8],
) -> Result<(), PcsError> {
    let (padded_len, _) = shape(vector_len)?;
    let setup = setup_for(padded_len)?;
    let _operation = operation_guard()?;
    let adjustment = i8_offset_adjustment(&setup, vector_len)?;
    let mut shifted = Vec::with_capacity(vector_len);
    for (column, encoded) in values
        .chunks_exact(vector_len)
        .zip(output.chunks_exact_mut(COMMITMENT_BYTES))
    {
        encoded.copy_from_slice(&commit_i8_with_setup(
            &setup,
            column,
            adjustment,
            &mut shifted,
        )?);
    }
    debug_assert_eq!(values.len(), vector_len * batch_count);
    debug_assert_eq!(output.len(), COMMITMENT_BYTES * batch_count);
    Ok(())
}

fn commit_i64_values(values: &[i64]) -> Result<[u8; COMMITMENT_BYTES], PcsError> {
    let (poly, padded_len) = poly_from_i64(values)?;
    commit_poly(&poly, padded_len)
}

fn evaluate_poly(poly: &[Fr], point: &[Fr]) -> Fr {
    MultilinearPolynomial::new(poly.to_vec()).evaluate(point)
}

fn evaluate_i8_matrix_values(
    values: &[i8],
    rows: usize,
    columns: usize,
    point: &[Fr],
) -> Result<Fr, PcsError> {
    let row_bits = exact_power_of_two(rows, "matrix row count")?;
    let column_bits = exact_power_of_two(columns, "matrix column count")?;
    let expected_values = rows
        .checked_mul(columns)
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "matrix size overflow"))?;
    if values.len() != expected_values || values.len() > MAX_BATCH_I8_BYTES {
        return Err(PcsError::new(
            ERR_LENGTH,
            "signed-i8 matrix buffer has the wrong length",
        ));
    }
    if point.len() != row_bits + column_bits {
        return Err(PcsError::new(
            ERR_LENGTH,
            "matrix MLE point has the wrong length",
        ));
    }

    let row_weights = EqPolynomial::new(point[..row_bits].to_vec()).evals();
    let column_weights = EqPolynomial::new(point[row_bits..].to_vec()).evals();
    in_worker_pool(|| {
        values
            .par_chunks_exact(columns)
            .zip(row_weights.par_iter())
            .map(|(row, row_weight)| {
                let row_evaluation = row.iter().zip(column_weights.iter()).fold(
                    Fr::ZERO,
                    |accumulator, (value, column_weight)| {
                        accumulator + signed_i64_to_scalar(*value as i64) * *column_weight
                    },
                );
                row_evaluation * *row_weight
            })
            .reduce(|| Fr::ZERO, |left, right| left + right)
    })
}

fn lean_projection_terminal_relation_value(
    values: &[i8],
    claim_count: usize,
    width: usize,
    point: &[Fr],
    alphas: &[Fr],
    terminal_fold_evaluations: &[Fr],
) -> Result<Fr, PcsError> {
    let (_, round_count) = shape(width)?;
    if claim_count == 0 || claim_count > MAX_LEAN_PROJECTION_MATERIAL_CLAIMS {
        return Err(PcsError::new(
            ERR_LENGTH,
            "lean projection terminal claim count is out of range",
        ));
    }
    let expected_values = claim_count
        .checked_mul(width)
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "terminal X size overflow"))?;
    if values.len() != expected_values || values.len() > MAX_LEAN_PROJECTION_BYTES {
        return Err(PcsError::new(
            ERR_LENGTH,
            "lean projection terminal X rows have the wrong length",
        ));
    }
    if point.len() != round_count {
        return Err(PcsError::new(
            ERR_LENGTH,
            "lean projection terminal point has the wrong length",
        ));
    }
    let fold_count = claim_count
        .checked_mul(LEAN_PROJECTION_FOLD_COUNT)
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "terminal fold count overflow"))?;
    if alphas.len() != fold_count || terminal_fold_evaluations.len() != fold_count {
        return Err(PcsError::new(
            ERR_LENGTH,
            "lean projection terminal scalar count is inconsistent",
        ));
    }

    let column_weights = EqPolynomial::new(point.to_vec()).evals();
    in_worker_pool(|| {
        values
            .par_chunks_exact(width)
            .enumerate()
            .map(|(claim_index, row)| {
                let x_evaluation = row.iter().zip(column_weights.iter()).fold(
                    Fr::ZERO,
                    |accumulator, (value, weight)| {
                        accumulator + signed_i64_to_scalar(*value as i64) * *weight
                    },
                );
                let fold_start = claim_index * LEAN_PROJECTION_FOLD_COUNT;
                let fold_weight = (0..LEAN_PROJECTION_FOLD_COUNT).fold(
                    Fr::ZERO,
                    |accumulator, fold_index| {
                        let index = fold_start + fold_index;
                        accumulator + alphas[index] * terminal_fold_evaluations[index]
                    },
                );
                x_evaluation * fold_weight
            })
            .reduce(|| Fr::ZERO, |left, right| left + right)
    })
}

struct Opening {
    commitment: [u8; COMMITMENT_BYTES],
    evaluation: [u8; SCALAR_BYTES],
    proof: Vec<u8>,
}

/// Transcript wrapper for values whose exact Nova representation is already
/// available. This is used only to mirror Nova's unchanged IPA transcript.
struct RawTranscriptRepr(Vec<u8>);

impl TranscriptReprTrait<pallas::Point> for RawTranscriptRepr {
    fn to_transcript_bytes(&self) -> Vec<u8> {
        self.0.clone()
    }
}

/// Serde-compatible mirror of Nova's private `InnerProductArgument` fields.
/// Bincode encodes structs by field order, and the experiment plus the tests
/// below require this encoding to remain byte-identical to Nova's proof.
#[derive(Deserialize, Serialize)]
struct OptimizedIpaProof {
    l_vec: Vec<pallas::Point>,
    r_vec: Vec<pallas::Point>,
    a_hat: Fr,
}

fn scalar_transcript_repr(value: &Fr) -> Vec<u8> {
    <Fr as TranscriptReprTrait<pallas::Point>>::to_transcript_bytes(value)
}

fn base_transcript_repr(value: &pallas::Base) -> Vec<u8> {
    <pallas::Base as TranscriptReprTrait<pallas::Point>>::to_transcript_bytes(value)
}

/// Match Nova's Pedersen commitment transcript representation:
/// affine x-coordinate, affine y-coordinate, and the infinity flag.
fn point_transcript_repr(point: &pallas::Point) -> Vec<u8> {
    let (x, y, is_infinity) = point.to_coordinates();
    let infinity: u8 = is_infinity.into();
    [
        base_transcript_repr(&x),
        base_transcript_repr(&y),
        vec![infinity],
    ]
    .concat()
}

fn ipa_inner_product(left: &[Fr], right: &[Fr]) -> Fr {
    debug_assert_eq!(left.len(), right.len());
    left.par_chunks(1024)
        .zip(right.par_chunks(1024))
        .map(|(left_chunk, right_chunk)| {
            left_chunk
                .iter()
                .zip(right_chunk.iter())
                .fold(Fr::ZERO, |acc, (left, right)| acc + *left * *right)
        })
        .reduce(|| Fr::ZERO, |left, right| left + right)
}

fn ipa_msm(scalars: &[Fr], bases: &[pallas::Affine]) -> pallas::Point {
    <pallas::Point as DlogGroupExt>::vartime_multiscalar_mul(scalars, bases)
}

/// Produce Nova's exact IPA argument without materializing a newly folded
/// commitment key at every round.
///
/// Nova folds each generator with a separate two-point MSM and converts every
/// result back to affine form. Instead, this prover keeps the accumulated
/// challenge coefficient for each original generator and computes each L/R
/// point as one Pippenger MSM over the original affine key. The witness folds,
/// transcript schedule, challenges, points, and final scalar are unchanged.
fn prove_ipa_optimized(
    ck: &CommitmentKey,
    ipa_evaluation_base: pallas::Affine,
    commitment: &Commitment,
    poly: &[Fr],
    point: &[Fr],
    evaluation: &Fr,
    transcript: &mut Keccak256Transcript<E>,
) -> Result<OptimizedIpaProof, PcsError> {
    let n = poly.len();
    if n == 0 || !n.is_power_of_two() || point.len() != n.trailing_zeros() as usize {
        return Err(PcsError::new(
            ERR_LENGTH,
            "optimized IPA dimensions are not canonical",
        ));
    }
    let bases = ck.generators();
    if bases.len() < n {
        return Err(PcsError::new(
            ERR_INTERNAL,
            "optimized IPA commitment key is too short",
        ));
    }

    let b_init = EqPolynomial::new(point.to_vec()).evals();
    prove_ipa_optimized_with_public_vector(
        ck,
        ipa_evaluation_base,
        commitment,
        poly,
        &b_init,
        evaluation,
        transcript,
    )
}

fn prove_ipa_optimized_with_public_vector(
    ck: &CommitmentKey,
    ipa_evaluation_base: pallas::Affine,
    commitment: &Commitment,
    poly: &[Fr],
    public_vector: &[Fr],
    evaluation: &Fr,
    transcript: &mut Keccak256Transcript<E>,
) -> Result<OptimizedIpaProof, PcsError> {
    let n = poly.len();
    if n == 0
        || !n.is_power_of_two()
        || public_vector.len() != n
        || ipa_inner_product(poly, public_vector) != *evaluation
    {
        return Err(PcsError::new(
            ERR_LENGTH,
            "optimized IPA public-vector dimensions or evaluation are invalid",
        ));
    }
    let bases = ck.generators();
    if bases.len() < n {
        return Err(PcsError::new(
            ERR_INTERNAL,
            "optimized IPA commitment key is too short",
        ));
    }

    transcript.dom_sep(b"IPA");
    let instance_repr = [
        commitment.to_transcript_bytes(),
        scalar_transcript_repr(evaluation),
    ]
    .concat();
    transcript.absorb(b"U", &RawTranscriptRepr(instance_repr));
    let randomizer = transcript
        .squeeze(b"r")
        .map_err(|_| PcsError::new(ERR_CRYPTO, "IPA transcript squeeze failed"))?;
    let evaluation_base = (pallas::Point::from(ipa_evaluation_base) * randomizer).affine();

    let rounds = n.trailing_zeros() as usize;
    let mut a = poly.to_vec();
    let mut b = public_vector.to_vec();
    let mut generator_weights = vec![Fr::ONE; n];
    let mut l_points = Vec::with_capacity(rounds);
    let mut r_points = Vec::with_capacity(rounds);
    let mut active_len = n;

    let mut l_scalars = Vec::with_capacity(n / 2 + 1);
    let mut l_bases = Vec::with_capacity(n / 2 + 1);
    let mut r_scalars = Vec::with_capacity(n / 2 + 1);
    let mut r_bases = Vec::with_capacity(n / 2 + 1);

    for _ in 0..rounds {
        let half = active_len / 2;
        let c_l = ipa_inner_product(&a[..half], &b[half..active_len]);
        let c_r = ipa_inner_product(&a[half..active_len], &b[..half]);

        l_scalars.clear();
        l_bases.clear();
        r_scalars.clear();
        r_bases.clear();
        for (index, base) in bases[..n].iter().enumerate() {
            let position = index & (active_len - 1);
            if position >= half {
                l_scalars.push(generator_weights[index] * a[position - half]);
                l_bases.push(*base);
            } else {
                r_scalars.push(generator_weights[index] * a[position + half]);
                r_bases.push(*base);
            }
        }
        l_scalars.push(c_l);
        l_bases.push(evaluation_base);
        r_scalars.push(c_r);
        r_bases.push(evaluation_base);

        let (l_point, r_point) = rayon::join(
            || ipa_msm(&l_scalars, &l_bases),
            || ipa_msm(&r_scalars, &r_bases),
        );

        transcript.absorb(b"L", &RawTranscriptRepr(point_transcript_repr(&l_point)));
        transcript.absorb(b"R", &RawTranscriptRepr(point_transcript_repr(&r_point)));
        let challenge = transcript
            .squeeze(b"r")
            .map_err(|_| PcsError::new(ERR_CRYPTO, "IPA transcript squeeze failed"))?;
        let challenge_inverse = Option::<Fr>::from(challenge.invert())
            .ok_or_else(|| PcsError::new(ERR_CRYPTO, "IPA transcript produced a zero challenge"))?;

        let (a_next, b_next) = rayon::join(
            || {
                (0..half)
                    .into_par_iter()
                    .map(|index| a[index] * challenge + challenge_inverse * a[index + half])
                    .collect::<Vec<_>>()
            },
            || {
                (0..half)
                    .into_par_iter()
                    .map(|index| b[index] * challenge_inverse + challenge * b[index + half])
                    .collect::<Vec<_>>()
            },
        );
        a = a_next;
        b = b_next;
        generator_weights
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, weight)| {
                let position = index & (active_len - 1);
                *weight *= if position >= half {
                    challenge
                } else {
                    challenge_inverse
                };
            });

        l_points.push(l_point);
        r_points.push(r_point);
        active_len = half;
    }

    Ok(OptimizedIpaProof {
        l_vec: l_points,
        r_vec: r_points,
        a_hat: a[0],
    })
}

fn decode_optimized_ipa_proof(
    bytes: &[u8],
    expected_rounds: usize,
) -> Result<OptimizedIpaProof, PcsError> {
    let (proof, consumed) = bincode::serde::decode_from_slice::<OptimizedIpaProof, _>(
        bytes,
        config::standard().with_limit::<4096>(),
    )
    .map_err(|_| PcsError::new(ERR_FORMAT, "invalid arbitrary-linear IPA proof"))?;
    if consumed != bytes.len()
        || proof.l_vec.len() != expected_rounds
        || proof.r_vec.len() != expected_rounds
        || expected_rounds >= 32
    {
        return Err(PcsError::new(
            ERR_FORMAT,
            "arbitrary-linear IPA proof has a wrong shape",
        ));
    }
    let encoded = bincode::serde::encode_to_vec(&proof, config::standard())
        .map_err(|_| PcsError::new(ERR_INTERNAL, "IPA proof re-encoding failed"))?;
    if encoded != bytes {
        return Err(PcsError::new(
            ERR_FORMAT,
            "non-canonical arbitrary-linear IPA proof encoding",
        ));
    }
    Ok(proof)
}

fn verify_ipa_with_public_vector(
    ck: &CommitmentKey,
    ipa_evaluation_base: pallas::Affine,
    commitment: pallas::Point,
    public_vector: &[Fr],
    evaluation: Fr,
    proof: &OptimizedIpaProof,
    transcript: &mut Keccak256Transcript<E>,
) -> Result<bool, PcsError> {
    let n = public_vector.len();
    if n == 0
        || !n.is_power_of_two()
        || proof.l_vec.len() != n.trailing_zeros() as usize
        || proof.r_vec.len() != proof.l_vec.len()
    {
        return Err(PcsError::new(
            ERR_LENGTH,
            "arbitrary-linear IPA verification shape is invalid",
        ));
    }
    let bases = ck.generators();
    if bases.len() < n {
        return Err(PcsError::new(
            ERR_INTERNAL,
            "arbitrary-linear IPA commitment key is too short",
        ));
    }

    transcript.dom_sep(b"IPA");
    let commitment_wrapper = decode_commitment(
        commitment
            .affine()
            .to_bytes()
            .as_ref(),
    )?;
    let instance_repr = [
        commitment_wrapper.to_transcript_bytes(),
        scalar_transcript_repr(&evaluation),
    ]
    .concat();
    transcript.absorb(b"U", &RawTranscriptRepr(instance_repr));
    let randomizer = transcript
        .squeeze(b"r")
        .map_err(|_| PcsError::new(ERR_CRYPTO, "IPA transcript squeeze failed"))?;
    let evaluation_base = pallas::Point::from(ipa_evaluation_base) * randomizer;
    let p = commitment + evaluation_base * evaluation;

    let mut challenges = Vec::with_capacity(proof.l_vec.len());
    for (left, right) in proof.l_vec.iter().zip(proof.r_vec.iter()) {
        transcript.absorb(b"L", &RawTranscriptRepr(point_transcript_repr(left)));
        transcript.absorb(b"R", &RawTranscriptRepr(point_transcript_repr(right)));
        challenges.push(
            transcript
                .squeeze(b"r")
                .map_err(|_| PcsError::new(ERR_CRYPTO, "IPA transcript squeeze failed"))?,
        );
    }
    let challenge_inverses = challenges
        .iter()
        .map(|value| {
            Option::<Fr>::from(value.invert()).ok_or_else(|| {
                PcsError::new(ERR_CRYPTO, "IPA transcript produced a zero challenge")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let challenge_squares = challenges
        .iter()
        .map(|value| *value * *value)
        .collect::<Vec<_>>();
    let inverse_squares = challenge_inverses
        .iter()
        .map(|value| *value * *value)
        .collect::<Vec<_>>();

    let mut s = vec![Fr::ZERO; n];
    s[0] = challenge_inverses
        .iter()
        .copied()
        .fold(Fr::ONE, |acc, value| acc * value);
    for index in 1..n {
        let position = (31 - (index as u32).leading_zeros()) as usize;
        s[index] = s[index - (1 << position)]
            * challenge_squares[(proof.l_vec.len() - 1) - position];
    }
    let ck_hat = ipa_msm(&s, &bases[..n]);
    let b_hat = ipa_inner_product(public_vector, &s);

    let mut p_scalars = Vec::with_capacity(proof.l_vec.len() * 2 + 1);
    p_scalars.extend(challenge_squares.iter().copied());
    p_scalars.extend(inverse_squares.iter().copied());
    p_scalars.push(Fr::ONE);
    let mut p_bases = Vec::with_capacity(p_scalars.len());
    p_bases.extend(proof.l_vec.iter().map(|point| point.affine()));
    p_bases.extend(proof.r_vec.iter().map(|point| point.affine()));
    p_bases.push(p.affine());
    let p_hat = ipa_msm(&p_scalars, &p_bases);

    let expected =
        ck_hat * proof.a_hat + evaluation_base * (proof.a_hat * b_hat);
    Ok(p_hat == expected)
}

fn prove_poly(
    poly: &[Fr],
    vector_len: usize,
    padded_len: usize,
    encoding: Encoding,
    point: &[Fr],
    outer_digest: &[u8; 32],
) -> Result<Opening, PcsError> {
    let setup = setup_for(padded_len)?;
    let _operation = operation_guard()?;
    let commitment = in_worker_pool(|| CE::commit(&setup.ck, poly, &Fr::ZERO))?;
    let evaluation = evaluate_poly(poly, point);
    let mut transcript = opening_transcript(outer_digest, vector_len, padded_len, encoding, point);
    let proof = in_worker_pool(|| {
        prove_ipa_optimized(
            &setup.ck,
            setup.ipa_evaluation_base,
            &commitment,
            poly,
            point,
            &evaluation,
            &mut transcript,
        )
    })??;
    let proof = bincode::serde::encode_to_vec(&proof, config::standard())
        .map_err(|_| PcsError::new(ERR_INTERNAL, "IPA proof encoding failed"))?;
    preparse_proof(&proof, padded_len.trailing_zeros() as usize)?;
    Ok(Opening {
        commitment: encode_commitment(&commitment)?,
        evaluation: encode_scalar(&evaluation),
        proof,
    })
}

fn decode_linear_coefficients(
    coefficient_bytes: &[u8],
    padded_len: usize,
) -> Result<Vec<Fr>, PcsError> {
    let expected = padded_len
        .checked_mul(SCALAR_BYTES)
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "linear coefficient length overflow"))?;
    if coefficient_bytes.len() != expected {
        return Err(PcsError::new(
            ERR_LENGTH,
            "linear opening coefficient vector has a wrong length",
        ));
    }
    coefficient_bytes
        .chunks_exact(SCALAR_BYTES)
        .map(decode_scalar)
        .collect()
}

fn prove_linear_poly(
    poly: &[Fr],
    vector_len: usize,
    padded_len: usize,
    encoding: Encoding,
    coefficient_bytes: &[u8],
    outer_digest: &[u8; 32],
) -> Result<Opening, PcsError> {
    let coefficients = decode_linear_coefficients(coefficient_bytes, padded_len)?;
    if poly.len() != padded_len {
        return Err(PcsError::new(
            ERR_LENGTH,
            "linear opening polynomial has a wrong length",
        ));
    }
    let setup = setup_for(padded_len)?;
    let _operation = operation_guard()?;
    let commitment = in_worker_pool(|| CE::commit(&setup.ck, poly, &Fr::ZERO))?;
    let evaluation = ipa_inner_product(poly, &coefficients);
    let mut transcript = linear_opening_transcript(
        outer_digest,
        vector_len,
        padded_len,
        encoding,
        coefficient_bytes,
    );
    let proof = in_worker_pool(|| {
        prove_ipa_optimized_with_public_vector(
            &setup.ck,
            setup.ipa_evaluation_base,
            &commitment,
            poly,
            &coefficients,
            &evaluation,
            &mut transcript,
        )
    })??;
    let proof = bincode::serde::encode_to_vec(&proof, config::standard())
        .map_err(|_| PcsError::new(ERR_INTERNAL, "IPA proof encoding failed"))?;
    decode_optimized_ipa_proof(&proof, padded_len.trailing_zeros() as usize)?;
    Ok(Opening {
        commitment: encode_commitment(&commitment)?,
        evaluation: encode_scalar(&evaluation),
        proof,
    })
}

fn verify_linear_opening(
    commitment_bytes: &[u8],
    vector_len: usize,
    encoding: Encoding,
    coefficient_bytes: &[u8],
    outer_digest: &[u8; 32],
    evaluation_bytes: &[u8],
    proof_bytes: &[u8],
) -> Result<bool, PcsError> {
    let (padded_len, log_n) = shape(vector_len)?;
    let coefficients = decode_linear_coefficients(coefficient_bytes, padded_len)?;
    let setup = setup_for(padded_len)?;
    let _operation = operation_guard()?;
    decode_commitment(commitment_bytes)?;
    let commitment = decode_pallas_point(commitment_bytes)
        .ok_or_else(|| PcsError::new(ERR_FORMAT, "non-canonical Pallas commitment"))?;
    let evaluation = decode_scalar(evaluation_bytes)?;
    let proof = decode_optimized_ipa_proof(proof_bytes, log_n)?;
    let mut transcript = linear_opening_transcript(
        outer_digest,
        vector_len,
        padded_len,
        encoding,
        coefficient_bytes,
    );
    in_worker_pool(|| {
        verify_ipa_with_public_vector(
            &setup.ck,
            setup.ipa_evaluation_base,
            commitment,
            &coefficients,
            evaluation,
            &proof,
            &mut transcript,
        )
    })?
}

fn verify_opening(
    commitment_bytes: &[u8],
    vector_len: usize,
    encoding: Encoding,
    point_bytes: &[u8],
    outer_digest: &[u8; 32],
    evaluation_bytes: &[u8],
    proof_bytes: &[u8],
) -> Result<bool, PcsError> {
    let (padded_len, log_n) = shape(vector_len)?;
    let point = decode_point(point_bytes, log_n)?;
    let setup = setup_for(padded_len)?;
    // Point decompression and proof parsing perform curve operations, so they
    // must share the same setup exclusion gate as proving and verification.
    let _operation = operation_guard()?;
    let commitment = decode_commitment(commitment_bytes)?;
    let evaluation = decode_scalar(evaluation_bytes)?;
    let proof = decode_proof(proof_bytes, log_n)?;
    let mut transcript = opening_transcript(outer_digest, vector_len, padded_len, encoding, &point);
    Ok(in_worker_pool(|| {
        EE::verify(
            &setup.vk,
            &mut transcript,
            &commitment,
            &point,
            &evaluation,
            &proof,
        )
    })?
    .is_ok())
}

fn gemm_batch_proof_wire_len(block_count: usize, inner: usize) -> Result<usize, PcsError> {
    if block_count == 0 || block_count > MAX_GEMM_BATCH_BLOCKS || block_count > u16::MAX as usize {
        return Err(PcsError::new(
            ERR_LENGTH,
            format!("GEMM batch block count must be in 1..={MAX_GEMM_BATCH_BLOCKS}"),
        ));
    }
    let (padded, rounds) = shape(inner)?;
    if padded != inner {
        return Err(PcsError::new(
            ERR_LENGTH,
            "GEMM batch inner dimension must be a power of two",
        ));
    }
    10usize
        .checked_add(
            (3usize
                .checked_mul(block_count)
                .and_then(|value| value.checked_add(4 * rounds))
                .ok_or_else(|| PcsError::new(ERR_LENGTH, "GEMM batch proof size overflow"))?)
            .checked_mul(SCALAR_BYTES)
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "GEMM batch proof size overflow"))?,
        )
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "GEMM batch proof size overflow"))
}

fn exact_power_of_two(value: usize, label: &str) -> Result<usize, PcsError> {
    if value == 0 || !value.is_power_of_two() {
        return Err(PcsError::new(
            ERR_LENGTH,
            format!("{label} must be a positive power of two"),
        ));
    }
    Ok(value.trailing_zeros() as usize)
}

fn gemm_batch_record(fields: &[(&[u8], &[u8])]) -> Vec<u8> {
    let capacity = fields
        .iter()
        .map(|(label, value)| 4 + label.len() + 8 + value.len())
        .sum();
    let mut encoded = Vec::with_capacity(capacity);
    for (label, value) in fields {
        encoded.extend_from_slice(&(label.len() as u32).to_le_bytes());
        encoded.extend_from_slice(label);
        encoded.extend_from_slice(&(value.len() as u64).to_le_bytes());
        encoded.extend_from_slice(value);
    }
    encoded
}

fn sha256_array(parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part);
    }
    hasher.finalize().into()
}

fn hash_fp16_blocks_impl(
    blocks: &[u8],
    block_count: usize,
    logical_vector_length: usize,
) -> Result<Vec<[u8; 32]>, PcsError> {
    if block_count == 0 || block_count > MAX_MERKLE_BUILD_LEAVES {
        return Err(PcsError::new(
            ERR_LENGTH,
            format!("FP16 block count must be in 1..={MAX_MERKLE_BUILD_LEAVES}"),
        ));
    }
    if logical_vector_length == 0 || logical_vector_length > MAX_VECTOR_LEN {
        return Err(PcsError::new(
            ERR_LENGTH,
            format!("FP16 vector length must be in 1..={MAX_VECTOR_LEN}"),
        ));
    }
    let block_bytes = logical_vector_length
        .checked_mul(2)
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "FP16 block size overflow"))?;
    let expected = block_count
        .checked_mul(block_bytes)
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "FP16 batch size overflow"))?;
    if blocks.len() != expected {
        return Err(PcsError::new(
            ERR_LENGTH,
            "FP16 batch input length is inconsistent",
        ));
    }
    let length = (logical_vector_length as u64).to_le_bytes();
    in_worker_pool(|| {
        blocks
            .par_chunks_exact(block_bytes)
            .map(|block| sha256_array(&[PCS_FP16_BLOCK_DOMAIN, &length, block]))
            .collect()
    })
}

fn merkle_padding_hash(context_digest: &[u8; 32], level: usize, node_index: usize) -> [u8; 32] {
    let level_byte = [level as u8];
    let index = (node_index as u64).to_le_bytes();
    let zero_u64 = 0u64.to_le_bytes();
    let encoding = [0u8];
    let zero_commitment = [0u8; 32];
    sha256_array(&[
        PCS_MERKLE_PADDING_DOMAIN,
        context_digest,
        &level_byte,
        &index,
        &zero_u64,
        &zero_u64,
        &encoding,
        &zero_commitment,
    ])
}

fn build_merkle_levels_impl(
    context_digest: &[u8; 32],
    commitments: &[u8],
    logical_lengths: &[u64],
    encoding_ids: &[u8],
) -> Result<Vec<u8>, PcsError> {
    let leaf_count = logical_lengths.len();
    if leaf_count == 0 || leaf_count > MAX_MERKLE_BUILD_LEAVES {
        return Err(PcsError::new(
            ERR_LENGTH,
            format!("Merkle leaf count must be in 1..={MAX_MERKLE_BUILD_LEAVES}"),
        ));
    }
    if encoding_ids.len() != leaf_count
        || commitments.len()
            != leaf_count
                .checked_mul(COMMITMENT_BYTES)
                .ok_or_else(|| PcsError::new(ERR_LENGTH, "commitment size overflow"))?
    {
        return Err(PcsError::new(
            ERR_LENGTH,
            "Merkle leaf input lengths are inconsistent",
        ));
    }
    let padded_leaf_count = leaf_count
        .checked_next_power_of_two()
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "Merkle leaf count overflow"))?;
    let total_nodes = padded_leaf_count
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "Merkle tree size overflow"))?;
    let output_bytes = total_nodes
        .checked_mul(COMMITMENT_BYTES)
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "Merkle output size overflow"))?;

    let mut current = Vec::with_capacity(padded_leaf_count);
    for index in 0..leaf_count {
        let logical_length = logical_lengths[index];
        if logical_length == 0 || logical_length > MAX_VECTOR_LEN as u64 {
            return Err(PcsError::new(
                ERR_LENGTH,
                format!("Merkle vector length must be in 1..={MAX_VECTOR_LEN}"),
            ));
        }
        let padded_length = logical_length
            .checked_next_power_of_two()
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "Merkle vector length overflow"))?;
        let encoding = encoding_ids[index];
        if !(1..=5).contains(&encoding) {
            return Err(PcsError::new(
                ERR_ENCODING,
                "Merkle leaf encoding is unsupported",
            ));
        }
        let index_bytes = (index as u64).to_le_bytes();
        let logical_bytes = logical_length.to_le_bytes();
        let padded_bytes = padded_length.to_le_bytes();
        let encoding_byte = [encoding];
        let commitment = &commitments[index * COMMITMENT_BYTES..(index + 1) * COMMITMENT_BYTES];
        current.push(sha256_array(&[
            PCS_MERKLE_LEAF_DOMAIN,
            context_digest,
            &index_bytes,
            &logical_bytes,
            &padded_bytes,
            &encoding_byte,
            commitment,
        ]));
    }
    for index in leaf_count..padded_leaf_count {
        current.push(merkle_padding_hash(context_digest, 0, index));
    }

    let mut output = Vec::with_capacity(output_bytes);
    for digest in &current {
        output.extend_from_slice(digest);
    }
    let mut level = 1usize;
    while current.len() > 1 {
        let mut parents = Vec::with_capacity(current.len() / 2);
        for index in (0..current.len()).step_by(2) {
            let node_index = index / 2;
            let digest = if node_index
                .checked_mul(1usize << level)
                .is_some_and(|first_leaf| first_leaf >= leaf_count)
            {
                merkle_padding_hash(context_digest, level, node_index)
            } else {
                let level_byte = [level as u8];
                let node_bytes = (node_index as u64).to_le_bytes();
                sha256_array(&[
                    PCS_MERKLE_NODE_DOMAIN,
                    context_digest,
                    &level_byte,
                    &node_bytes,
                    &current[index],
                    &current[index + 1],
                ])
            };
            parents.push(digest);
        }
        for digest in &parents {
            output.extend_from_slice(digest);
        }
        current = parents;
        level += 1;
    }
    debug_assert_eq!(output.len(), output_bytes);
    Ok(output)
}

struct GemmBatchTranscript {
    state: [u8; 32],
}

impl GemmBatchTranscript {
    fn new(outer_digest: &[u8; 32]) -> Self {
        let record = gemm_batch_record(&[(b"outer_digest", outer_digest)]);
        Self {
            state: sha256_array(&[GEMM_BATCH_TRANSCRIPT_DOMAIN, &record]),
        }
    }

    fn absorb_scalar(&mut self, label: &[u8], value: &Fr) {
        let encoded = encode_scalar(value);
        let record = gemm_batch_record(&[(label, &encoded)]);
        self.state = sha256_array(&[&self.state, &record]);
    }

    fn squeeze_scalar(&mut self, label: &[u8]) -> Result<Fr, PcsError> {
        let modulus = BigUint::parse_bytes(
            b"40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001",
            16,
        )
        .ok_or_else(|| PcsError::new(ERR_INTERNAL, "invalid Pallas modulus constant"))?;
        let two_256 = BigUint::from(1u8) << 256usize;
        let limit = &two_256 - (&two_256 % &modulus);
        for attempt in 0..=u32::MAX {
            let attempt_bytes = attempt.to_le_bytes();
            let challenge_record =
                gemm_batch_record(&[(b"challenge_label", label), (b"attempt", &attempt_bytes)]);
            let digest = sha256_array(&[&self.state, &challenge_record]);
            let candidate = BigUint::from_bytes_le(&digest);
            if candidate < limit {
                let reduced = candidate % &modulus;
                let mut scalar_bytes = [0u8; SCALAR_BYTES];
                let encoded = reduced.to_bytes_le();
                scalar_bytes[..encoded.len()].copy_from_slice(&encoded);
                let scalar = decode_scalar(&scalar_bytes)?;
                let nested = gemm_batch_record(&[
                    (b"label", label),
                    (b"digest", &digest),
                    (b"scalar", &scalar_bytes),
                ]);
                let absorbed = gemm_batch_record(&[(b"challenge", &nested)]);
                self.state = sha256_array(&[&self.state, &absorbed]);
                return Ok(scalar);
            }
        }
        Err(PcsError::new(
            ERR_INTERNAL,
            "GEMM batch challenge rejection sampling did not terminate",
        ))
    }
}

fn lean_projection_record(label: &[u8], value: &[u8]) -> Result<Vec<u8>, PcsError> {
    let label_len = u16::try_from(label.len())
        .map_err(|_| PcsError::new(ERR_LENGTH, "lean projection transcript label is too long"))?;
    let value_len = u64::try_from(value.len())
        .map_err(|_| PcsError::new(ERR_LENGTH, "lean projection transcript value is too long"))?;
    let mut encoded = Vec::with_capacity(2 + label.len() + 8 + value.len());
    encoded.extend_from_slice(&label_len.to_le_bytes());
    encoded.extend_from_slice(label);
    encoded.extend_from_slice(&value_len.to_le_bytes());
    encoded.extend_from_slice(value);
    Ok(encoded)
}

fn combine_registered_catalog_commitments_u31_batch(
    catalog_id: [u8; 32],
    coefficient_bytes: &[u8],
    term_count: usize,
    fold_count: usize,
) -> Result<Vec<u8>, PcsError> {
    if term_count == 0 || term_count > MAX_CATALOG_COMBINE_TERMS {
        return Err(PcsError::new(
            ERR_LENGTH,
            format!("catalog commitment term count must be in 1..={MAX_CATALOG_COMBINE_TERMS}"),
        ));
    }
    if fold_count == 0 || fold_count > MAX_CATALOG_FOLDS {
        return Err(PcsError::new(
            ERR_LENGTH,
            format!("catalog commitment fold count must be in 1..={MAX_CATALOG_FOLDS}"),
        ));
    }
    let row_bytes = term_count
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "catalog coefficient row overflow"))?;
    let coefficient_len = row_bytes
        .checked_mul(fold_count)
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "catalog coefficient input overflow"))?;
    if coefficient_bytes.len() != coefficient_len {
        return Err(PcsError::new(
            ERR_LENGTH,
            "catalog coefficient input has the wrong byte length",
        ));
    }
    let bases = CATALOG_POINT_CACHE
        .get_or_init(|| Mutex::new(CatalogPointCache::default()))
        .lock()
        .map_err(|_| PcsError::new(ERR_INTERNAL, "catalog point cache is poisoned"))?
        .get(catalog_id)
        .ok_or_else(|| PcsError::new(ERR_FORMAT, "catalog point cache entry is unavailable"))?;
    if bases.len() != term_count {
        return Err(PcsError::new(
            ERR_LENGTH,
            "registered catalog term count does not match",
        ));
    }
    let _operation = operation_guard()?;
    let combined = in_worker_pool(|| {
        coefficient_bytes
            .par_chunks_exact(row_bytes)
            .map(|row| {
                let coefficients = row
                    .chunks_exact(std::mem::size_of::<u32>())
                    .map(|raw| {
                        let value = u32::from_le_bytes(
                            raw.try_into().map_err(|_| {
                                PcsError::new(
                                    ERR_FORMAT,
                                    "malformed u31 catalog coefficient",
                                )
                            })?,
                        );
                        if value >= 1 << 31 {
                            return Err(PcsError::new(
                                ERR_FORMAT,
                                "catalog coefficient is not an unsigned 31-bit integer",
                            ));
                        }
                        Ok(value as u64)
                    })
                    .collect::<Result<Vec<_>, PcsError>>()?;
                Ok(
                    <pallas::Point as DlogGroupExt>::vartime_multiscalar_mul_small_with_max_num_bits(
                        &coefficients,
                        bases.as_slice(),
                        31,
                    ),
                )
            })
            .collect::<Result<Vec<_>, PcsError>>()
    })??;
    let mut encoded = Vec::with_capacity(fold_count * COMMITMENT_BYTES);
    for point in combined {
        encoded.extend_from_slice(point.to_bytes().as_ref());
    }
    Ok(encoded)
}

struct LeanProjectionClaimMaterial {
    coefficients: Vec<u32>,
    targets: [Fr; LEAN_PROJECTION_FOLD_COUNT],
}

fn lean_projection_fold_materials(
    statement_digest: &[u8; 32],
    validator_nonce: &[u8; 32],
    input_rows_i8: &[u8],
    surrogate_rows_i64_le: &[u8],
    claim_count: usize,
    input_dim: usize,
    output_dim: usize,
) -> Result<Vec<LeanProjectionClaimMaterial>, PcsError> {
    if claim_count == 0 || claim_count > MAX_LEAN_PROJECTION_MATERIAL_CLAIMS {
        return Err(PcsError::new(
            ERR_LENGTH,
            format!(
                "lean projection material claim count must be in 1..={MAX_LEAN_PROJECTION_MATERIAL_CLAIMS}"
            ),
        ));
    }
    if input_dim == 0
        || input_dim > MAX_VECTOR_LEN
        || output_dim == 0
        || output_dim > MAX_CATALOG_COMBINE_TERMS
    {
        return Err(PcsError::new(
            ERR_LENGTH,
            "lean projection material dimensions are out of range",
        ));
    }
    let expected_x = claim_count
        .checked_mul(input_dim)
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "lean projection X size overflow"))?;
    let output_cells = claim_count
        .checked_mul(output_dim)
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "lean projection output size overflow"))?;
    let expected_s = output_cells
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "lean projection output byte size overflow"))?;
    if input_rows_i8.len() != expected_x || surrogate_rows_i64_le.len() != expected_s {
        return Err(PcsError::new(
            ERR_LENGTH,
            "lean projection material buffers have the wrong length",
        ));
    }
    if expected_x
        .checked_add(expected_s)
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "lean projection material size overflow"))?
        > MAX_LEAN_PROJECTION_BYTES
    {
        return Err(PcsError::new(
            ERR_LENGTH,
            "lean projection material buffers exceed the native bound",
        ));
    }
    if !validator_nonce.iter().any(|value| *value != 0) {
        return Err(PcsError::new(
            ERR_FORMAT,
            "lean projection validator nonce must be nonzero",
        ));
    }

    let coefficient_bytes_per_claim = LEAN_PROJECTION_FOLD_COUNT
        .checked_mul(output_dim)
        .and_then(|value| value.checked_mul(std::mem::size_of::<u32>()))
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "lean projection coefficient size overflow"))?;
    let output_bytes_per_claim = output_dim
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "lean projection row size overflow"))?;
    let materials = in_worker_pool(|| {
        (0..claim_count)
            .into_par_iter()
            .map(|claim_index| {
                let x_start = claim_index * input_dim;
                let x_row = &input_rows_i8[x_start..x_start + input_dim];
                let output_start = claim_index * output_bytes_per_claim;
                let output_row =
                    &surrogate_rows_i64_le[output_start..output_start + output_bytes_per_claim];

                let mut shake = Shake256::default();
                sha3::digest::Update::update(&mut shake, LEAN_PROJECTION_FOLD_TRANSCRIPT_DOMAIN);
                sha3::digest::Update::update(&mut shake, statement_digest);
                sha3::digest::Update::update(&mut shake, validator_nonce);
                sha3::digest::Update::update(&mut shake, x_row);
                sha3::digest::Update::update(&mut shake, output_row);
                let mut reader = shake.finalize_xof();
                let mut coefficient_bytes = vec![0u8; coefficient_bytes_per_claim];
                reader.read(&mut coefficient_bytes);
                let coefficients = coefficient_bytes
                    .chunks_exact(std::mem::size_of::<u32>())
                    .map(|word| {
                        let mut raw = [0u8; 4];
                        raw.copy_from_slice(word);
                        u32::from_le_bytes(raw) & ((1u32 << 31) - 1)
                    })
                    .collect::<Vec<_>>();

                let outputs = output_row
                    .chunks_exact(std::mem::size_of::<i64>())
                    .map(|raw| {
                        let mut encoded = [0u8; 8];
                        encoded.copy_from_slice(raw);
                        i64::from_le_bytes(encoded)
                    })
                    .collect::<Vec<_>>();
                let mut targets = [Fr::ZERO; LEAN_PROJECTION_FOLD_COUNT];
                for (fold_index, coefficient_row) in
                    coefficients.chunks_exact(output_dim).enumerate()
                {
                    targets[fold_index] = outputs
                        .iter()
                        .zip(coefficient_row.iter())
                        .fold(Fr::ZERO, |accumulator, (output, coefficient)| {
                            accumulator
                                + signed_i64_to_scalar(*output)
                                    * Fr::from(*coefficient as u64)
                        });
                }
                Ok(LeanProjectionClaimMaterial {
                    coefficients,
                    targets,
                })
            })
            .collect::<Result<Vec<_>, PcsError>>()
    })??;

    Ok(materials)
}

fn lean_projection_claim_materials(
    statement_digest: &[u8; 32],
    validator_nonce: &[u8; 32],
    catalog_id: [u8; 32],
    input_rows_i8: &[u8],
    surrogate_rows_i64_le: &[u8],
    claim_count: usize,
    input_dim: usize,
    output_dim: usize,
) -> Result<Vec<u8>, PcsError> {
    let materials = lean_projection_fold_materials(
        statement_digest,
        validator_nonce,
        input_rows_i8,
        surrogate_rows_i64_le,
        claim_count,
        input_dim,
        output_dim,
    )?;
    let coefficient_bytes_per_claim = LEAN_PROJECTION_FOLD_COUNT
        .checked_mul(output_dim)
        .and_then(|value| value.checked_mul(std::mem::size_of::<u32>()))
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "lean projection coefficient size overflow"))?;
    let claims_per_batch = MAX_CATALOG_FOLDS / LEAN_PROJECTION_FOLD_COUNT;
    let mut commitments = vec![[[0u8; COMMITMENT_BYTES]; LEAN_PROJECTION_FOLD_COUNT]; claim_count];
    for offset in (0..claim_count).step_by(claims_per_batch) {
        let stop = (offset + claims_per_batch).min(claim_count);
        let mut packed = Vec::with_capacity(
            (stop - offset)
                .checked_mul(coefficient_bytes_per_claim)
                .ok_or_else(|| {
                    PcsError::new(ERR_LENGTH, "lean projection coefficient batch overflow")
                })?,
        );
        for material in &materials[offset..stop] {
            for coefficient in &material.coefficients {
                packed.extend_from_slice(&coefficient.to_le_bytes());
            }
        }
        let combined = combine_registered_catalog_commitments_u31_batch(
            catalog_id,
            &packed,
            output_dim,
            (stop - offset) * LEAN_PROJECTION_FOLD_COUNT,
        )?;
        for (claim_slot, claim_commitments) in combined
            .chunks_exact(LEAN_PROJECTION_FOLD_COUNT * COMMITMENT_BYTES)
            .enumerate()
        {
            for (fold_index, commitment) in
                claim_commitments.chunks_exact(COMMITMENT_BYTES).enumerate()
            {
                commitments[offset + claim_slot][fold_index].copy_from_slice(commitment);
            }
        }
    }

    let output_len = claim_count
        .checked_mul(2 * LEAN_PROJECTION_FOLD_COUNT * SCALAR_BYTES)
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "lean projection material output overflow"))?;
    let mut output = Vec::with_capacity(output_len);
    for (material, claim_commitments) in materials.iter().zip(commitments.iter()) {
        for target in &material.targets {
            output.extend_from_slice(&encode_scalar(target));
        }
        for commitment in claim_commitments {
            output.extend_from_slice(commitment);
        }
    }
    debug_assert_eq!(output.len(), output_len);
    Ok(output)
}

fn lean_projection_claim_targets(
    statement_digest: &[u8; 32],
    validator_nonce: &[u8; 32],
    input_rows_i8: &[u8],
    surrogate_rows_i64_le: &[u8],
    claim_count: usize,
    input_dim: usize,
    output_dim: usize,
) -> Result<Vec<u8>, PcsError> {
    let materials = lean_projection_fold_materials(
        statement_digest,
        validator_nonce,
        input_rows_i8,
        surrogate_rows_i64_le,
        claim_count,
        input_dim,
        output_dim,
    )?;
    let output_len = claim_count
        .checked_mul(LEAN_PROJECTION_FOLD_COUNT * SCALAR_BYTES)
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "lean projection target output overflow"))?;
    let mut output = Vec::with_capacity(output_len);
    for material in materials {
        for target in material.targets {
            output.extend_from_slice(&encode_scalar(&target));
        }
    }
    debug_assert_eq!(output.len(), output_len);
    Ok(output)
}

fn lean_projection_aggregate_commitment(
    statement_digest: &[u8; 32],
    validator_nonce: &[u8; 32],
    catalog_id: [u8; 32],
    input_rows_i8: &[u8],
    surrogate_rows_i64_le: &[u8],
    beta_bytes: &[u8],
    claim_count: usize,
    input_dim: usize,
    output_dim: usize,
) -> Result<[u8; COMMITMENT_BYTES], PcsError> {
    let materials = lean_projection_fold_materials(
        statement_digest,
        validator_nonce,
        input_rows_i8,
        surrogate_rows_i64_le,
        claim_count,
        input_dim,
        output_dim,
    )?;
    let beta_count = claim_count
        .checked_mul(LEAN_PROJECTION_FOLD_COUNT)
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "lean projection beta count overflow"))?;
    let expected_beta_bytes = beta_count
        .checked_mul(SCALAR_BYTES)
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "lean projection beta byte size overflow"))?;
    if beta_bytes.len() != expected_beta_bytes {
        return Err(PcsError::new(
            ERR_LENGTH,
            "lean projection aggregate betas have the wrong length",
        ));
    }
    let betas = beta_bytes
        .chunks_exact(SCALAR_BYTES)
        .map(decode_scalar)
        .collect::<Result<Vec<_>, PcsError>>()?;
    let bases = CATALOG_POINT_CACHE
        .get_or_init(|| Mutex::new(CatalogPointCache::default()))
        .lock()
        .map_err(|_| PcsError::new(ERR_INTERNAL, "catalog point cache is poisoned"))?
        .get(catalog_id)
        .ok_or_else(|| PcsError::new(ERR_FORMAT, "catalog point cache entry is unavailable"))?;
    if bases.len() != output_dim {
        return Err(PcsError::new(
            ERR_LENGTH,
            "registered catalog term count does not match",
        ));
    }
    let coefficients = in_worker_pool(|| {
        let mut aggregated = vec![Fr::ZERO; output_dim];
        aggregated
            .par_chunks_mut(512)
            .enumerate()
            .for_each(|(chunk_index, chunk)| {
                let start = chunk_index * 512;
                for (claim_index, material) in materials.iter().enumerate() {
                    for fold_index in 0..LEAN_PROJECTION_FOLD_COUNT {
                        let beta =
                            betas[claim_index * LEAN_PROJECTION_FOLD_COUNT + fold_index];
                        let row_start = fold_index * output_dim + start;
                        let coefficient_chunk =
                            &material.coefficients[row_start..row_start + chunk.len()];
                        for (accumulator, coefficient) in
                            chunk.iter_mut().zip(coefficient_chunk.iter())
                        {
                            *accumulator += beta * Fr::from(*coefficient as u64);
                        }
                    }
                }
            });
        aggregated
    })?;
    let _operation = operation_guard()?;
    let commitment = in_worker_pool(|| {
        <pallas::Point as DlogGroupExt>::vartime_multiscalar_mul(
            coefficients.as_slice(),
            bases.as_slice(),
        )
    })?;
    let encoded = commitment.to_bytes();
    let mut output = [0u8; COMMITMENT_BYTES];
    output.copy_from_slice(encoded.as_ref());
    Ok(output)
}

struct LeanProjectionTranscript {
    state: [u8; 32],
}

impl LeanProjectionTranscript {
    fn from_state(state: &[u8; 32]) -> Self {
        Self { state: *state }
    }

    fn absorb(&mut self, label: &[u8], value: &[u8]) -> Result<(), PcsError> {
        let record = lean_projection_record(label, value)?;
        self.state = sha256_array(&[&self.state, &record]);
        Ok(())
    }

    fn squeeze_scalar(&mut self, label: &[u8]) -> Result<Fr, PcsError> {
        let modulus = BigUint::parse_bytes(
            b"40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001",
            16,
        )
        .ok_or_else(|| PcsError::new(ERR_INTERNAL, "invalid Pallas modulus constant"))?;
        let two_256 = BigUint::from(1u8) << 256usize;
        let limit = &two_256 - (&two_256 % &modulus);
        for counter in 0..=u32::MAX {
            let challenge_label = lean_projection_record(b"challenge_label", label)?;
            let counter_record =
                lean_projection_record(b"counter", &counter.to_le_bytes())?;
            let digest = sha256_array(&[&self.state, &challenge_label, &counter_record]);
            let candidate = BigUint::from_bytes_le(&digest);
            if candidate < limit {
                let reduced = candidate % &modulus;
                let mut scalar_bytes = [0u8; SCALAR_BYTES];
                let encoded = reduced.to_bytes_le();
                scalar_bytes[..encoded.len()].copy_from_slice(&encoded);
                let scalar = decode_scalar(&scalar_bytes)?;
                let mut nested = lean_projection_record(b"label", label)?;
                nested.extend_from_slice(&lean_projection_record(b"digest", &digest)?);
                nested.extend_from_slice(&lean_projection_record(b"scalar", &scalar_bytes)?);
                self.absorb(b"challenge", &nested)?;
                return Ok(scalar);
            }
        }
        Err(PcsError::new(
            ERR_INTERNAL,
            "lean projection challenge rejection sampling did not terminate",
        ))
    }
}

fn lean_projection_native_output_capacity(
    vector_count: usize,
    width: usize,
) -> Result<usize, PcsError> {
    if vector_count == 0 || vector_count > MAX_LEAN_PROJECTION_VECTORS {
        return Err(PcsError::new(
            ERR_LENGTH,
            format!(
                "lean projection vector count must be in 1..={MAX_LEAN_PROJECTION_VECTORS}"
            ),
        ));
    }
    let (_, rounds) = shape(width)?;
    18usize
        .checked_add(
            rounds
                .checked_mul(3 * SCALAR_BYTES)
                .ok_or_else(|| {
                    PcsError::new(ERR_LENGTH, "lean projection proof size overflow")
                })?,
        )
        .and_then(|value| value.checked_add(vector_count * SCALAR_BYTES))
        .and_then(|value| value.checked_add(2 * SCALAR_BYTES + MAX_PROOF_BYTES))
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "lean projection proof size overflow"))
}

fn lean_projection_group_prove_impl(
    transcript_state: &[u8; 32],
    x_values_i8: &[u8],
    z_values_i64_le: &[u8],
    alpha_bytes: &[u8],
    target_bytes: &[u8],
    vector_count: usize,
    width: usize,
) -> Result<Vec<u8>, PcsError> {
    let profile = std::env::var_os("VERATHOS_PCS_PROFILE").is_some();
    let profile_started = std::time::Instant::now();
    let (padded_width, round_count) = shape(width)?;
    if padded_width != width {
        return Err(PcsError::new(
            ERR_LENGTH,
            "lean projection width must be a power of two",
        ));
    }
    lean_projection_native_output_capacity(vector_count, width)?;
    let element_count = vector_count
        .checked_mul(width)
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "lean projection element count overflow"))?;
    let x_len = element_count;
    let z_len = element_count
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "lean projection witness size overflow"))?;
    let scalar_len = vector_count
        .checked_mul(SCALAR_BYTES)
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "lean projection scalar size overflow"))?;
    if x_values_i8.len() != x_len
        || z_values_i64_le.len() != z_len
        || alpha_bytes.len() != scalar_len
        || target_bytes.len() != scalar_len
        || x_len.checked_add(z_len).unwrap_or(usize::MAX) > MAX_LEAN_PROJECTION_BYTES
    {
        return Err(PcsError::new(
            ERR_LENGTH,
            "lean projection native witness buffers have the wrong length",
        ));
    }
    let alphas = alpha_bytes
        .chunks_exact(SCALAR_BYTES)
        .map(decode_scalar)
        .collect::<Result<Vec<_>, _>>()?;
    let targets = target_bytes
        .chunks_exact(SCALAR_BYTES)
        .map(decode_scalar)
        .collect::<Result<Vec<_>, _>>()?;
    let z_row_bytes = width
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "lean projection row size overflow"))?;
    let mut z_unique_by_row = HashMap::<&[u8], usize>::with_capacity(vector_count);
    let mut z_vector_map = Vec::with_capacity(vector_count);
    let mut z_unique_sources = Vec::with_capacity(vector_count);
    for vector_index in 0..vector_count {
        let start = vector_index * z_row_bytes;
        let row = &z_values_i64_le[start..start + z_row_bytes];
        let unique_index = match z_unique_by_row.get(row) {
            Some(index) => *index,
            None => {
                let index = z_unique_sources.len();
                z_unique_by_row.insert(row, index);
                z_unique_sources.push(vector_index);
                index
            }
        };
        z_vector_map.push(unique_index);
    }
    drop(z_unique_by_row);
    let z_raw = z_values_i64_le
        .chunks_exact(std::mem::size_of::<i64>())
        .map(|chunk| {
            let mut encoded = [0u8; 8];
            encoded.copy_from_slice(chunk);
            i64::from_le_bytes(encoded)
        })
        .collect::<Vec<_>>();
    let mut vectors_by_z = vec![Vec::<usize>::new(); z_unique_sources.len()];
    for (vector_index, unique_index) in z_vector_map.iter().enumerate() {
        vectors_by_z[*unique_index].push(vector_index);
    }
    let alpha_multiples = in_worker_pool(|| {
        alphas
            .par_iter()
            .map(|alpha| {
                std::array::from_fn::<Fr, 256, _>(|raw| {
                    *alpha * signed_i64_to_scalar((raw as u8 as i8) as i64)
                })
            })
            .collect::<Vec<_>>()
    })?;
    // The transcript coefficients are public by this point.  For every
    // distinct authenticated static fold z, combine all matching X rows as
    // X_z = sum(alpha_i * X_i).  This is algebraically identical to the
    // original sum_i alpha_i * <X_i, z_i>, while reducing the sumcheck from
    // one pair per claim/fold to one pair per distinct static fold.  The
    // terminal values and PCS opening remain in their canonical original
    // vector order below, so this is a prover-only optimization.
    let mut x_fields = in_worker_pool(|| {
        (0..z_unique_sources.len() * width)
            .into_par_iter()
            .map(|flat_index| {
                let unique_index = flat_index / width;
                let offset = flat_index % width;
                vectors_by_z[unique_index]
                    .iter()
                    .fold(Fr::ZERO, |acc, vector_index| {
                        acc + alpha_multiples[*vector_index]
                            [x_values_i8[*vector_index * width + offset] as usize]
                    })
            })
            .collect::<Vec<_>>()
    })?;
    drop(alpha_multiples);
    drop(vectors_by_z);
    if profile {
        eprintln!(
            "[PCS-LEAN-PROVE] phase=aggregate-x seconds={:.3} vectors={} unique_static={}",
            profile_started.elapsed().as_secs_f64(),
            vector_count,
            z_unique_sources.len(),
        );
    }
    let sumcheck_started = std::time::Instant::now();
    let mut running = alphas
        .iter()
        .zip(targets.iter())
        .fold(Fr::ZERO, |acc, (alpha, target)| acc + *alpha * *target);
    let mut transcript = LeanProjectionTranscript::from_state(transcript_state);
    let mut rounds = Vec::with_capacity(round_count);
    let mut point = Vec::with_capacity(round_count);
    let mut z_fields: Option<Vec<Fr>> = None;
    let mut active_width = width;

    for round_index in 0..round_count {
        let half = active_width / 2;
        let (g0, g1, g2) = in_worker_pool(|| {
            (0..z_unique_sources.len())
                .into_par_iter()
                .map(|unique_index| {
                    let x_base = unique_index * active_width;
                    let z_base = if round_index == 0 {
                        z_unique_sources[unique_index] * width
                    } else {
                        unique_index * active_width
                    };
                    let mut local0 = Fr::ZERO;
                    let mut local1 = Fr::ZERO;
                    let mut local2 = Fr::ZERO;
                    for offset in 0..half {
                        let (z0, z1) = if round_index == 0 {
                            (
                                signed_i64_to_scalar(z_raw[z_base + offset]),
                                signed_i64_to_scalar(z_raw[z_base + half + offset]),
                            )
                        } else {
                            let zs = z_fields.as_ref().expect("folded z witness");
                            (
                                zs[z_base + offset],
                                zs[z_base + half + offset],
                            )
                        };
                        let x0 = x_fields[x_base + offset];
                        let x1 = x_fields[x_base + half + offset];
                        local0 += x0 * z0;
                        local1 += x1 * z1;
                        local2 += (x1 + x1 - x0) * (z1 + z1 - z0);
                    }
                    (local0, local1, local2)
                })
                .reduce(
                    || (Fr::ZERO, Fr::ZERO, Fr::ZERO),
                    |left, right| {
                        (left.0 + right.0, left.1 + right.1, left.2 + right.2)
                    },
                )
        })?;
        if g0 + g1 != running {
            return Err(PcsError::new(
                ERR_CRYPTO,
                "lean projection witness does not satisfy the full-row relation",
            ));
        }
        rounds.push((g0, g1, g2));
        let mut round_bytes = Vec::with_capacity(3 * SCALAR_BYTES);
        round_bytes.extend_from_slice(&encode_scalar(&g0));
        round_bytes.extend_from_slice(&encode_scalar(&g1));
        round_bytes.extend_from_slice(&encode_scalar(&g2));
        transcript.absorb(b"sumcheck_round", &round_bytes)?;
        let challenge = transcript.squeeze_scalar(&indexed_label(
            b"inner_challenge",
            round_index,
        ))?;
        point.push(challenge);

        let next_x_count = z_unique_sources.len() * half;
        let next_z_count = z_unique_sources.len() * half;
        let (next_x, next_z) = in_worker_pool(|| {
            rayon::join(
                || {
                    (0..next_x_count)
                        .into_par_iter()
                        .map(|flat_index| {
                            let offset = flat_index % half;
                            let unique_index = flat_index / half;
                            let base = unique_index * active_width;
                            let x0 = x_fields[base + offset];
                            let x1 = x_fields[base + half + offset];
                            x0 + challenge * (x1 - x0)
                        })
                        .collect::<Vec<_>>()
                },
                || {
                    (0..next_z_count)
                        .into_par_iter()
                        .map(|flat_index| {
                            let unique_index = flat_index / half;
                            let offset = flat_index % half;
                            let base = if round_index == 0 {
                                z_unique_sources[unique_index] * width
                            } else {
                                unique_index * active_width
                            };
                            let (z0, z1) = if round_index == 0 {
                                (
                                    signed_i64_to_scalar(z_raw[base + offset]),
                                    signed_i64_to_scalar(z_raw[base + half + offset]),
                                )
                            } else {
                                let zs = z_fields.as_ref().expect("folded z witness");
                                (zs[base + offset], zs[base + half + offset])
                            };
                            z0 + challenge * (z1 - z0)
                        })
                        .collect::<Vec<_>>()
                },
            )
        })?;
        x_fields = next_x;
        z_fields = Some(next_z);
        active_width = half;

        let two_inverse = Option::<Fr>::from(Fr::from(2u64).invert())
            .ok_or_else(|| PcsError::new(ERR_INTERNAL, "Pallas two is not invertible"))?;
        running = g0 * (challenge - Fr::ONE) * (challenge - Fr::from(2u64))
            * two_inverse
            - g1 * challenge * (challenge - Fr::from(2u64))
            + g2 * challenge * (challenge - Fr::ONE) * two_inverse;
    }
    if profile {
        eprintln!(
            "[PCS-LEAN-PROVE] phase=sumcheck seconds={:.3}",
            sumcheck_started.elapsed().as_secs_f64(),
        );
    }

    let terminal_z_unique = z_fields
        .as_ref()
        .ok_or_else(|| PcsError::new(ERR_INTERNAL, "lean projection fold is empty"))?;
    let terminal_z = z_vector_map
        .iter()
        .map(|unique_index| terminal_z_unique[*unique_index])
        .collect::<Vec<_>>();
    if terminal_z.len() != vector_count {
        return Err(PcsError::new(
            ERR_INTERNAL,
            "lean projection terminal fold count is inconsistent",
        ));
    }
    let mut terminal_bytes = Vec::with_capacity(vector_count * SCALAR_BYTES);
    for value in &terminal_z {
        terminal_bytes.extend_from_slice(&encode_scalar(value));
    }
    transcript.absorb(b"terminal_fold_evaluations", &terminal_bytes)?;
    let mut betas = Vec::with_capacity(vector_count);
    for index in 0..vector_count {
        betas.push(transcript.squeeze_scalar(&indexed_label(b"opening_beta", index))?);
    }
    let mut unique_betas = vec![Fr::ZERO; z_unique_sources.len()];
    for (vector_index, beta) in betas.iter().enumerate() {
        unique_betas[z_vector_map[vector_index]] += *beta;
    }
    let combined = in_worker_pool(|| {
        (0..width)
            .into_par_iter()
            .map(|index| {
                (0..z_unique_sources.len()).fold(Fr::ZERO, |acc, unique_index| {
                    acc + unique_betas[unique_index]
                        * signed_i64_to_scalar(
                            z_raw[z_unique_sources[unique_index] * width + index],
                        )
                })
            })
            .collect::<Vec<_>>()
    })?;
    let opening_outer_digest = sha256_array(&[
        LEAN_PROJECTION_BATCH_TRANSCRIPT_DOMAIN,
        b"/OPENING/",
        &transcript.state,
    ]);
    let opening_started = std::time::Instant::now();
    let opening = prove_poly(
        &combined,
        width,
        width,
        Encoding::PallasScalar,
        &point,
        &opening_outer_digest,
    )?;
    if profile {
        eprintln!(
            "[PCS-LEAN-PROVE] phase=ipa-opening seconds={:.3} total={:.3}",
            opening_started.elapsed().as_secs_f64(),
            profile_started.elapsed().as_secs_f64(),
        );
    }
    let expected_evaluation = betas
        .iter()
        .zip(terminal_z.iter())
        .fold(Fr::ZERO, |acc, (beta, value)| acc + *beta * *value);
    if decode_scalar(&opening.evaluation)? != expected_evaluation {
        return Err(PcsError::new(
            ERR_INTERNAL,
            "lean projection opening evaluation is inconsistent",
        ));
    }
    let proof_len = u16::try_from(opening.proof.len())
        .map_err(|_| PcsError::new(ERR_LENGTH, "lean projection IPA proof is too long"))?;
    let vector_count_u32 = u32::try_from(vector_count)
        .map_err(|_| PcsError::new(ERR_LENGTH, "lean projection vector count is too large"))?;
    let width_u32 = u32::try_from(width)
        .map_err(|_| PcsError::new(ERR_LENGTH, "lean projection width is too large"))?;
    let round_count_u16 = u16::try_from(round_count)
        .map_err(|_| PcsError::new(ERR_LENGTH, "lean projection round count is too large"))?;
    let mut encoded = Vec::with_capacity(
        lean_projection_native_output_capacity(vector_count, width)?,
    );
    encoded.extend_from_slice(LEAN_PROJECTION_NATIVE_MAGIC);
    encoded.extend_from_slice(&LEAN_PROJECTION_NATIVE_VERSION.to_le_bytes());
    encoded.extend_from_slice(&width_u32.to_le_bytes());
    encoded.extend_from_slice(&vector_count_u32.to_le_bytes());
    encoded.extend_from_slice(&round_count_u16.to_le_bytes());
    encoded.extend_from_slice(&proof_len.to_le_bytes());
    for (g0, g1, g2) in rounds {
        encoded.extend_from_slice(&encode_scalar(&g0));
        encoded.extend_from_slice(&encode_scalar(&g1));
        encoded.extend_from_slice(&encode_scalar(&g2));
    }
    encoded.extend_from_slice(&terminal_bytes);
    encoded.extend_from_slice(&opening.commitment);
    encoded.extend_from_slice(&opening.evaluation);
    encoded.extend_from_slice(&opening.proof);
    Ok(encoded)
}

fn indexed_label(prefix: &[u8], index: usize) -> Vec<u8> {
    let mut label = Vec::with_capacity(prefix.len() + 4);
    label.extend_from_slice(prefix);
    label.extend_from_slice(&(index as u32).to_le_bytes());
    label
}

fn fold_mle(mut values: Vec<Fr>, point: &[Fr]) -> Result<Fr, PcsError> {
    if values.len() != 1usize << point.len() {
        return Err(PcsError::new(
            ERR_LENGTH,
            "GEMM MLE table does not match its point",
        ));
    }
    for challenge in point {
        let half = values.len() / 2;
        for index in 0..half {
            values[index] = values[index] + *challenge * (values[index + half] - values[index]);
        }
        values.truncate(half);
    }
    Ok(values[0])
}

fn fold_table_once(values: &mut Vec<Fr>, challenge: Fr) {
    let half = values.len() / 2;
    for index in 0..half {
        values[index] = values[index] + challenge * (values[index + half] - values[index]);
    }
    values.truncate(half);
}

fn gemm_batch_prove_i8_impl(
    batch_digest: &[u8; 32],
    x_values: &[i8],
    w_values: &[i8],
    y_values: &[i64],
    valid_inner: &[u32],
    block_count: usize,
    rows: usize,
    inner: usize,
    columns: usize,
) -> Result<Vec<u8>, PcsError> {
    let row_bits = exact_power_of_two(rows, "GEMM batch rows")?;
    let inner_bits = exact_power_of_two(inner, "GEMM batch inner dimension")?;
    let column_bits = exact_power_of_two(columns, "GEMM batch columns")?;
    gemm_batch_proof_wire_len(block_count, inner)?;
    if valid_inner.len() != block_count {
        return Err(PcsError::new(ERR_LENGTH, "wrong valid-inner count"));
    }
    let x_per_block = rows
        .checked_mul(inner)
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "GEMM X size overflow"))?;
    let w_per_block = inner
        .checked_mul(columns)
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "GEMM W size overflow"))?;
    let y_per_block = rows
        .checked_mul(columns)
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "GEMM Y size overflow"))?;
    if x_values.len() != block_count * x_per_block
        || w_values.len() != block_count * w_per_block
        || y_values.len() != block_count * y_per_block
    {
        return Err(PcsError::new(
            ERR_LENGTH,
            "GEMM batch witness buffers have the wrong length",
        ));
    }
    let total_bytes = x_values
        .len()
        .checked_add(w_values.len())
        .and_then(|value| value.checked_add(y_values.len() * std::mem::size_of::<i64>()))
        .and_then(|value| value.checked_add(valid_inner.len() * 4))
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "GEMM batch witness size overflow"))?;
    if total_bytes > MAX_GEMM_BATCH_BYTES {
        return Err(PcsError::new(
            ERR_LENGTH,
            format!("GEMM batch witness exceeds {MAX_GEMM_BATCH_BYTES} bytes"),
        ));
    }

    let mut transcript = GemmBatchTranscript::new(batch_digest);
    let mut row_point = Vec::with_capacity(row_bits);
    for index in 0..row_bits {
        row_point.push(transcript.squeeze_scalar(&indexed_label(b"r_i", index))?);
    }
    let mut column_point = Vec::with_capacity(column_bits);
    for index in 0..column_bits {
        column_point.push(transcript.squeeze_scalar(&indexed_label(b"r_j", index))?);
    }

    let prepared_tables = (0..block_count)
        .into_par_iter()
        .map(|block| {
            let valid = valid_inner[block] as usize;
            if valid == 0 || valid > inner {
                return Err(PcsError::new(
                    ERR_LENGTH,
                    "GEMM valid-inner dimension is out of range",
                ));
            }
            let x_raw = &x_values[block * x_per_block..(block + 1) * x_per_block];
            let w_raw = &w_values[block * w_per_block..(block + 1) * w_per_block];
            let y_raw = &y_values[block * y_per_block..(block + 1) * y_per_block];
            for row in 0..rows {
                if x_raw[row * inner + valid..(row + 1) * inner]
                    .iter()
                    .any(|value| *value != 0)
                {
                    return Err(PcsError::new(ERR_FORMAT, "GEMM X padding is nonzero"));
                }
            }
            if w_raw[valid * columns..].iter().any(|value| *value != 0) {
                return Err(PcsError::new(ERR_FORMAT, "GEMM W padding is nonzero"));
            }

            let x_table = (0..inner)
                .into_par_iter()
                .map(|k| {
                    let values = (0..rows)
                        .map(|row| signed_i64_to_scalar(x_raw[row * inner + k] as i64))
                        .collect();
                    fold_mle(values, &row_point)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let w_table = (0..inner)
                .into_par_iter()
                .map(|k| {
                    let values = (0..columns)
                        .map(|column| signed_i64_to_scalar(w_raw[k * columns + column] as i64))
                        .collect();
                    fold_mle(values, &column_point)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let y_by_column = (0..columns)
                .into_par_iter()
                .map(|column| {
                    let values = (0..rows)
                        .map(|row| signed_i64_to_scalar(y_raw[row * columns + column]))
                        .collect();
                    fold_mle(values, &row_point)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let claimed_y = fold_mle(y_by_column, &column_point)?;
            let mask_table = (0..inner)
                .map(|index| if index < valid { Fr::ONE } else { Fr::ZERO })
                .collect::<Vec<_>>();
            Ok((x_table, w_table, mask_table, claimed_y))
        })
        .collect::<Result<Vec<_>, PcsError>>()?;
    let mut x_tables = Vec::with_capacity(block_count);
    let mut w_tables = Vec::with_capacity(block_count);
    let mut mask_tables = Vec::with_capacity(block_count);
    let mut claimed_y = Vec::with_capacity(block_count);
    for (x_table, w_table, mask_table, y_value) in prepared_tables {
        x_tables.push(x_table);
        w_tables.push(w_table);
        mask_tables.push(mask_table);
        claimed_y.push(y_value);
    }

    for (index, value) in claimed_y.iter().enumerate() {
        transcript.absorb_scalar(&indexed_label(b"claimed_y_eval", index), value);
    }
    let mut coefficients = Vec::with_capacity(block_count);
    coefficients.push(Fr::ONE);
    for index in 1..block_count {
        coefficients.push(transcript.squeeze_scalar(&indexed_label(b"relation_alpha", index))?);
    }

    let mut current_claim = Fr::ZERO;
    for (coefficient, value) in coefficients.iter().zip(&claimed_y) {
        current_claim += *coefficient * *value;
    }
    let mut rounds = Vec::<[Fr; 4]>::with_capacity(inner_bits);
    for round_index in 0..inner_bits {
        let half = x_tables[0].len() / 2;
        let aggregate = (0..block_count * half)
            .into_par_iter()
            .map(|flat_index| {
                let block = flat_index / half;
                let index = flat_index % half;
                let x0 = x_tables[block][index];
                let x_delta = x_tables[block][index + half] - x0;
                let w0 = w_tables[block][index];
                let w_delta = w_tables[block][index + half] - w0;
                let mask0 = mask_tables[block][index];
                let mask_delta = mask_tables[block][index + half] - mask0;
                let mut values = [Fr::ZERO; 4];
                for (point, output) in values.iter_mut().enumerate() {
                    let coordinate = Fr::from(point as u64);
                    let value = (x0 + coordinate * x_delta)
                        * (w0 + coordinate * w_delta)
                        * (mask0 + coordinate * mask_delta);
                    *output = coefficients[block] * value;
                }
                values
            })
            .reduce(
                || [Fr::ZERO; 4],
                |left, right| std::array::from_fn(|index| left[index] + right[index]),
            );
        if round_index == 0 && aggregate[0] + aggregate[1] != current_claim {
            return Err(PcsError::new(
                ERR_CRYPTO,
                "GEMM batch witness does not satisfy the claimed Y relation",
            ));
        }
        transcript.absorb_scalar(&indexed_label(b"round_e0", round_index), &aggregate[0]);
        transcript.absorb_scalar(&indexed_label(b"round_e1", round_index), &aggregate[1]);
        transcript.absorb_scalar(&indexed_label(b"round_e2", round_index), &aggregate[2]);
        transcript.absorb_scalar(&indexed_label(b"round_e3", round_index), &aggregate[3]);
        let challenge = transcript.squeeze_scalar(&indexed_label(b"r_k", round_index))?;
        x_tables
            .par_iter_mut()
            .zip(w_tables.par_iter_mut())
            .zip(mask_tables.par_iter_mut())
            .for_each(|((x_table, w_table), mask_table)| {
                fold_table_once(x_table, challenge);
                fold_table_once(w_table, challenge);
                fold_table_once(mask_table, challenge);
            });
        rounds.push(aggregate);
        current_claim = Fr::ZERO;
    }

    let expected_len = gemm_batch_proof_wire_len(block_count, inner)?;
    let mut output = Vec::with_capacity(expected_len);
    output.extend_from_slice(GEMM_BATCH_PROOF_MAGIC);
    output.extend_from_slice(&(PROTOCOL_VERSION as u16).to_le_bytes());
    output.extend_from_slice(&(block_count as u16).to_le_bytes());
    output.extend_from_slice(&(inner_bits as u16).to_le_bytes());
    for value in &claimed_y {
        output.extend_from_slice(&encode_scalar(value));
    }
    for round in &rounds {
        for value in round {
            output.extend_from_slice(&encode_scalar(value));
        }
    }
    for table in &x_tables {
        output.extend_from_slice(&encode_scalar(&table[0]));
    }
    for table in &w_tables {
        output.extend_from_slice(&encode_scalar(&table[0]));
    }
    debug_assert_eq!(output.len(), expected_len);
    Ok(output)
}

fn prove_i8_linear_combination_impl(
    values: &[i8],
    vector_len: usize,
    term_count: usize,
    coefficients_raw: &[u8],
    point: &[Fr],
    outer_digest: &[u8; 32],
) -> Result<Opening, PcsError> {
    if term_count == 0 || term_count > MAX_COMBINE_TERMS {
        return Err(PcsError::new(
            ERR_LENGTH,
            format!("linear-combination term count must be in 1..={MAX_COMBINE_TERMS}"),
        ));
    }
    let (padded_len, log_n) = shape(vector_len)?;
    if padded_len != vector_len || point.len() != log_n {
        return Err(PcsError::new(
            ERR_LENGTH,
            "linear-combination vector or point dimension is not canonical",
        ));
    }
    let expected_values = vector_len
        .checked_mul(term_count)
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "linear-combination size overflow"))?;
    if values.len() != expected_values || values.len() > MAX_BATCH_I8_BYTES {
        return Err(PcsError::new(
            ERR_LENGTH,
            "linear-combination signed-i8 buffer has the wrong length",
        ));
    }
    if coefficients_raw.len() != term_count * SCALAR_BYTES {
        return Err(PcsError::new(
            ERR_LENGTH,
            "linear-combination coefficient buffer has the wrong length",
        ));
    }
    let coefficients = coefficients_raw
        .chunks_exact(SCALAR_BYTES)
        .map(decode_scalar)
        .collect::<Result<Vec<_>, _>>()?;
    let poly = in_worker_pool(|| {
        let multiples = coefficients
            .par_iter()
            .map(|coefficient| {
                std::array::from_fn::<Fr, 256, _>(|raw| {
                    *coefficient * signed_i64_to_scalar((raw as u8 as i8) as i64)
                })
            })
            .collect::<Vec<_>>();
        (0..vector_len)
            .into_par_iter()
            .map(|index| {
                let mut value_offset = index;
                let mut sum = Fr::ZERO;
                for table in &multiples {
                    sum += table[values[value_offset] as u8 as usize];
                    value_offset += vector_len;
                }
                sum
            })
            .collect::<Vec<_>>()
    })?;
    prove_poly(
        &poly,
        vector_len,
        padded_len,
        Encoding::PallasScalar,
        point,
        outer_digest,
    )
}

fn prove_i64_linear_combination_linear_impl(
    values: &[i64],
    vector_len: usize,
    term_count: usize,
    combination_coefficients_raw: &[u8],
    linear_coefficients_raw: &[u8],
    outer_digest: &[u8; 32],
) -> Result<Opening, PcsError> {
    if term_count == 0 || term_count > MAX_COMBINE_TERMS {
        return Err(PcsError::new(
            ERR_LENGTH,
            format!("linear-combination term count must be in 1..={MAX_COMBINE_TERMS}"),
        ));
    }
    let (padded_len, _) = shape(vector_len)?;
    if padded_len != vector_len {
        return Err(PcsError::new(
            ERR_LENGTH,
            "linear-combination vector length must be a power of two",
        ));
    }
    let expected_values = vector_len
        .checked_mul(term_count)
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "linear-combination size overflow"))?;
    let value_bytes = values
        .len()
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or_else(|| PcsError::new(ERR_LENGTH, "linear-combination byte size overflow"))?;
    if values.len() != expected_values || value_bytes > MAX_BATCH_I64_BYTES {
        return Err(PcsError::new(
            ERR_LENGTH,
            "linear-combination signed-i64 buffer has the wrong length",
        ));
    }
    if combination_coefficients_raw.len() != term_count * SCALAR_BYTES {
        return Err(PcsError::new(
            ERR_LENGTH,
            "linear-combination coefficient buffer has the wrong length",
        ));
    }
    if linear_coefficients_raw.len() != padded_len * SCALAR_BYTES {
        return Err(PcsError::new(
            ERR_LENGTH,
            "linear-functional coefficient buffer has the wrong length",
        ));
    }
    let coefficients = combination_coefficients_raw
        .chunks_exact(SCALAR_BYTES)
        .map(decode_scalar)
        .collect::<Result<Vec<_>, _>>()?;
    let poly = in_worker_pool(|| {
        (0..vector_len)
            .into_par_iter()
            .map(|index| {
                coefficients
                    .iter()
                    .enumerate()
                    .fold(Fr::ZERO, |sum, (term, coefficient)| {
                        sum + *coefficient * signed_i64_to_scalar(values[term * vector_len + index])
                    })
            })
            .collect::<Vec<_>>()
    })?;
    prove_linear_poly(
        &poly,
        vector_len,
        padded_len,
        Encoding::SignedI64,
        linear_coefficients_raw,
        outer_digest,
    )
}

fn ffi_status<F>(operation: F) -> i32
where
    F: FnOnce() -> Result<(), PcsError>,
{
    clear_last_error();
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            set_last_error(error.message);
            error.code
        }
        Err(_) => {
            set_last_error("native PCS panic contained at FFI boundary");
            ERR_INTERNAL
        }
    }
}

fn ffi_verify<F>(operation: F) -> i32
where
    F: FnOnce() -> Result<bool, PcsError>,
{
    clear_last_error();
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(true)) => 1,
        Ok(Ok(false)) => 0,
        Ok(Err(error)) => {
            set_last_error(error.message);
            error.code
        }
        Err(_) => {
            set_last_error("native PCS panic contained at FFI boundary");
            ERR_INTERNAL
        }
    }
}

unsafe fn required_input<'a, T>(
    pointer: *const T,
    length: usize,
    label: &str,
) -> Result<&'a [T], PcsError> {
    if pointer.is_null() {
        return Err(PcsError::new(ERR_NULL, format!("null {label} pointer")));
    }
    // SAFETY: the C ABI requires the caller to provide a readable allocation
    // of `length` elements.  Null is rejected above and lengths are capped by
    // each exported entry point before this helper is called.
    Ok(unsafe { std::slice::from_raw_parts(pointer, length) })
}

unsafe fn optional_empty_input<'a, T>(
    pointer: *const T,
    length: usize,
    label: &str,
) -> Result<&'a [T], PcsError> {
    if length == 0 {
        return Ok(&[]);
    }
    // SAFETY: delegated to required_input with a non-zero length.
    unsafe { required_input(pointer, length, label) }
}

unsafe fn output_slice<'a>(
    pointer: *mut u8,
    length: usize,
    label: &str,
) -> Result<&'a mut [u8], PcsError> {
    if pointer.is_null() {
        return Err(PcsError::new(ERR_NULL, format!("null {label} pointer")));
    }
    // SAFETY: the C ABI requires a writable allocation of `length` bytes.
    Ok(unsafe { std::slice::from_raw_parts_mut(pointer, length) })
}

fn outer_array(bytes: &[u8]) -> Result<[u8; 32], PcsError> {
    bytes
        .try_into()
        .map_err(|_| PcsError::new(ERR_LENGTH, "outer digest must be exactly 32 bytes"))
}

#[no_mangle]
pub extern "C" fn verathos_pcs_v2_abi_version() -> u32 {
    ABI_VERSION
}

#[no_mangle]
pub extern "C" fn verathos_pcs_v2_max_vector_len() -> usize {
    MAX_VECTOR_LEN
}

#[no_mangle]
pub extern "C" fn verathos_pcs_v2_max_combine_terms() -> usize {
    MAX_COMBINE_TERMS
}

#[no_mangle]
pub extern "C" fn verathos_pcs_v2_max_catalog_combine_terms() -> usize {
    MAX_CATALOG_COMBINE_TERMS
}

#[no_mangle]
pub extern "C" fn verathos_pcs_v2_max_validate_commitments() -> usize {
    MAX_VALIDATE_COMMITMENTS
}

#[no_mangle]
pub extern "C" fn verathos_pcs_v2_max_batch_commitments() -> usize {
    MAX_BATCH_COMMITMENTS
}

#[no_mangle]
pub extern "C" fn verathos_pcs_v2_max_batch_i8_bytes() -> usize {
    MAX_BATCH_I8_BYTES
}

#[no_mangle]
pub extern "C" fn verathos_pcs_v2_worker_threads() -> usize {
    worker_pool()
        .map(ThreadPool::current_num_threads)
        .unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn verathos_pcs_v2_proof_size(vector_len: usize) -> usize {
    shape(vector_len)
        .and_then(|(_, log_n)| proof_wire_len(log_n))
        .unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn verathos_pcs_v2_gemm_batch_proof_size(block_count: usize, inner: usize) -> usize {
    gemm_batch_proof_wire_len(block_count, inner).unwrap_or(0)
}

/// Produces one transcript-bound aggregate sumcheck for equal padded GEMM blocks.
///
/// X is packed block-major as `blocks * rows * inner` signed bytes, W as
/// `blocks * inner * columns`, and Y as `blocks * rows * columns` signed i64s.
/// All dimensions and explicit buffer lengths are checked before dereferencing.
#[no_mangle]
pub unsafe extern "C" fn verathos_pcs_v2_gemm_batch_prove_i8(
    batch_digest: *const u8,
    x_values: *const i8,
    x_value_count: usize,
    w_values: *const i8,
    w_value_count: usize,
    y_values: *const i64,
    y_value_count: usize,
    valid_inner: *const u32,
    valid_inner_count: usize,
    block_count: usize,
    rows: usize,
    inner: usize,
    columns: usize,
    output_proof: *mut u8,
    proof_capacity: usize,
    output_proof_len: *mut usize,
) -> i32 {
    ffi_status(|| {
        let expected = gemm_batch_proof_wire_len(block_count, inner)?;
        exact_power_of_two(rows, "GEMM batch rows")?;
        exact_power_of_two(columns, "GEMM batch columns")?;
        if proof_capacity < expected {
            return Err(PcsError::new(
                ERR_BUFFER,
                "GEMM batch proof output buffer is too small",
            ));
        }
        if output_proof_len.is_null() {
            return Err(PcsError::new(
                ERR_NULL,
                "null GEMM batch proof-length output",
            ));
        }
        let expected_x = block_count
            .checked_mul(rows)
            .and_then(|value| value.checked_mul(inner))
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "GEMM X size overflow"))?;
        let expected_w = block_count
            .checked_mul(inner)
            .and_then(|value| value.checked_mul(columns))
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "GEMM W size overflow"))?;
        let expected_y = block_count
            .checked_mul(rows)
            .and_then(|value| value.checked_mul(columns))
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "GEMM Y size overflow"))?;
        if x_value_count != expected_x
            || w_value_count != expected_w
            || y_value_count != expected_y
            || valid_inner_count != block_count
        {
            return Err(PcsError::new(
                ERR_LENGTH,
                "GEMM batch explicit buffer lengths are inconsistent",
            ));
        }
        let digest_raw = unsafe { required_input(batch_digest, 32, "batch digest")? };
        let digest = outer_array(digest_raw)?;
        let x = unsafe { required_input(x_values, x_value_count, "GEMM X values")? };
        let w = unsafe { required_input(w_values, w_value_count, "GEMM W values")? };
        let y = unsafe { required_input(y_values, y_value_count, "GEMM Y values")? };
        let valid =
            unsafe { required_input(valid_inner, valid_inner_count, "GEMM valid-inner values")? };
        let proof = in_worker_pool(|| {
            gemm_batch_prove_i8_impl(&digest, x, w, y, valid, block_count, rows, inner, columns)
        })??;
        debug_assert_eq!(proof.len(), expected);
        unsafe {
            output_slice(output_proof, proof_capacity, "GEMM batch proof output")?[..proof.len()]
                .copy_from_slice(&proof);
            *output_proof_len = proof.len();
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn verathos_pcs_v2_lean_projection_group_proof_capacity(
    vector_count: usize,
    width: usize,
) -> usize {
    lean_projection_native_output_capacity(vector_count, width).unwrap_or(0)
}

/// Produces one transcript-compatible lean projection aggregate sumcheck and
/// Pallas IPA opening. The transcript state is the exact state after the
/// caller derives all claim-alpha challenges.
///
/// # Safety
///
/// Every input pointer must reference its explicit byte length. The output
/// pointer must reference `proof_capacity` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn verathos_pcs_v2_lean_projection_group_prove(
    transcript_state: *const u8,
    x_values_i8: *const u8,
    x_values_len: usize,
    z_values_i64_le: *const u8,
    z_values_len: usize,
    alpha_bytes: *const u8,
    alpha_bytes_len: usize,
    target_bytes: *const u8,
    target_bytes_len: usize,
    vector_count: usize,
    width: usize,
    output_proof: *mut u8,
    proof_capacity: usize,
    output_proof_len: *mut usize,
) -> i32 {
    ffi_status(|| {
        let required_capacity =
            lean_projection_native_output_capacity(vector_count, width)?;
        if proof_capacity < required_capacity {
            return Err(PcsError::new(
                ERR_BUFFER,
                "lean projection proof output buffer is too small",
            ));
        }
        if output_proof_len.is_null() {
            return Err(PcsError::new(
                ERR_NULL,
                "null lean projection proof-length output",
            ));
        }
        let state_raw =
            unsafe { required_input(transcript_state, 32, "lean projection transcript state")? };
        let state = outer_array(state_raw)?;
        let x = unsafe {
            required_input(x_values_i8, x_values_len, "lean projection X witness")?
        };
        let z = unsafe {
            required_input(
                z_values_i64_le,
                z_values_len,
                "lean projection folded-weight witness",
            )?
        };
        let alphas = unsafe {
            required_input(alpha_bytes, alpha_bytes_len, "lean projection alphas")?
        };
        let targets = unsafe {
            required_input(target_bytes, target_bytes_len, "lean projection targets")?
        };
        let proof = lean_projection_group_prove_impl(
            &state,
            x,
            z,
            alphas,
            targets,
            vector_count,
            width,
        )?;
        if proof.len() > proof_capacity {
            return Err(PcsError::new(
                ERR_BUFFER,
                "lean projection proof exceeded its output capacity",
            ));
        }
        unsafe {
            output_slice(
                output_proof,
                proof_capacity,
                "lean projection proof output",
            )?[..proof.len()]
                .copy_from_slice(&proof);
            *output_proof_len = proof.len();
        }
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn verathos_pcs_v2_lean_projection_group_prove_deduplicated(
    transcript_state: *const u8,
    unique_x_values_i8: *const u8,
    unique_x_values_len: usize,
    unique_x_count: usize,
    x_vector_map: *const u32,
    x_vector_map_len: usize,
    unique_z_values_i64_le: *const u8,
    unique_z_values_len: usize,
    unique_z_count: usize,
    z_vector_map: *const u32,
    z_vector_map_len: usize,
    alpha_bytes: *const u8,
    alpha_bytes_len: usize,
    target_bytes: *const u8,
    target_bytes_len: usize,
    vector_count: usize,
    width: usize,
    output_proof: *mut u8,
    proof_capacity: usize,
    output_proof_len: *mut usize,
) -> i32 {
    ffi_status(|| {
        let required_capacity =
            lean_projection_native_output_capacity(vector_count, width)?;
        if proof_capacity < required_capacity {
            return Err(PcsError::new(
                ERR_BUFFER,
                "lean projection proof output buffer is too small",
            ));
        }
        if output_proof_len.is_null() {
            return Err(PcsError::new(
                ERR_NULL,
                "null lean projection proof-length output",
            ));
        }
        if unique_x_count == 0
            || unique_x_count > vector_count
            || unique_z_count == 0
            || unique_z_count > vector_count
            || x_vector_map_len != vector_count
            || z_vector_map_len != vector_count
        {
            return Err(PcsError::new(
                ERR_LENGTH,
                "lean projection deduplicated inventory is malformed",
            ));
        }
        let expected_x_len = unique_x_count
            .checked_mul(width)
            .ok_or_else(|| {
                PcsError::new(
                    ERR_LENGTH,
                    "lean projection deduplicated X length overflow",
                )
            })?;
        let expected_z_len = unique_z_count
            .checked_mul(width)
            .and_then(|value| {
                value.checked_mul(std::mem::size_of::<i64>())
            })
            .ok_or_else(|| {
                PcsError::new(
                    ERR_LENGTH,
                    "lean projection deduplicated Z length overflow",
                )
            })?;
        if unique_x_values_len != expected_x_len
            || unique_z_values_len != expected_z_len
        {
            return Err(PcsError::new(
                ERR_LENGTH,
                "lean projection deduplicated witness buffers are malformed",
            ));
        }
        let state_raw = unsafe {
            required_input(
                transcript_state,
                32,
                "lean projection transcript state",
            )?
        };
        let state = outer_array(state_raw)?;
        let unique_x = unsafe {
            required_input(
                unique_x_values_i8,
                unique_x_values_len,
                "lean projection unique X witness",
            )?
        };
        let x_map = unsafe {
            required_input(
                x_vector_map,
                x_vector_map_len,
                "lean projection X vector map",
            )?
        };
        let unique_z = unsafe {
            required_input(
                unique_z_values_i64_le,
                unique_z_values_len,
                "lean projection unique Z witness",
            )?
        };
        let z_map = unsafe {
            required_input(
                z_vector_map,
                z_vector_map_len,
                "lean projection Z vector map",
            )?
        };
        if x_map.iter().any(|index| *index as usize >= unique_x_count)
            || z_map.iter().any(|index| *index as usize >= unique_z_count)
        {
            return Err(PcsError::new(
                ERR_LENGTH,
                "lean projection deduplicated vector map is out of range",
            ));
        }
        let alphas = unsafe {
            required_input(
                alpha_bytes,
                alpha_bytes_len,
                "lean projection alphas",
            )?
        };
        let targets = unsafe {
            required_input(
                target_bytes,
                target_bytes_len,
                "lean projection targets",
            )?
        };
        let mut x = Vec::with_capacity(
            vector_count.checked_mul(width).ok_or_else(|| {
                PcsError::new(
                    ERR_LENGTH,
                    "lean projection expanded X length overflow",
                )
            })?,
        );
        for index in x_map {
            let start = *index as usize * width;
            x.extend_from_slice(&unique_x[start..start + width]);
        }
        let z_row_bytes = width
            .checked_mul(std::mem::size_of::<i64>())
            .ok_or_else(|| {
                PcsError::new(
                    ERR_LENGTH,
                    "lean projection expanded Z row overflow",
                )
            })?;
        let mut z = Vec::with_capacity(
            vector_count.checked_mul(z_row_bytes).ok_or_else(|| {
                PcsError::new(
                    ERR_LENGTH,
                    "lean projection expanded Z length overflow",
                )
            })?,
        );
        for index in z_map {
            let start = *index as usize * z_row_bytes;
            z.extend_from_slice(&unique_z[start..start + z_row_bytes]);
        }
        let proof = lean_projection_group_prove_impl(
            &state,
            &x,
            &z,
            alphas,
            targets,
            vector_count,
            width,
        )?;
        if proof.len() > proof_capacity {
            return Err(PcsError::new(
                ERR_BUFFER,
                "lean projection proof exceeded its output capacity",
            ));
        }
        unsafe {
            output_slice(
                output_proof,
                proof_capacity,
                "lean projection proof output",
            )?[..proof.len()]
                .copy_from_slice(&proof);
            *output_proof_len = proof.len();
        }
        Ok(())
    })
}

/// Hashes a packed batch of equal-length canonical little-endian FP16 blocks.
#[no_mangle]
pub unsafe extern "C" fn verathos_pcs_v2_hash_fp16_blocks(
    blocks: *const u8,
    blocks_len: usize,
    block_count: usize,
    logical_vector_length: usize,
    output_hashes: *mut u8,
    output_capacity: usize,
) -> i32 {
    ffi_status(|| {
        let expected_output = block_count
            .checked_mul(COMMITMENT_BYTES)
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "FP16 hash output size overflow"))?;
        if output_capacity < expected_output {
            return Err(PcsError::new(
                ERR_BUFFER,
                "FP16 hash output buffer is too small",
            ));
        }
        let input = unsafe { required_input(blocks, blocks_len, "FP16 blocks")? };
        let hashes = hash_fp16_blocks_impl(input, block_count, logical_vector_length)?;
        let output = unsafe { output_slice(output_hashes, output_capacity, "FP16 hashes")? };
        for (index, digest) in hashes.iter().enumerate() {
            output[index * COMMITMENT_BYTES..(index + 1) * COMMITMENT_BYTES]
                .copy_from_slice(digest);
        }
        Ok(())
    })
}

/// Builds all unwrapped Merkle levels from canonical leaf metadata.
///
/// Output is level-major, beginning with the padded leaf level and ending
/// with the single top node.
#[no_mangle]
pub unsafe extern "C" fn verathos_pcs_v2_build_merkle_levels(
    context_digest: *const u8,
    commitments: *const u8,
    commitments_len: usize,
    logical_lengths: *const u64,
    logical_length_count: usize,
    encoding_ids: *const u8,
    encoding_count: usize,
    output_levels: *mut u8,
    output_capacity: usize,
    output_len: *mut usize,
) -> i32 {
    ffi_status(|| {
        if output_len.is_null() {
            return Err(PcsError::new(ERR_NULL, "null Merkle output-length pointer"));
        }
        let context_raw = unsafe { required_input(context_digest, 32, "Merkle context digest")? };
        let context: [u8; 32] = context_raw
            .try_into()
            .map_err(|_| PcsError::new(ERR_LENGTH, "invalid Merkle context digest"))?;
        let commitment_bytes =
            unsafe { required_input(commitments, commitments_len, "Merkle commitments")? };
        let lengths = unsafe {
            required_input(
                logical_lengths,
                logical_length_count,
                "Merkle logical lengths",
            )?
        };
        let encodings =
            unsafe { required_input(encoding_ids, encoding_count, "Merkle encodings")? };
        let encoded = build_merkle_levels_impl(&context, commitment_bytes, lengths, encodings)?;
        if output_capacity < encoded.len() {
            return Err(PcsError::new(
                ERR_BUFFER,
                "Merkle level output buffer is too small",
            ));
        }
        unsafe {
            output_slice(output_levels, output_capacity, "Merkle levels")?[..encoded.len()]
                .copy_from_slice(&encoded);
            *output_len = encoded.len();
        }
        Ok(())
    })
}

/// Copies the calling thread's last error as a NUL-terminated UTF-8 string.
/// Returns the full message length excluding the terminator.
///
/// # Safety
///
/// If `capacity` is non-zero, `output` must point to `capacity` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn verathos_pcs_v2_last_error(output: *mut u8, capacity: usize) -> usize {
    LAST_ERROR.with(|slot| {
        let message = slot.borrow();
        if capacity > 0 && !output.is_null() {
            let count = message.len().min(capacity - 1);
            // SAFETY: caller provides `capacity` writable bytes and count is bounded.
            unsafe {
                ptr::copy_nonoverlapping(message.as_ptr(), output, count);
                *output.add(count) = 0;
            }
        }
        message.len()
    })
}

macro_rules! export_vector_functions {
    ($commit_fn:ident, $evaluate_fn:ident, $prove_fn:ident, $linear_prove_fn:ident, $ty:ty, $poly_fn:ident, $commit_impl:ident, $encoding:expr) => {
        /// Commits to one signed vector using the encoding named by this symbol.
        ///
        /// # Safety
        ///
        /// `values` must point to `vector_len` readable elements and
        /// `output_commitment` must point to 32 writable bytes.
        #[no_mangle]
        pub unsafe extern "C" fn $commit_fn(
            values: *const $ty,
            vector_len: usize,
            output_commitment: *mut u8,
        ) -> i32 {
            ffi_status(|| {
                shape(vector_len)?;
                // SAFETY: vector length was capped and the pointer contract is documented by the ABI.
                let values = unsafe { required_input(values, vector_len, "values")? };
                let commitment = $commit_impl(values)?;
                // SAFETY: fixed-size output pointer is required by the ABI.
                let output = unsafe {
                    output_slice(output_commitment, COMMITMENT_BYTES, "commitment output")?
                };
                output.copy_from_slice(&commitment);
                Ok(())
            })
        }

        /// Evaluates one signed vector's multilinear extension.
        ///
        /// # Safety
        ///
        /// `values` must point to `vector_len` readable elements,
        /// `point_bytes` must point to `point_count * 32` readable bytes (or
        /// may be null when `point_count` is zero), and `output_evaluation`
        /// must point to 32 writable bytes.
        #[no_mangle]
        pub unsafe extern "C" fn $evaluate_fn(
            values: *const $ty,
            vector_len: usize,
            point_bytes: *const u8,
            point_count: usize,
            output_evaluation: *mut u8,
        ) -> i32 {
            ffi_status(|| {
                let (_, log_n) = shape(vector_len)?;
                if point_count != log_n {
                    return Err(PcsError::new(ERR_LENGTH, "wrong MLE point length"));
                }
                // SAFETY: all lengths are capped before pointer conversion.
                let values = unsafe { required_input(values, vector_len, "values")? };
                let point_raw = unsafe {
                    optional_empty_input(point_bytes, point_count * SCALAR_BYTES, "point")?
                };
                let point = decode_point(point_raw, log_n)?;
                let (poly, _) = $poly_fn(values)?;
                let evaluation = encode_scalar(&evaluate_poly(&poly, &point));
                // SAFETY: fixed-size output pointer is required by the ABI.
                let output =
                    unsafe { output_slice(output_evaluation, SCALAR_BYTES, "evaluation output")? };
                output.copy_from_slice(&evaluation);
                Ok(())
            })
        }

        /// Commits to and opens one signed vector's multilinear extension.
        ///
        /// # Safety
        ///
        /// `values`, `point_bytes`, and `outer_digest` must reference readable
        /// buffers of the lengths implied by their arguments. `point_bytes`
        /// may be null only when `point_count` is zero. The three output
        /// buffers and `output_proof_len` must be writable for their declared
        /// sizes and must not overlap any input or one another.
        #[no_mangle]
        pub unsafe extern "C" fn $prove_fn(
            values: *const $ty,
            vector_len: usize,
            point_bytes: *const u8,
            point_count: usize,
            outer_digest: *const u8,
            output_commitment: *mut u8,
            output_evaluation: *mut u8,
            output_proof: *mut u8,
            proof_capacity: usize,
            output_proof_len: *mut usize,
        ) -> i32 {
            ffi_status(|| {
                let (padded_len, log_n) = shape(vector_len)?;
                if point_count != log_n {
                    return Err(PcsError::new(ERR_LENGTH, "wrong MLE point length"));
                }
                let expected_proof = proof_wire_len(log_n)?;
                if proof_capacity < expected_proof {
                    return Err(PcsError::new(
                        ERR_BUFFER,
                        "proof output buffer is too small",
                    ));
                }
                if output_proof_len.is_null() {
                    return Err(PcsError::new(ERR_NULL, "null proof-length output pointer"));
                }
                // SAFETY: all lengths are capped before pointer conversion.
                let values = unsafe { required_input(values, vector_len, "values")? };
                let point_raw = unsafe {
                    optional_empty_input(point_bytes, point_count * SCALAR_BYTES, "point")?
                };
                let outer_raw = unsafe { required_input(outer_digest, 32, "outer digest")? };
                let point = decode_point(point_raw, log_n)?;
                let outer = outer_array(outer_raw)?;
                let (poly, actual_padded_len) = $poly_fn(values)?;
                debug_assert_eq!(padded_len, actual_padded_len);
                let opening = prove_poly(&poly, vector_len, padded_len, $encoding, &point, &outer)?;
                // SAFETY: fixed-size outputs and the checked proof capacity are ABI requirements.
                unsafe {
                    output_slice(output_commitment, COMMITMENT_BYTES, "commitment output")?
                        .copy_from_slice(&opening.commitment);
                    output_slice(output_evaluation, SCALAR_BYTES, "evaluation output")?
                        .copy_from_slice(&opening.evaluation);
                    output_slice(output_proof, proof_capacity, "proof output")?
                        [..opening.proof.len()]
                        .copy_from_slice(&opening.proof);
                    *output_proof_len = opening.proof.len();
                }
                Ok(())
            })
        }

        /// Commits to and opens an arbitrary public linear functional of one
        /// signed vector. Coefficients are canonical padded Pallas scalars and
        /// are transcript-bound but not included in the proof payload.
        #[no_mangle]
        pub unsafe extern "C" fn $linear_prove_fn(
            values: *const $ty,
            vector_len: usize,
            coefficient_bytes: *const u8,
            coefficient_count: usize,
            outer_digest: *const u8,
            output_commitment: *mut u8,
            output_evaluation: *mut u8,
            output_proof: *mut u8,
            proof_capacity: usize,
            output_proof_len: *mut usize,
        ) -> i32 {
            ffi_status(|| {
                let (padded_len, log_n) = shape(vector_len)?;
                if coefficient_count != padded_len {
                    return Err(PcsError::new(
                        ERR_LENGTH,
                        "linear opening coefficient count is wrong",
                    ));
                }
                let expected_proof = proof_wire_len(log_n)?;
                if proof_capacity < expected_proof {
                    return Err(PcsError::new(
                        ERR_BUFFER,
                        "proof output buffer is too small",
                    ));
                }
                if output_proof_len.is_null() {
                    return Err(PcsError::new(
                        ERR_NULL,
                        "null proof-length output pointer",
                    ));
                }
                let values = unsafe { required_input(values, vector_len, "values")? };
                let coefficient_len = coefficient_count
                    .checked_mul(SCALAR_BYTES)
                    .ok_or_else(|| {
                        PcsError::new(ERR_LENGTH, "linear coefficient length overflow")
                    })?;
                let coefficient_raw = unsafe {
                    required_input(
                        coefficient_bytes,
                        coefficient_len,
                        "linear coefficients",
                    )?
                };
                let outer_raw =
                    unsafe { required_input(outer_digest, 32, "outer digest")? };
                let outer = outer_array(outer_raw)?;
                let (poly, actual_padded_len) = $poly_fn(values)?;
                debug_assert_eq!(padded_len, actual_padded_len);
                let opening = prove_linear_poly(
                    &poly,
                    vector_len,
                    padded_len,
                    $encoding,
                    coefficient_raw,
                    &outer,
                )?;
                unsafe {
                    output_slice(
                        output_commitment,
                        COMMITMENT_BYTES,
                        "commitment output",
                    )?
                    .copy_from_slice(&opening.commitment);
                    output_slice(
                        output_evaluation,
                        SCALAR_BYTES,
                        "evaluation output",
                    )?
                    .copy_from_slice(&opening.evaluation);
                    output_slice(output_proof, proof_capacity, "proof output")?
                        [..opening.proof.len()]
                        .copy_from_slice(&opening.proof);
                    *output_proof_len = opening.proof.len();
                }
                Ok(())
            })
        }
    };
}

export_vector_functions!(
    verathos_pcs_v2_commit_i8,
    verathos_pcs_v2_evaluate_i8,
    verathos_pcs_v2_prove_i8,
    verathos_pcs_v3_prove_linear_i8,
    i8,
    poly_from_i8,
    commit_i8_values,
    Encoding::SignedI8
);

/// Commits to a packed batch of equal-length signed-i8 vectors.
///
/// Vectors are contiguous in batch-major order. The output contains one
/// canonical 32-byte commitment per vector in the same order.
///
/// # Safety
///
/// `values` must point to exactly `values_len` readable i8 elements and
/// `output_commitments` must point to exactly `output_len` writable bytes.
/// The input and output allocations must not overlap.
#[no_mangle]
pub unsafe extern "C" fn verathos_pcs_v2_commit_i8_batch(
    values: *const i8,
    values_len: usize,
    vector_len: usize,
    batch_count: usize,
    output_commitments: *mut u8,
    output_len: usize,
) -> i32 {
    ffi_status(|| {
        shape(vector_len)?;
        if batch_count == 0 || batch_count > MAX_BATCH_COMMITMENTS {
            return Err(PcsError::new(
                ERR_LENGTH,
                format!("i8 commitment batch count must be in 1..={MAX_BATCH_COMMITMENTS}"),
            ));
        }
        let expected_values = vector_len
            .checked_mul(batch_count)
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "i8 batch input length overflow"))?;
        if expected_values > MAX_BATCH_I8_BYTES || values_len != expected_values {
            return Err(PcsError::new(
                ERR_LENGTH,
                "i8 batch input has the wrong byte length",
            ));
        }
        let expected_output = batch_count
            .checked_mul(COMMITMENT_BYTES)
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "i8 batch output length overflow"))?;
        if output_len != expected_output {
            return Err(PcsError::new(
                ERR_BUFFER,
                "i8 batch output has the wrong byte length",
            ));
        }
        // SAFETY: all lengths were checked and bounded above.
        let values = unsafe { required_input(values, values_len, "i8 batch values")? };
        let output =
            unsafe { output_slice(output_commitments, output_len, "i8 batch commitment output")? };
        commit_i8_batch_values(values, vector_len, batch_count, output)
    })
}

/// Evaluates one row-major signed-i8 matrix multilinear extension.
///
/// The point is ordered as row coordinates followed by column coordinates.
///
/// # Safety
///
/// `values` must point to `values_len` readable signed bytes,
/// `point_bytes` must point to `point_count * 32` readable bytes (or may be
/// null when `point_count` is zero), and `output_evaluation` must point to
/// 32 writable bytes.
#[no_mangle]
pub unsafe extern "C" fn verathos_pcs_v2_evaluate_i8_matrix(
    values: *const i8,
    values_len: usize,
    rows: usize,
    columns: usize,
    point_bytes: *const u8,
    point_count: usize,
    output_evaluation: *mut u8,
) -> i32 {
    ffi_status(|| {
        let row_bits = exact_power_of_two(rows, "matrix row count")?;
        let column_bits = exact_power_of_two(columns, "matrix column count")?;
        if point_count != row_bits + column_bits {
            return Err(PcsError::new(
                ERR_LENGTH,
                "matrix MLE point has the wrong length",
            ));
        }
        let expected_values = rows
            .checked_mul(columns)
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "matrix size overflow"))?;
        if values_len != expected_values || values_len > MAX_BATCH_I8_BYTES {
            return Err(PcsError::new(
                ERR_LENGTH,
                "signed-i8 matrix buffer has the wrong length",
            ));
        }
        // SAFETY: all lengths are bounded before pointer conversion.
        let values = unsafe { required_input(values, values_len, "matrix values")? };
        let point_raw = unsafe {
            optional_empty_input(point_bytes, point_count * SCALAR_BYTES, "matrix point")?
        };
        let point = decode_point(point_raw, point_count)?;
        let evaluation = evaluate_i8_matrix_values(values, rows, columns, &point)?;
        // SAFETY: the ABI requires one fixed-size writable output.
        let output = unsafe { output_slice(output_evaluation, SCALAR_BYTES, "evaluation output")? };
        output.copy_from_slice(&encode_scalar(&evaluation));
        Ok(())
    })
}

/// Evaluates the terminal relation of one lean projection width group.
///
/// # Safety
///
/// `values` must contain `claim_count * width` signed bytes. Point, alpha,
/// and terminal buffers contain canonical 32-byte scalar encodings with
/// lengths derived from `width` and `claim_count`.
#[no_mangle]
pub unsafe extern "C" fn verathos_pcs_v2_lean_projection_terminal_relation(
    values: *const i8,
    values_len: usize,
    claim_count: usize,
    width: usize,
    point_bytes: *const u8,
    point_count: usize,
    alpha_bytes: *const u8,
    alpha_bytes_len: usize,
    terminal_bytes: *const u8,
    terminal_bytes_len: usize,
    output_evaluation: *mut u8,
) -> i32 {
    ffi_status(|| {
        let (_, round_count) = shape(width)?;
        if point_count != round_count {
            return Err(PcsError::new(
                ERR_LENGTH,
                "lean projection terminal point count is inconsistent",
            ));
        }
        let expected_values = claim_count
            .checked_mul(width)
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "terminal X size overflow"))?;
        if values_len != expected_values || values_len > MAX_LEAN_PROJECTION_BYTES {
            return Err(PcsError::new(
                ERR_LENGTH,
                "lean projection terminal X rows have the wrong length",
            ));
        }
        let scalar_count = claim_count
            .checked_mul(LEAN_PROJECTION_FOLD_COUNT)
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "terminal scalar count overflow"))?;
        let expected_scalar_bytes = scalar_count
            .checked_mul(SCALAR_BYTES)
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "terminal scalar size overflow"))?;
        if alpha_bytes_len != expected_scalar_bytes
            || terminal_bytes_len != expected_scalar_bytes
        {
            return Err(PcsError::new(
                ERR_LENGTH,
                "lean projection terminal scalar buffers have the wrong length",
            ));
        }
        let values = unsafe { required_input(values, values_len, "terminal X rows")? };
        let point_raw = unsafe {
            optional_empty_input(
                point_bytes,
                point_count * SCALAR_BYTES,
                "terminal point",
            )?
        };
        let alpha_raw =
            unsafe { required_input(alpha_bytes, alpha_bytes_len, "terminal alphas")? };
        let terminal_raw = unsafe {
            required_input(
                terminal_bytes,
                terminal_bytes_len,
                "terminal fold evaluations",
            )?
        };
        let point = decode_point(point_raw, point_count)?;
        let alphas = alpha_raw
            .chunks_exact(SCALAR_BYTES)
            .map(decode_scalar)
            .collect::<Result<Vec<_>, PcsError>>()?;
        let terminal = terminal_raw
            .chunks_exact(SCALAR_BYTES)
            .map(decode_scalar)
            .collect::<Result<Vec<_>, PcsError>>()?;
        let evaluation = lean_projection_terminal_relation_value(
            values,
            claim_count,
            width,
            &point,
            &alphas,
            &terminal,
        )?;
        let output =
            unsafe { output_slice(output_evaluation, SCALAR_BYTES, "terminal evaluation")? };
        output.copy_from_slice(&encode_scalar(&evaluation));
        Ok(())
    })
}

export_vector_functions!(
    verathos_pcs_v2_commit_i64,
    verathos_pcs_v2_evaluate_i64,
    verathos_pcs_v2_prove_i64,
    verathos_pcs_v3_prove_linear_i64,
    i64,
    poly_from_i64,
    commit_i64_values,
    Encoding::SignedI64
);

/// Commits to one vector of canonical little-endian Pallas scalars.
///
/// # Safety
///
/// `values` must point to `vector_len * 32` readable bytes and
/// `output_commitment` must point to 32 writable bytes.
#[no_mangle]
pub unsafe extern "C" fn verathos_pcs_v2_commit_field(
    values: *const u8,
    vector_len: usize,
    output_commitment: *mut u8,
) -> i32 {
    ffi_status(|| {
        shape(vector_len)?;
        let values_len = vector_len
            .checked_mul(SCALAR_BYTES)
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "field-vector byte length overflow"))?;
        // SAFETY: vector length and its byte length were capped above.
        let values = unsafe { required_input(values, values_len, "field values")? };
        let (poly, padded_len) = poly_from_scalar_bytes(values, vector_len)?;
        let commitment = commit_poly(&poly, padded_len)?;
        // SAFETY: fixed-size output pointer is required by the ABI.
        let output =
            unsafe { output_slice(output_commitment, COMMITMENT_BYTES, "commitment output")? };
        output.copy_from_slice(&commitment);
        Ok(())
    })
}

/// Evaluates one canonical Pallas field vector's multilinear extension.
///
/// # Safety
///
/// `values` must point to `vector_len * 32` readable bytes. `point_bytes`
/// must point to `point_count * 32` readable bytes, or may be null when
/// `point_count` is zero. `output_evaluation` must point to 32 writable bytes.
#[no_mangle]
pub unsafe extern "C" fn verathos_pcs_v2_evaluate_field(
    values: *const u8,
    vector_len: usize,
    point_bytes: *const u8,
    point_count: usize,
    output_evaluation: *mut u8,
) -> i32 {
    ffi_status(|| {
        let (_, log_n) = shape(vector_len)?;
        if point_count != log_n {
            return Err(PcsError::new(ERR_LENGTH, "wrong MLE point length"));
        }
        let values_len = vector_len
            .checked_mul(SCALAR_BYTES)
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "field-vector byte length overflow"))?;
        // SAFETY: all lengths are capped before pointer conversion.
        let values = unsafe { required_input(values, values_len, "field values")? };
        let point_raw =
            unsafe { optional_empty_input(point_bytes, point_count * SCALAR_BYTES, "point")? };
        let point = decode_point(point_raw, log_n)?;
        let (poly, _) = poly_from_scalar_bytes(values, vector_len)?;
        let evaluation = encode_scalar(&evaluate_poly(&poly, &point));
        // SAFETY: fixed-size output pointer is required by the ABI.
        let output = unsafe { output_slice(output_evaluation, SCALAR_BYTES, "evaluation output")? };
        output.copy_from_slice(&evaluation);
        Ok(())
    })
}

/// Commits to and opens one canonical Pallas field vector.
///
/// # Safety
///
/// `values`, `point_bytes`, and `outer_digest` must reference readable
/// buffers of the lengths implied by their arguments. `point_bytes` may be
/// null only when `point_count` is zero. The three output buffers and
/// `output_proof_len` must be writable for their declared sizes and must not
/// overlap any input or one another.
#[no_mangle]
pub unsafe extern "C" fn verathos_pcs_v2_prove_field(
    values: *const u8,
    vector_len: usize,
    point_bytes: *const u8,
    point_count: usize,
    outer_digest: *const u8,
    output_commitment: *mut u8,
    output_evaluation: *mut u8,
    output_proof: *mut u8,
    proof_capacity: usize,
    output_proof_len: *mut usize,
) -> i32 {
    ffi_status(|| {
        let (padded_len, log_n) = shape(vector_len)?;
        if point_count != log_n {
            return Err(PcsError::new(ERR_LENGTH, "wrong MLE point length"));
        }
        let expected_proof = proof_wire_len(log_n)?;
        if proof_capacity < expected_proof {
            return Err(PcsError::new(
                ERR_BUFFER,
                "proof output buffer is too small",
            ));
        }
        if output_proof_len.is_null() {
            return Err(PcsError::new(ERR_NULL, "null proof-length output pointer"));
        }
        let values_len = vector_len
            .checked_mul(SCALAR_BYTES)
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "field-vector byte length overflow"))?;
        // SAFETY: all lengths are capped before pointer conversion.
        let values = unsafe { required_input(values, values_len, "field values")? };
        let point_raw =
            unsafe { optional_empty_input(point_bytes, point_count * SCALAR_BYTES, "point")? };
        let outer_raw = unsafe { required_input(outer_digest, 32, "outer digest")? };
        let point = decode_point(point_raw, log_n)?;
        let outer = outer_array(outer_raw)?;
        let (poly, actual_padded_len) = poly_from_scalar_bytes(values, vector_len)?;
        debug_assert_eq!(padded_len, actual_padded_len);
        let opening = prove_poly(
            &poly,
            vector_len,
            padded_len,
            Encoding::PallasScalar,
            &point,
            &outer,
        )?;
        // SAFETY: fixed-size outputs and checked proof capacity are ABI requirements.
        unsafe {
            output_slice(output_commitment, COMMITMENT_BYTES, "commitment output")?
                .copy_from_slice(&opening.commitment);
            output_slice(output_evaluation, SCALAR_BYTES, "evaluation output")?
                .copy_from_slice(&opening.evaluation);
            output_slice(output_proof, proof_capacity, "proof output")?[..opening.proof.len()]
                .copy_from_slice(&opening.proof);
            *output_proof_len = opening.proof.len();
        }
        Ok(())
    })
}

/// Opens a transcript-derived linear combination of equal-length signed-i8
/// vectors as one canonical Pallas-field polynomial.
#[no_mangle]
pub unsafe extern "C" fn verathos_pcs_v2_prove_i8_linear_combination(
    values: *const i8,
    value_count: usize,
    vector_len: usize,
    term_count: usize,
    coefficients: *const u8,
    coefficient_count: usize,
    point_bytes: *const u8,
    point_count: usize,
    outer_digest: *const u8,
    output_commitment: *mut u8,
    output_evaluation: *mut u8,
    output_proof: *mut u8,
    proof_capacity: usize,
    output_proof_len: *mut usize,
) -> i32 {
    ffi_status(|| {
        let (padded_len, log_n) = shape(vector_len)?;
        if padded_len != vector_len || point_count != log_n {
            return Err(PcsError::new(
                ERR_LENGTH,
                "linear-combination vector or point dimension is not canonical",
            ));
        }
        let expected_values = vector_len
            .checked_mul(term_count)
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "linear-combination size overflow"))?;
        let expected_coefficients = term_count
            .checked_mul(SCALAR_BYTES)
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "coefficient size overflow"))?;
        if value_count != expected_values || coefficient_count != expected_coefficients {
            return Err(PcsError::new(
                ERR_LENGTH,
                "linear-combination explicit buffer lengths are inconsistent",
            ));
        }
        let expected_proof = proof_wire_len(log_n)?;
        if proof_capacity < expected_proof || output_proof_len.is_null() {
            return Err(PcsError::new(
                ERR_BUFFER,
                "linear-combination proof output is too small",
            ));
        }
        let values = unsafe { required_input(values, value_count, "linear values")? };
        let coefficients =
            unsafe { required_input(coefficients, coefficient_count, "linear coefficients")? };
        let point_raw =
            unsafe { optional_empty_input(point_bytes, point_count * SCALAR_BYTES, "point")? };
        let point = decode_point(point_raw, log_n)?;
        let outer_raw = unsafe { required_input(outer_digest, 32, "outer digest")? };
        let opening = prove_i8_linear_combination_impl(
            values,
            vector_len,
            term_count,
            coefficients,
            &point,
            &outer_array(outer_raw)?,
        )?;
        unsafe {
            output_slice(output_commitment, COMMITMENT_BYTES, "commitment output")?
                .copy_from_slice(&opening.commitment);
            output_slice(output_evaluation, SCALAR_BYTES, "evaluation output")?
                .copy_from_slice(&opening.evaluation);
            output_slice(output_proof, proof_capacity, "proof output")?[..opening.proof.len()]
                .copy_from_slice(&opening.proof);
            *output_proof_len = opening.proof.len();
        }
        Ok(())
    })
}

/// Opens one arbitrary public linear functional of a transcript-derived
/// linear combination of equal-length signed-i64 vectors.
///
/// # Safety
///
/// Every input pointer must reference the exact readable byte/element count
/// supplied by the adjacent length argument. Output buffers must be writable
/// for their declared sizes and must not overlap any input.
#[no_mangle]
pub unsafe extern "C" fn verathos_pcs_v3_prove_i64_linear_combination_linear(
    values: *const i64,
    value_count: usize,
    vector_len: usize,
    term_count: usize,
    combination_coefficients: *const u8,
    combination_coefficient_count: usize,
    linear_coefficients: *const u8,
    linear_coefficient_count: usize,
    outer_digest: *const u8,
    output_commitment: *mut u8,
    output_evaluation: *mut u8,
    output_proof: *mut u8,
    proof_capacity: usize,
    output_proof_len: *mut usize,
) -> i32 {
    ffi_status(|| {
        let (padded_len, log_n) = shape(vector_len)?;
        if padded_len != vector_len {
            return Err(PcsError::new(
                ERR_LENGTH,
                "linear-combination vector length must be a power of two",
            ));
        }
        let expected_values = vector_len
            .checked_mul(term_count)
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "linear-combination size overflow"))?;
        let expected_combination_coefficients = term_count
            .checked_mul(SCALAR_BYTES)
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "coefficient size overflow"))?;
        if value_count != expected_values
            || combination_coefficient_count != expected_combination_coefficients
            || linear_coefficient_count != padded_len * SCALAR_BYTES
        {
            return Err(PcsError::new(
                ERR_LENGTH,
                "linear-combination explicit buffer lengths are inconsistent",
            ));
        }
        let expected_proof = proof_wire_len(log_n)?;
        if proof_capacity < expected_proof || output_proof_len.is_null() {
            return Err(PcsError::new(
                ERR_BUFFER,
                "linear-combination proof output is too small",
            ));
        }
        let values = unsafe { required_input(values, value_count, "linear values")? };
        let combination_coefficients = unsafe {
            required_input(
                combination_coefficients,
                combination_coefficient_count,
                "combination coefficients",
            )?
        };
        let linear_coefficients = unsafe {
            required_input(
                linear_coefficients,
                linear_coefficient_count,
                "linear coefficients",
            )?
        };
        let outer_raw = unsafe { required_input(outer_digest, 32, "outer digest")? };
        let opening = prove_i64_linear_combination_linear_impl(
            values,
            vector_len,
            term_count,
            combination_coefficients,
            linear_coefficients,
            &outer_array(outer_raw)?,
        )?;
        unsafe {
            output_slice(output_commitment, COMMITMENT_BYTES, "commitment output")?
                .copy_from_slice(&opening.commitment);
            output_slice(output_evaluation, SCALAR_BYTES, "evaluation output")?
                .copy_from_slice(&opening.evaluation);
            output_slice(output_proof, proof_capacity, "proof output")?[..opening.proof.len()]
                .copy_from_slice(&opening.proof);
            *output_proof_len = opening.proof.len();
        }
        Ok(())
    })
}

/// Computes `sum(commitments[i] * coefficients[i])` for canonical Pallas
/// commitments and canonical little-endian Pallas scalars.
///
/// # Safety
///
/// `commitments` and `coefficients` must each point to `term_count * 32`
/// readable bytes. `output_commitment` must point to 32 writable bytes.
#[no_mangle]
pub unsafe extern "C" fn verathos_pcs_v2_combine_commitments(
    commitments: *const u8,
    coefficients: *const u8,
    term_count: usize,
    output_commitment: *mut u8,
) -> i32 {
    ffi_status(|| {
        if term_count == 0 || term_count > MAX_COMBINE_TERMS {
            return Err(PcsError::new(
                ERR_LENGTH,
                format!("commitment term count must be in 1..={MAX_COMBINE_TERMS}"),
            ));
        }
        let commitment_len = term_count
            .checked_mul(COMMITMENT_BYTES)
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "commitment input length overflow"))?;
        let coefficient_len = term_count
            .checked_mul(SCALAR_BYTES)
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "coefficient input length overflow"))?;
        // SAFETY: term count and byte lengths were capped above.
        let commitments = unsafe { required_input(commitments, commitment_len, "commitments")? };
        let coefficients =
            unsafe { required_input(coefficients, coefficient_len, "coefficients")? };
        let combined = combine_commitments(commitments, coefficients, term_count)?;
        // SAFETY: fixed-size output pointer is required by the ABI.
        let output =
            unsafe { output_slice(output_commitment, COMMITMENT_BYTES, "commitment output")? };
        output.copy_from_slice(&combined);
        Ok(())
    })
}

/// Validates and caches one static catalog commitment set. The returned ID is
/// a native domain-separated SHA-256 digest of the exact bytes and term count.
///
/// # Safety
///
/// `commitments` must point to `term_count * 32` readable bytes and
/// `output_catalog_id` must point to 32 writable bytes.
#[no_mangle]
pub unsafe extern "C" fn verathos_pcs_v2_register_catalog_commitments(
    commitments: *const u8,
    term_count: usize,
    output_catalog_id: *mut u8,
) -> i32 {
    ffi_status(|| {
        if term_count == 0 || term_count > MAX_CATALOG_COMBINE_TERMS {
            return Err(PcsError::new(
                ERR_LENGTH,
                format!("catalog commitment term count must be in 1..={MAX_CATALOG_COMBINE_TERMS}"),
            ));
        }
        let commitment_len = term_count
            .checked_mul(COMMITMENT_BYTES)
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "commitment input length overflow"))?;
        // SAFETY: term count and byte length are bounded above.
        let commitments = unsafe { required_input(commitments, commitment_len, "commitments")? };
        let catalog_id = register_catalog_commitments(commitments, term_count)?;
        // SAFETY: fixed-size output pointer is required by the ABI.
        unsafe { output_slice(output_catalog_id, 32, "catalog ID output")? }
            .copy_from_slice(&catalog_id);
        Ok(())
    })
}

/// Combines a previously registered catalog with unsigned 63-bit
/// little-endian coefficients.
///
/// # Safety
///
/// `catalog_id` must point to 32 readable bytes, `coefficients` must point to
/// `term_count * 8` readable bytes, and `output_commitment` must point to 32
/// writable bytes.
#[no_mangle]
pub unsafe extern "C" fn verathos_pcs_v2_combine_registered_catalog_u63(
    catalog_id: *const u8,
    coefficients: *const u8,
    term_count: usize,
    output_commitment: *mut u8,
) -> i32 {
    ffi_status(|| {
        if term_count == 0 || term_count > MAX_CATALOG_COMBINE_TERMS {
            return Err(PcsError::new(
                ERR_LENGTH,
                format!("catalog commitment term count must be in 1..={MAX_CATALOG_COMBINE_TERMS}"),
            ));
        }
        let coefficient_len = term_count
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "coefficient input length overflow"))?;
        // SAFETY: all fixed and derived lengths are bounded above.
        let catalog_id = unsafe { required_input(catalog_id, 32, "catalog ID")? };
        let coefficients =
            unsafe { required_input(coefficients, coefficient_len, "coefficients")? };
        let mut catalog_id_array = [0u8; 32];
        catalog_id_array.copy_from_slice(catalog_id);
        let combined =
            combine_registered_catalog_commitments_u63(catalog_id_array, coefficients, term_count)?;
        // SAFETY: fixed-size output pointer is required by the ABI.
        unsafe { output_slice(output_commitment, COMMITMENT_BYTES, "commitment output")? }
            .copy_from_slice(&combined);
        Ok(())
    })
}

/// Combines multiple unsigned-7-bit folds of one registered catalog.
/// Coefficients are fold-major and outputs are packed compressed points.
///
/// # Safety
///
/// `catalog_id` must point to 32 readable bytes, `coefficients` must point to
/// `term_count * fold_count` readable bytes, and `output_commitments` must
/// point to `fold_count * 32` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn verathos_pcs_v2_combine_registered_catalog_u7_batch(
    catalog_id: *const u8,
    coefficients: *const u8,
    term_count: usize,
    fold_count: usize,
    output_commitments: *mut u8,
) -> i32 {
    ffi_status(|| {
        if term_count == 0 || term_count > MAX_CATALOG_COMBINE_TERMS {
            return Err(PcsError::new(
                ERR_LENGTH,
                format!("catalog commitment term count must be in 1..={MAX_CATALOG_COMBINE_TERMS}"),
            ));
        }
        if fold_count == 0 || fold_count > MAX_CATALOG_FOLDS {
            return Err(PcsError::new(
                ERR_LENGTH,
                format!("catalog fold count must be in 1..={MAX_CATALOG_FOLDS}"),
            ));
        }
        let coefficient_len = term_count
            .checked_mul(fold_count)
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "coefficient input length overflow"))?;
        let output_len = fold_count
            .checked_mul(COMMITMENT_BYTES)
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "commitment output length overflow"))?;
        // SAFETY: all fixed and derived lengths are bounded above.
        let catalog_id = unsafe { required_input(catalog_id, 32, "catalog ID")? };
        let coefficients =
            unsafe { required_input(coefficients, coefficient_len, "coefficients")? };
        let mut catalog_id_array = [0u8; 32];
        catalog_id_array.copy_from_slice(catalog_id);
        let combined = combine_registered_catalog_commitments_u7_batch(
            catalog_id_array,
            coefficients,
            term_count,
            fold_count,
        )?;
        // SAFETY: derived output length is bounded above.
        unsafe { output_slice(output_commitments, output_len, "commitment output")? }
            .copy_from_slice(&combined);
        Ok(())
    })
}

/// Combines multiple unsigned-31-bit folds of one registered catalog.
/// Coefficients are fold-major little-endian u32 values and outputs are
/// packed compressed points.
///
/// # Safety
///
/// `catalog_id` must point to 32 readable bytes, `coefficients` must point to
/// `term_count * fold_count * 4` readable bytes, and `output_commitments`
/// must point to `fold_count * 32` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn verathos_pcs_v2_combine_registered_catalog_u31_batch(
    catalog_id: *const u8,
    coefficients: *const u8,
    term_count: usize,
    fold_count: usize,
    output_commitments: *mut u8,
) -> i32 {
    ffi_status(|| {
        if term_count == 0 || term_count > MAX_CATALOG_COMBINE_TERMS {
            return Err(PcsError::new(
                ERR_LENGTH,
                format!("catalog commitment term count must be in 1..={MAX_CATALOG_COMBINE_TERMS}"),
            ));
        }
        if fold_count == 0 || fold_count > MAX_CATALOG_FOLDS {
            return Err(PcsError::new(
                ERR_LENGTH,
                format!("catalog commitment fold count must be in 1..={MAX_CATALOG_FOLDS}"),
            ));
        }
        let coefficient_len = term_count
            .checked_mul(fold_count)
            .and_then(|count| count.checked_mul(std::mem::size_of::<u32>()))
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "catalog coefficient input overflow"))?;
        let output_len = fold_count
            .checked_mul(COMMITMENT_BYTES)
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "commitment output length overflow"))?;
        let catalog_id = unsafe { required_input(catalog_id, 32, "catalog ID")? };
        let coefficients =
            unsafe { required_input(coefficients, coefficient_len, "coefficients")? };
        let mut catalog_id_array = [0u8; 32];
        catalog_id_array.copy_from_slice(catalog_id);
        let combined = combine_registered_catalog_commitments_u31_batch(
            catalog_id_array,
            coefficients,
            term_count,
            fold_count,
        )?;
        unsafe { output_slice(output_commitments, output_len, "commitment output")? }
            .copy_from_slice(&combined);
        Ok(())
    })
}

/// Builds claim-major Pallas coefficient vectors from transcript-derived u31
/// folds.  For every claim `c` and term `j`, the output is
/// `sum_f beta[c,f] * coefficient[c,f,j]`.  Remaining padded output cells are
/// canonical zero scalars.
///
/// # Safety
///
/// `coefficients` must reference
/// `claim_count * fold_count * term_count * 4` readable bytes. `betas` must
/// reference `claim_count * fold_count * 32` readable bytes.
/// `output_coefficients` must reference `padded_len * 32` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn verathos_pcs_v3_fold_u31_linear_coefficients(
    coefficients: *const u8,
    betas: *const u8,
    claim_count: usize,
    term_count: usize,
    fold_count: usize,
    padded_len: usize,
    output_coefficients: *mut u8,
) -> i32 {
    ffi_status(|| {
        if claim_count == 0 || claim_count > MAX_LEAN_PROJECTION_MATERIAL_CLAIMS {
            return Err(PcsError::new(
                ERR_LENGTH,
                "linear-fold claim count is out of range",
            ));
        }
        if term_count == 0 || term_count > MAX_CATALOG_COMBINE_TERMS {
            return Err(PcsError::new(
                ERR_LENGTH,
                "linear-fold term count is out of range",
            ));
        }
        if fold_count == 0 || fold_count > MAX_CATALOG_FOLDS {
            return Err(PcsError::new(
                ERR_LENGTH,
                "linear-fold count is out of range",
            ));
        }
        let active_len = claim_count
            .checked_mul(term_count)
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "linear-fold active length overflow"))?;
        if padded_len < active_len
            || padded_len > MAX_VECTOR_LEN
            || !padded_len.is_power_of_two()
        {
            return Err(PcsError::new(
                ERR_LENGTH,
                "linear-fold padded length is out of range",
            ));
        }
        let coefficient_len = claim_count
            .checked_mul(fold_count)
            .and_then(|value| value.checked_mul(term_count))
            .and_then(|value| value.checked_mul(std::mem::size_of::<u32>()))
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "linear-fold coefficient input overflow"))?;
        let beta_count = claim_count
            .checked_mul(fold_count)
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "linear-fold beta count overflow"))?;
        let beta_len = beta_count
            .checked_mul(SCALAR_BYTES)
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "linear-fold beta input overflow"))?;
        let output_len = padded_len
            .checked_mul(SCALAR_BYTES)
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "linear-fold output overflow"))?;
        let coefficient_bytes = unsafe {
            required_input(coefficients, coefficient_len, "linear-fold coefficients")?
        };
        let beta_bytes = unsafe {
            required_input(betas, beta_len, "linear-fold betas")?
        };
        let beta_scalars = beta_bytes
            .chunks_exact(SCALAR_BYTES)
            .map(decode_scalar)
            .collect::<Result<Vec<_>, PcsError>>()?;
        let encoded = in_worker_pool(|| {
            (0..padded_len)
                .into_par_iter()
                .map(|index| {
                    if index >= active_len {
                        return Ok([0u8; SCALAR_BYTES]);
                    }
                    let claim_index = index / term_count;
                    let term_index = index % term_count;
                    let mut value = Fr::ZERO;
                    for fold_index in 0..fold_count {
                        let coefficient_index =
                            (claim_index * fold_count + fold_index)
                                * term_count
                                + term_index;
                        let offset = coefficient_index
                            * std::mem::size_of::<u32>();
                        let mut raw = [0u8; std::mem::size_of::<u32>()];
                        raw.copy_from_slice(
                            &coefficient_bytes[
                                offset..offset + std::mem::size_of::<u32>()
                            ],
                        );
                        let coefficient = u32::from_le_bytes(raw);
                        if coefficient >= 1 << 31 {
                            return Err(PcsError::new(
                                ERR_FORMAT,
                                "linear-fold coefficient is not unsigned 31-bit",
                            ));
                        }
                        value += beta_scalars[
                            claim_index * fold_count + fold_index
                        ] * Fr::from(coefficient as u64);
                    }
                    Ok(encode_scalar(&value))
                })
                .collect::<Result<Vec<_>, PcsError>>()
        })??;
        let output = unsafe {
            output_slice(
                output_coefficients,
                output_len,
                "linear-fold coefficient output",
            )?
        };
        for (destination, scalar) in output
            .chunks_exact_mut(SCALAR_BYTES)
            .zip(encoded.iter())
        {
            destination.copy_from_slice(scalar);
        }
        Ok(())
    })
}

/// Derives the exact transcript-bound u31 folds, signed-output targets, and
/// authenticated catalog commitments for equal-operation lean projection
/// claims. The output is claim-major: four Pallas targets followed by four
/// compressed commitments per claim.
///
/// # Safety
///
/// Fixed digests and catalog IDs must each reference 32 bytes. X and
/// surrogate buffers must match the explicit claim/dimension products.
/// `output_materials` must reference `claim_count * 256` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn verathos_pcs_v2_lean_projection_claim_materials(
    statement_digest: *const u8,
    validator_nonce: *const u8,
    catalog_id: *const u8,
    input_rows_i8: *const u8,
    input_rows_len: usize,
    surrogate_rows_i64_le: *const u8,
    surrogate_rows_len: usize,
    claim_count: usize,
    input_dim: usize,
    output_dim: usize,
    output_materials: *mut u8,
    output_capacity: usize,
) -> i32 {
    ffi_status(|| {
        if claim_count == 0 || claim_count > MAX_LEAN_PROJECTION_MATERIAL_CLAIMS {
            return Err(PcsError::new(
                ERR_LENGTH,
                "lean projection material claim count is out of range",
            ));
        }
        let expected_output = claim_count
            .checked_mul(2 * LEAN_PROJECTION_FOLD_COUNT * SCALAR_BYTES)
            .ok_or_else(|| {
                PcsError::new(ERR_LENGTH, "lean projection material output size overflow")
            })?;
        if output_capacity != expected_output {
            return Err(PcsError::new(
                ERR_LENGTH,
                "lean projection material output capacity is inconsistent",
            ));
        }
        let statement =
            unsafe { required_input(statement_digest, 32, "lean projection statement digest")? };
        let nonce =
            unsafe { required_input(validator_nonce, 32, "lean projection validator nonce")? };
        let catalog = unsafe { required_input(catalog_id, 32, "lean projection catalog ID")? };
        let x = unsafe { required_input(input_rows_i8, input_rows_len, "lean projection X rows")? };
        let surrogate = unsafe {
            required_input(
                surrogate_rows_i64_le,
                surrogate_rows_len,
                "lean projection surrogate rows",
            )?
        };
        let mut statement_array = [0u8; 32];
        statement_array.copy_from_slice(statement);
        let mut nonce_array = [0u8; 32];
        nonce_array.copy_from_slice(nonce);
        let mut catalog_array = [0u8; 32];
        catalog_array.copy_from_slice(catalog);
        let materials = lean_projection_claim_materials(
            &statement_array,
            &nonce_array,
            catalog_array,
            x,
            surrogate,
            claim_count,
            input_dim,
            output_dim,
        )?;
        if materials.len() != expected_output {
            return Err(PcsError::new(
                ERR_INTERNAL,
                "lean projection material output length is inconsistent",
            ));
        }
        unsafe {
            output_slice(
                output_materials,
                output_capacity,
                "lean projection material output",
            )?
        }
        .copy_from_slice(&materials);
        Ok(())
    })
}

/// Derives the exact transcript-bound signed-output targets for equal-operation
/// lean projection claims without constructing per-claim catalog commitments.
///
/// # Safety
///
/// Fixed digests must each reference 32 bytes. X and surrogate buffers must
/// match the explicit claim/dimension products. `output_targets` must reference
/// `claim_count * 4 * 32` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn verathos_pcs_v2_lean_projection_claim_targets(
    statement_digest: *const u8,
    validator_nonce: *const u8,
    input_rows_i8: *const u8,
    input_rows_len: usize,
    surrogate_rows_i64_le: *const u8,
    surrogate_rows_len: usize,
    claim_count: usize,
    input_dim: usize,
    output_dim: usize,
    output_targets: *mut u8,
    output_capacity: usize,
) -> i32 {
    ffi_status(|| {
        if claim_count == 0 || claim_count > MAX_LEAN_PROJECTION_MATERIAL_CLAIMS {
            return Err(PcsError::new(
                ERR_LENGTH,
                "lean projection target claim count is out of range",
            ));
        }
        let expected_output = claim_count
            .checked_mul(LEAN_PROJECTION_FOLD_COUNT * SCALAR_BYTES)
            .ok_or_else(|| {
                PcsError::new(ERR_LENGTH, "lean projection target output size overflow")
            })?;
        if output_capacity != expected_output {
            return Err(PcsError::new(
                ERR_LENGTH,
                "lean projection target output capacity is inconsistent",
            ));
        }
        let statement =
            unsafe { required_input(statement_digest, 32, "lean projection statement digest")? };
        let nonce =
            unsafe { required_input(validator_nonce, 32, "lean projection validator nonce")? };
        let x = unsafe { required_input(input_rows_i8, input_rows_len, "lean projection X rows")? };
        let surrogate = unsafe {
            required_input(
                surrogate_rows_i64_le,
                surrogate_rows_len,
                "lean projection surrogate rows",
            )?
        };
        let mut statement_array = [0u8; 32];
        statement_array.copy_from_slice(statement);
        let mut nonce_array = [0u8; 32];
        nonce_array.copy_from_slice(nonce);
        let targets = lean_projection_claim_targets(
            &statement_array,
            &nonce_array,
            x,
            surrogate,
            claim_count,
            input_dim,
            output_dim,
        )?;
        if targets.len() != expected_output {
            return Err(PcsError::new(
                ERR_INTERNAL,
                "lean projection target output length is inconsistent",
            ));
        }
        unsafe {
            output_slice(
                output_targets,
                output_capacity,
                "lean projection target output",
            )?
        }
        .copy_from_slice(&targets);
        Ok(())
    })
}

/// Computes one authenticated catalog commitment for the beta-weighted
/// aggregate of all four folded-weight vectors across equal-operation claims.
///
/// # Safety
///
/// Fixed digests and catalog IDs must each reference 32 bytes. X, surrogate,
/// and beta buffers must match the explicit claim/dimension products.
/// `output_commitment` must reference exactly 32 writable bytes.
#[no_mangle]
pub unsafe extern "C" fn verathos_pcs_v2_lean_projection_aggregate_commitment(
    statement_digest: *const u8,
    validator_nonce: *const u8,
    catalog_id: *const u8,
    input_rows_i8: *const u8,
    input_rows_len: usize,
    surrogate_rows_i64_le: *const u8,
    surrogate_rows_len: usize,
    beta_bytes: *const u8,
    beta_bytes_len: usize,
    claim_count: usize,
    input_dim: usize,
    output_dim: usize,
    output_commitment: *mut u8,
    output_capacity: usize,
) -> i32 {
    ffi_status(|| {
        if output_capacity != COMMITMENT_BYTES {
            return Err(PcsError::new(
                ERR_LENGTH,
                "lean projection aggregate commitment capacity is inconsistent",
            ));
        }
        let statement =
            unsafe { required_input(statement_digest, 32, "lean projection statement digest")? };
        let nonce =
            unsafe { required_input(validator_nonce, 32, "lean projection validator nonce")? };
        let catalog = unsafe { required_input(catalog_id, 32, "lean projection catalog ID")? };
        let x = unsafe { required_input(input_rows_i8, input_rows_len, "lean projection X rows")? };
        let surrogate = unsafe {
            required_input(
                surrogate_rows_i64_le,
                surrogate_rows_len,
                "lean projection surrogate rows",
            )?
        };
        let betas = unsafe {
            required_input(
                beta_bytes,
                beta_bytes_len,
                "lean projection aggregate betas",
            )?
        };
        let mut statement_array = [0u8; 32];
        statement_array.copy_from_slice(statement);
        let mut nonce_array = [0u8; 32];
        nonce_array.copy_from_slice(nonce);
        let mut catalog_array = [0u8; 32];
        catalog_array.copy_from_slice(catalog);
        let commitment = lean_projection_aggregate_commitment(
            &statement_array,
            &nonce_array,
            catalog_array,
            x,
            surrogate,
            betas,
            claim_count,
            input_dim,
            output_dim,
        )?;
        unsafe {
            output_slice(
                output_commitment,
                output_capacity,
                "lean projection aggregate commitment",
            )?
        }
        .copy_from_slice(&commitment);
        Ok(())
    })
}

/// Validates a packed batch of canonical compressed Pallas commitments.
///
/// # Safety
///
/// `commitments` must point to `commitment_count * 32` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn verathos_pcs_v2_validate_commitments(
    commitments: *const u8,
    commitment_count: usize,
) -> i32 {
    ffi_status(|| {
        if commitment_count == 0 || commitment_count > MAX_VALIDATE_COMMITMENTS {
            return Err(PcsError::new(
                ERR_LENGTH,
                format!("commitment validation count must be in 1..={MAX_VALIDATE_COMMITMENTS}"),
            ));
        }
        let commitment_len = commitment_count
            .checked_mul(COMMITMENT_BYTES)
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "commitment input length overflow"))?;
        // SAFETY: the count and derived byte length were bounded above.
        let commitments = unsafe { required_input(commitments, commitment_len, "commitments")? };
        validate_commitment_encodings(commitments, commitment_count)
    })
}

/// Verifies one statement-bound multilinear opening.
///
/// # Safety
///
/// Every non-null input must reference a readable buffer of its fixed or
/// declared length. `point_bytes` may be null only when `point_count` is zero.
#[no_mangle]
pub unsafe extern "C" fn verathos_pcs_v2_verify(
    commitment: *const u8,
    vector_len: usize,
    encoding: u32,
    point_bytes: *const u8,
    point_count: usize,
    outer_digest: *const u8,
    evaluation: *const u8,
    proof: *const u8,
    proof_len: usize,
) -> i32 {
    ffi_verify(|| {
        let (_, log_n) = shape(vector_len)?;
        if point_count != log_n {
            return Err(PcsError::new(ERR_LENGTH, "wrong MLE point length"));
        }
        if proof_len != proof_wire_len(log_n)? {
            return Err(PcsError::new(ERR_FORMAT, "wrong IPA proof length"));
        }
        let encoding = Encoding::parse(encoding)?;
        // SAFETY: all lengths are fixed or capped before pointer conversion.
        let commitment = unsafe { required_input(commitment, COMMITMENT_BYTES, "commitment")? };
        let point =
            unsafe { optional_empty_input(point_bytes, point_count * SCALAR_BYTES, "point")? };
        let outer = unsafe { required_input(outer_digest, 32, "outer digest")? };
        let evaluation = unsafe { required_input(evaluation, SCALAR_BYTES, "evaluation")? };
        let proof = unsafe { required_input(proof, proof_len, "proof")? };
        verify_opening(
            commitment,
            vector_len,
            encoding,
            point,
            &outer_array(outer)?,
            evaluation,
            proof,
        )
    })
}

/// Verifies one arbitrary-public-vector IPA opening.
///
/// The coefficient vector contains exactly the padded vector length in
/// canonical little-endian Pallas scalars. It is transcript-bound and is not
/// carried inside the proof.
#[no_mangle]
pub unsafe extern "C" fn verathos_pcs_v3_verify_linear(
    commitment: *const u8,
    vector_len: usize,
    encoding: u32,
    coefficient_bytes: *const u8,
    coefficient_count: usize,
    outer_digest: *const u8,
    evaluation: *const u8,
    proof: *const u8,
    proof_len: usize,
) -> i32 {
    ffi_verify(|| {
        let (padded_len, log_n) = shape(vector_len)?;
        if coefficient_count != padded_len {
            return Err(PcsError::new(
                ERR_LENGTH,
                "linear opening coefficient count is wrong",
            ));
        }
        if proof_len != proof_wire_len(log_n)? {
            return Err(PcsError::new(ERR_FORMAT, "wrong IPA proof length"));
        }
        let encoding = Encoding::parse(encoding)?;
        let coefficient_len = coefficient_count
            .checked_mul(SCALAR_BYTES)
            .ok_or_else(|| PcsError::new(ERR_LENGTH, "linear coefficient length overflow"))?;
        let commitment =
            unsafe { required_input(commitment, COMMITMENT_BYTES, "commitment")? };
        let coefficients = unsafe {
            required_input(
                coefficient_bytes,
                coefficient_len,
                "linear coefficients",
            )?
        };
        let outer = unsafe { required_input(outer_digest, 32, "outer digest")? };
        let evaluation =
            unsafe { required_input(evaluation, SCALAR_BYTES, "evaluation")? };
        let proof = unsafe { required_input(proof, proof_len, "proof")? };
        verify_linear_opening(
            commitment,
            vector_len,
            encoding,
            coefficients,
            &outer_array(outer)?,
            evaluation,
            proof,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn scalar(value: u64) -> Fr {
        Fr::from(value)
    }

    fn regular_i8_commitment(values: &[i8]) -> [u8; COMMITMENT_BYTES] {
        let (poly, padded_len) = poly_from_i8(values).unwrap();
        commit_poly(&poly, padded_len).unwrap()
    }

    #[test]
    fn cpu_list_parser_counts_ranges_and_rejects_malformed_values() {
        assert_eq!(parse_cpu_list_count("0-3,8,10-11\n"), Some(7));
        assert_eq!(parse_cpu_list_count("5"), Some(1));
        assert_eq!(parse_cpu_list_count(""), None);
        assert_eq!(parse_cpu_list_count("3-1"), None);
        assert_eq!(parse_cpu_list_count("0-2-4"), None);
    }

    #[test]
    fn shifted_small_i8_commitment_matches_regular_scalar_commitment() {
        for length in [1usize, 2, 3, 5, 8, 9, 17, 31, 5_120] {
            let mixed: Vec<i8> = (0..length)
                .map(|index| ((index * 73 + 19) as u8) as i8)
                .collect();
            for values in [
                mixed,
                vec![i8::MIN; length],
                vec![i8::MAX; length],
                vec![0i8; length],
            ] {
                let expected = regular_i8_commitment(&values);
                assert_eq!(commit_i8_values(&values).unwrap(), expected);
                // Exercise the cached correction path as well as its creation.
                assert_eq!(commit_i8_values(&values).unwrap(), expected);
            }
        }
    }

    #[test]
    fn batched_i8_commitments_match_serial_and_regular_commitments() {
        let columns = [
            vec![i8::MIN, -17, -1, 0, i8::MAX],
            vec![5, 4, 3, 2, 1],
            vec![-99, 88, -77, 66, -55],
            vec![0, 0, 0, 0, 0],
        ];
        let packed: Vec<i8> = columns.iter().flatten().copied().collect();
        let mut output = vec![0u8; columns.len() * COMMITMENT_BYTES];
        commit_i8_batch_values(&packed, columns[0].len(), columns.len(), &mut output).unwrap();

        for (column, encoded) in columns.iter().zip(output.chunks_exact(COMMITMENT_BYTES)) {
            assert_eq!(encoded, commit_i8_values(column).unwrap());
            assert_eq!(encoded, regular_i8_commitment(column));
        }
    }

    #[test]
    fn matrix_i8_evaluation_matches_flat_reference() {
        let rows = 4usize;
        let columns = 8usize;
        let values = (0..rows * columns)
            .map(|index| ((index * 29 + 7) as u8) as i8)
            .collect::<Vec<_>>();
        let point = (0..rows.trailing_zeros() as usize + columns.trailing_zeros() as usize)
            .map(|index| scalar((index * 13 + 3) as u64))
            .collect::<Vec<_>>();
        let (poly, padded_len) = poly_from_i8(&values).unwrap();
        assert_eq!(padded_len, rows * columns);
        assert_eq!(
            evaluate_i8_matrix_values(&values, rows, columns, &point).unwrap(),
            evaluate_poly(&poly, &point),
        );
    }

    #[test]
    fn batched_i8_ffi_rejects_nonexact_lengths() {
        let values = [1i8, 2, 3, 4, 5, 6];
        let mut output = [0u8; 2 * COMMITMENT_BYTES];
        // SAFETY: test buffers satisfy their declared allocations. The
        // deliberately wrong logical lengths are rejected before conversion.
        unsafe {
            assert_eq!(
                verathos_pcs_v2_commit_i8_batch(
                    values.as_ptr(),
                    values.len() - 1,
                    3,
                    2,
                    output.as_mut_ptr(),
                    output.len(),
                ),
                ERR_LENGTH
            );
            assert_eq!(
                verathos_pcs_v2_commit_i8_batch(
                    values.as_ptr(),
                    values.len(),
                    3,
                    2,
                    output.as_mut_ptr(),
                    output.len() - 1,
                ),
                ERR_BUFFER
            );
        }
    }

    #[test]
    fn honest_non_power_of_two_i8_roundtrip() {
        let values = [-3i8, 5, 7, -11, 13];
        let (poly, padded_len) = poly_from_i8(&values).unwrap();
        assert_eq!(padded_len, 8);
        let point = vec![scalar(2), scalar(3), scalar(5)];
        let outer = [0x42u8; 32];
        let opening = prove_poly(
            &poly,
            values.len(),
            padded_len,
            Encoding::SignedI8,
            &point,
            &outer,
        )
        .unwrap();
        assert_eq!(opening.commitment.len(), COMMITMENT_BYTES);
        assert_eq!(opening.proof.len(), proof_wire_len(3).unwrap());
        let point_bytes: Vec<u8> = point.iter().flat_map(encode_scalar).collect();
        assert!(verify_opening(
            &opening.commitment,
            values.len(),
            Encoding::SignedI8,
            &point_bytes,
            &outer,
            &opening.evaluation,
            &opening.proof,
        )
        .unwrap());
    }

    #[test]
    fn arbitrary_linear_i8_roundtrip_and_binding() {
        let values = [-3i8, 5, 7, -11, 13];
        let (poly, padded_len) = poly_from_i8(&values).unwrap();
        let coefficients = (0..padded_len)
            .map(|index| scalar((index * 17 + 3) as u64))
            .collect::<Vec<_>>();
        let coefficient_bytes = coefficients
            .iter()
            .flat_map(encode_scalar)
            .collect::<Vec<_>>();
        let outer = [0x81u8; 32];
        let opening = prove_linear_poly(
            &poly,
            values.len(),
            padded_len,
            Encoding::SignedI8,
            &coefficient_bytes,
            &outer,
        )
        .unwrap();
        assert!(verify_linear_opening(
            &opening.commitment,
            values.len(),
            Encoding::SignedI8,
            &coefficient_bytes,
            &outer,
            &opening.evaluation,
            &opening.proof,
        )
        .unwrap());

        let mut wrong_coefficients = coefficient_bytes.clone();
        wrong_coefficients[0] ^= 1;
        assert!(!verify_linear_opening(
            &opening.commitment,
            values.len(),
            Encoding::SignedI8,
            &wrong_coefficients,
            &outer,
            &opening.evaluation,
            &opening.proof,
        )
        .unwrap());
        let wrong_outer = [0x82u8; 32];
        assert!(!verify_linear_opening(
            &opening.commitment,
            values.len(),
            Encoding::SignedI8,
            &coefficient_bytes,
            &wrong_outer,
            &opening.evaluation,
            &opening.proof,
        )
        .unwrap());
    }

    #[test]
    fn signed_i64_boolean_evaluations_follow_msb_first_order() {
        let values = [-3i64, 7, 11];
        let (poly, _) = poly_from_i64(&values).unwrap();
        let at_01 = evaluate_poly(&poly, &[Fr::ZERO, Fr::ONE]);
        let at_10 = evaluate_poly(&poly, &[Fr::ONE, Fr::ZERO]);
        assert_eq!(at_01, scalar(7));
        assert_eq!(at_10, scalar(11));
        assert_eq!(poly[0], -scalar(3));
        assert_eq!(poly[3], Fr::ZERO);
    }

    #[test]
    fn commitments_and_proofs_are_deterministic() {
        let values = [1i8, -2, 3, -4, 5, -6, 7, -8];
        let (poly, padded_len) = poly_from_i8(&values).unwrap();
        let point = vec![scalar(9), scalar(10), scalar(11)];
        let outer = [0x19u8; 32];
        let first = prove_poly(
            &poly,
            values.len(),
            padded_len,
            Encoding::SignedI8,
            &point,
            &outer,
        )
        .unwrap();
        let second = prove_poly(
            &poly,
            values.len(),
            padded_len,
            Encoding::SignedI8,
            &point,
            &outer,
        )
        .unwrap();
        assert_eq!(first.commitment, second.commitment);
        assert_eq!(first.evaluation, second.evaluation);
        assert_eq!(first.proof, second.proof);
    }

    #[test]
    fn optimized_ipa_is_byte_identical_to_nova_and_cross_verifies() {
        for padded_len in [8usize, 64, 8_192] {
            let setup = setup_for(padded_len).unwrap();
            let poly: Vec<Fr> = (0..padded_len)
                .map(|index| scalar(((index * 73 + 19) % 65_521) as u64))
                .collect();
            let point: Vec<Fr> = (0..padded_len.trailing_zeros() as usize)
                .map(|index| scalar((index + 3) as u64))
                .collect();
            let outer = [padded_len.trailing_zeros() as u8; 32];
            let evaluation = evaluate_poly(&poly, &point);

            let (commitment, nova_bytes, optimized_bytes) = {
                let _operation = operation_guard().unwrap();
                let commitment =
                    in_worker_pool(|| CE::commit(&setup.ck, &poly, &Fr::ZERO)).unwrap();

                let mut nova_transcript = opening_transcript(
                    &outer,
                    padded_len,
                    padded_len,
                    Encoding::PallasScalar,
                    &point,
                );
                let nova_proof = in_worker_pool(|| {
                    EE::prove(
                        &setup.ck,
                        &setup.pk,
                        &mut nova_transcript,
                        &commitment,
                        &poly,
                        &point,
                        &evaluation,
                    )
                })
                .unwrap()
                .unwrap();
                let nova_bytes =
                    bincode::serde::encode_to_vec(&nova_proof, config::standard()).unwrap();

                let mut optimized_transcript = opening_transcript(
                    &outer,
                    padded_len,
                    padded_len,
                    Encoding::PallasScalar,
                    &point,
                );
                let optimized_proof = in_worker_pool(|| {
                    prove_ipa_optimized(
                        &setup.ck,
                        setup.ipa_evaluation_base,
                        &commitment,
                        &poly,
                        &point,
                        &evaluation,
                        &mut optimized_transcript,
                    )
                })
                .unwrap()
                .unwrap();
                let optimized_bytes =
                    bincode::serde::encode_to_vec(&optimized_proof, config::standard()).unwrap();
                (commitment, nova_bytes, optimized_bytes)
            };

            assert_eq!(optimized_bytes, nova_bytes, "padded_len={padded_len}");
            let decoded =
                decode_proof(&optimized_bytes, padded_len.trailing_zeros() as usize).unwrap();
            let mut verify_transcript = opening_transcript(
                &outer,
                padded_len,
                padded_len,
                Encoding::PallasScalar,
                &point,
            );
            let _operation = operation_guard().unwrap();
            in_worker_pool(|| {
                EE::verify(
                    &setup.vk,
                    &mut verify_transcript,
                    &commitment,
                    &point,
                    &evaluation,
                    &decoded,
                )
            })
            .unwrap()
            .unwrap();
        }
    }

    #[test]
    fn parallel_i8_linear_combination_matches_direct_field_fold() {
        let vectors = [
            [i8::MIN, -17, -1, 0, 1, 7, 42, i8::MAX],
            [11, 9, 7, 5, 3, 1, -1, -3],
            [-99, 88, -77, 66, -55, 44, -33, 22],
        ];
        let coefficients = [scalar(13), -scalar(17), scalar(23)];
        let values = vectors.iter().flatten().copied().collect::<Vec<_>>();
        let coefficient_bytes = coefficients
            .iter()
            .flat_map(encode_scalar)
            .collect::<Vec<_>>();
        let point = vec![scalar(29), scalar(31), scalar(37)];
        let outer = [0xa5; 32];

        let opening = prove_i8_linear_combination_impl(
            &values,
            vectors[0].len(),
            vectors.len(),
            &coefficient_bytes,
            &point,
            &outer,
        )
        .unwrap();
        let direct = (0..vectors[0].len())
            .map(|index| {
                vectors
                    .iter()
                    .zip(coefficients)
                    .fold(Fr::ZERO, |sum, (vector, coefficient)| {
                        sum + signed_i64_to_scalar(vector[index] as i64) * coefficient
                    })
            })
            .collect::<Vec<_>>();
        let expected = prove_poly(
            &direct,
            direct.len(),
            direct.len(),
            Encoding::PallasScalar,
            &point,
            &outer,
        )
        .unwrap();

        assert_eq!(opening.commitment, expected.commitment);
        assert_eq!(opening.evaluation, expected.evaluation);
        assert_eq!(opening.proof, expected.proof);
    }

    #[test]
    fn statement_mutations_fail() {
        let values = [1i8, 2, 3, 4];
        let (poly, padded_len) = poly_from_i8(&values).unwrap();
        let point = vec![scalar(12), scalar(13)];
        let point_bytes: Vec<u8> = point.iter().flat_map(encode_scalar).collect();
        let outer = [0x77u8; 32];
        let opening = prove_poly(
            &poly,
            values.len(),
            padded_len,
            Encoding::SignedI8,
            &point,
            &outer,
        )
        .unwrap();

        let mut wrong_outer = outer;
        wrong_outer[0] ^= 1;
        assert!(!verify_opening(
            &opening.commitment,
            values.len(),
            Encoding::SignedI8,
            &point_bytes,
            &wrong_outer,
            &opening.evaluation,
            &opening.proof,
        )
        .unwrap());

        let mut wrong_eval = opening.evaluation;
        wrong_eval[0] ^= 1;
        assert!(!verify_opening(
            &opening.commitment,
            values.len(),
            Encoding::SignedI8,
            &point_bytes,
            &outer,
            &wrong_eval,
            &opening.proof,
        )
        .unwrap());

        assert!(!verify_opening(
            &opening.commitment,
            values.len(),
            Encoding::SignedI64,
            &point_bytes,
            &outer,
            &opening.evaluation,
            &opening.proof,
        )
        .unwrap());
    }

    #[test]
    fn malformed_proofs_are_rejected_without_panics() {
        let values = [4i8, 3, 2, 1];
        let (poly, padded_len) = poly_from_i8(&values).unwrap();
        let point = vec![scalar(2), scalar(8)];
        let point_bytes: Vec<u8> = point.iter().flat_map(encode_scalar).collect();
        let outer = [0x55u8; 32];
        let opening = prove_poly(
            &poly,
            values.len(),
            padded_len,
            Encoding::SignedI8,
            &point,
            &outer,
        )
        .unwrap();
        for length in 0..opening.proof.len() {
            assert!(verify_opening(
                &opening.commitment,
                values.len(),
                Encoding::SignedI8,
                &point_bytes,
                &outer,
                &opening.evaluation,
                &opening.proof[..length],
            )
            .is_err());
        }
        let mut tampered = opening.proof.clone();
        tampered[1] ^= 1;
        match verify_opening(
            &opening.commitment,
            values.len(),
            Encoding::SignedI8,
            &point_bytes,
            &outer,
            &opening.evaluation,
            &tampered,
        ) {
            Ok(valid) => assert!(!valid),
            Err(error) => assert_eq!(error.code, ERR_FORMAT),
        }
    }

    #[test]
    fn cached_verifier_is_thread_safe() {
        let values = [1i8, -1, 2, -2, 3, -3, 4, -4];
        let (poly, padded_len) = poly_from_i8(&values).unwrap();
        let point = vec![scalar(2), scalar(3), scalar(4)];
        let point_bytes: Vec<u8> = point.iter().flat_map(encode_scalar).collect();
        let outer = [0xabu8; 32];
        let opening = Arc::new(
            prove_poly(
                &poly,
                values.len(),
                padded_len,
                Encoding::SignedI8,
                &point,
                &outer,
            )
            .unwrap(),
        );
        let mut handles = Vec::new();
        for _ in 0..4 {
            let opening = Arc::clone(&opening);
            let point_bytes = point_bytes.clone();
            handles.push(thread::spawn(move || {
                verify_opening(
                    &opening.commitment,
                    values.len(),
                    Encoding::SignedI8,
                    &point_bytes,
                    &outer,
                    &opening.evaluation,
                    &opening.proof,
                )
                .unwrap()
            }));
        }
        for handle in handles {
            assert!(handle.join().unwrap());
        }
    }

    #[test]
    fn canonical_field_vectors_roundtrip() {
        let values = [scalar(1), -scalar(2), scalar(3), scalar(17), scalar(99)];
        let value_bytes: Vec<u8> = values.iter().flat_map(encode_scalar).collect();
        let (poly, padded_len) = poly_from_scalar_bytes(&value_bytes, values.len()).unwrap();
        assert_eq!(padded_len, 8);
        assert_eq!(&poly[..values.len()], &values);
        assert_eq!(&poly[values.len()..], &[Fr::ZERO; 3]);

        let point = vec![scalar(5), scalar(7), scalar(11)];
        let point_bytes: Vec<u8> = point.iter().flat_map(encode_scalar).collect();
        let outer = [0x5au8; 32];
        let opening = prove_poly(
            &poly,
            values.len(),
            padded_len,
            Encoding::PallasScalar,
            &point,
            &outer,
        )
        .unwrap();
        assert!(verify_opening(
            &opening.commitment,
            values.len(),
            Encoding::PallasScalar,
            &point_bytes,
            &outer,
            &opening.evaluation,
            &opening.proof,
        )
        .unwrap());

        let mut noncanonical = value_bytes;
        noncanonical[..SCALAR_BYTES].fill(0xff);
        assert_eq!(
            poly_from_scalar_bytes(&noncanonical, values.len())
                .unwrap_err()
                .code,
            ERR_FORMAT
        );
    }

    #[test]
    fn commitment_linear_combination_matches_folded_field_vector() {
        let rows = [
            vec![scalar(1), scalar(3), scalar(5), scalar(7)],
            vec![scalar(2), scalar(4), scalar(6), scalar(8)],
            vec![scalar(11), scalar(13), scalar(17), scalar(19)],
        ];
        let coefficients = [scalar(23), scalar(29), scalar(31)];
        let commitments: Vec<[u8; COMMITMENT_BYTES]> = rows
            .iter()
            .map(|row| commit_poly(row, row.len()).unwrap())
            .collect();
        let commitment_bytes: Vec<u8> = commitments.iter().flatten().copied().collect();
        let coefficient_bytes: Vec<u8> = coefficients.iter().flat_map(encode_scalar).collect();
        let combined =
            combine_commitments(&commitment_bytes, &coefficient_bytes, rows.len()).unwrap();

        let folded: Vec<Fr> = (0..rows[0].len())
            .map(|column| {
                rows.iter()
                    .zip(coefficients)
                    .fold(Fr::ZERO, |sum, (row, coefficient)| {
                        sum + row[column] * coefficient
                    })
            })
            .collect();
        let direct = commit_poly(&folded, folded.len()).unwrap();
        assert_eq!(combined, direct);

        let point = vec![scalar(37), scalar(41)];
        let point_bytes: Vec<u8> = point.iter().flat_map(encode_scalar).collect();
        let outer = [0x91u8; 32];
        let opening = prove_poly(
            &folded,
            folded.len(),
            folded.len(),
            Encoding::PallasScalar,
            &point,
            &outer,
        )
        .unwrap();
        assert_eq!(opening.commitment, combined);
        assert!(verify_opening(
            &combined,
            folded.len(),
            Encoding::PallasScalar,
            &point_bytes,
            &outer,
            &opening.evaluation,
            &opening.proof,
        )
        .unwrap());

        let mut wrong_coefficients = coefficients;
        wrong_coefficients[0] += Fr::ONE;
        let wrong_coefficient_bytes: Vec<u8> =
            wrong_coefficients.iter().flat_map(encode_scalar).collect();
        let wrong_combined =
            combine_commitments(&commitment_bytes, &wrong_coefficient_bytes, rows.len()).unwrap();
        assert!(!verify_opening(
            &wrong_combined,
            folded.len(),
            Encoding::PallasScalar,
            &point_bytes,
            &outer,
            &opening.evaluation,
            &opening.proof,
        )
        .unwrap());
    }

    #[test]
    fn commitment_combination_is_thread_safe() {
        let rows = [
            vec![scalar(1), scalar(2), scalar(3), scalar(4)],
            vec![scalar(5), scalar(6), scalar(7), scalar(8)],
        ];
        let commitments: Vec<u8> = rows
            .iter()
            .flat_map(|row| commit_poly(row, row.len()).unwrap().into_iter())
            .collect();
        let coefficients: Vec<u8> = [scalar(9), scalar(10)]
            .iter()
            .flat_map(encode_scalar)
            .collect();
        let expected = combine_commitments(&commitments, &coefficients, rows.len()).unwrap();
        let term_count = rows.len();
        let commitments = Arc::new(commitments);
        let coefficients = Arc::new(coefficients);
        let mut handles = Vec::new();
        for _ in 0..4 {
            let commitments = Arc::clone(&commitments);
            let coefficients = Arc::clone(&coefficients);
            handles.push(thread::spawn(move || {
                for _ in 0..32 {
                    assert_eq!(
                        combine_commitments(&commitments, &coefficients, term_count).unwrap(),
                        expected
                    );
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
    }
}
