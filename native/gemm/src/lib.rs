//! GEMM sumcheck prover over the Pallas scalar field plus int8 matrix MLE
//! evaluation, called from Python via ctypes.
//!
//! The C ABI is byte-oriented: scalars are canonical 32-byte little-endian
//! Pallas scalar encodings, status codes are `0` on success and negative on
//! failure, and the last error message is kept per thread.

#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(not(target_endian = "little"))]
compile_error!("verathos-gemm requires a little-endian target");

use halo2curves::{
    ff::{Field, FromUniformBytes, PrimeField},
    pasta::Fq as Fr,
};
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use sha2::{Digest, Sha256};
use std::{
    cell::RefCell,
    env, fs,
    panic::{catch_unwind, AssertUnwindSafe},
    ptr,
    sync::OnceLock,
};

/// ABI version checked by the ctypes binding.
pub const ABI_VERSION: u32 = 1;
pub const SCALAR_BYTES: usize = 32;
/// Largest accepted GEMM inner dimension.
pub const MAX_INNER: usize = 1 << 17;
pub const MAX_GEMM_BATCH_BLOCKS: usize = 64;
/// Largest accepted GEMM witness (X, W, Y and valid-inner buffers).
pub const MAX_GEMM_BATCH_BYTES: usize = 16 << 20;
/// Largest accepted int8 matrix for MLE evaluation.
pub const MAX_MATRIX_BYTES: usize = 16 << 20;
const DEFAULT_WORKER_THREADS: usize = 8;
const MAX_WORKER_THREADS: usize = 64;

const GEMM_BATCH_TRANSCRIPT_DOMAIN: &[u8] = b"VERATHOS/GEMM/BATCH_SUMCHECK/PALLAS_SCALAR/SHA256";
const GEMM_BATCH_PROOF_MAGIC: &[u8; 4] = b"VGB1";
const GEMM_BATCH_PROOF_VERSION: u16 = 1;
/// `3 * modulus` as little-endian u64 limbs: the largest multiple of the
/// Pallas scalar modulus below 2^256, used for unbiased challenge sampling.
const CHALLENGE_LIMIT: [u64; 4] = [
    0xa4d4_c163_0000_0003,
    0x66d3_caf4_1cbd_fa98,
    0x0000_0000_0000_0000,
    0xc000_0000_0000_0000,
];

const ERR_NULL: i32 = -1;
const ERR_LENGTH: i32 = -2;
const ERR_FORMAT: i32 = -4;
const ERR_CRYPTO: i32 = -5;
const ERR_BUFFER: i32 = -6;
const ERR_INTERNAL: i32 = -99;

#[derive(Debug)]
struct Error {
    code: i32,
    message: String,
}

impl Error {
    fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

static WORKER_POOL: OnceLock<Result<ThreadPool, String>> = OnceLock::new();

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_last_error(message: impl Into<String>) {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = message.into());
}

fn clear_last_error() {
    LAST_ERROR.with(|slot| slot.borrow_mut().clear());
}

/// Worker count: `VERATHOS_GEMM_THREADS` if set to 1..=64, otherwise the
/// available (cgroup-limited) CPUs capped at 8.
fn configured_worker_threads() -> usize {
    env::var("VERATHOS_GEMM_THREADS")
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

fn worker_pool() -> Result<&'static ThreadPool, Error> {
    WORKER_POOL
        .get_or_init(|| {
            ThreadPoolBuilder::new()
                .num_threads(configured_worker_threads())
                .thread_name(|index| format!("verathos-gemm-{index}"))
                .build()
                .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(|message| {
            Error::new(
                ERR_INTERNAL,
                format!("worker pool initialization failed: {message}"),
            )
        })
}

fn in_worker_pool<F, T>(operation: F) -> Result<T, Error>
where
    F: FnOnce() -> T + Send,
    T: Send,
{
    Ok(worker_pool()?.install(operation))
}

fn signed_i64_to_scalar(value: i64) -> Fr {
    if value >= 0 {
        Fr::from(value as u64)
    } else {
        -Fr::from(value.unsigned_abs())
    }
}

fn decode_scalar(bytes: &[u8]) -> Result<Fr, Error> {
    if bytes.len() != SCALAR_BYTES {
        return Err(Error::new(ERR_FORMAT, "scalar must be exactly 32 bytes"));
    }
    let mut repr = <Fr as PrimeField>::Repr::default();
    repr.as_mut().copy_from_slice(bytes);
    let Some(scalar) = Option::<Fr>::from(Fr::from_repr(repr)) else {
        return Err(Error::new(ERR_FORMAT, "non-canonical Pallas scalar"));
    };
    if scalar.to_repr().as_ref() != bytes {
        return Err(Error::new(ERR_FORMAT, "non-canonical Pallas scalar"));
    }
    Ok(scalar)
}

fn encode_scalar(scalar: &Fr) -> [u8; SCALAR_BYTES] {
    let repr = scalar.to_repr();
    let mut out = [0u8; SCALAR_BYTES];
    out.copy_from_slice(repr.as_ref());
    out
}

fn decode_point(point_bytes: &[u8], expected_len: usize) -> Result<Vec<Fr>, Error> {
    let expected = expected_len
        .checked_mul(SCALAR_BYTES)
        .ok_or_else(|| Error::new(ERR_LENGTH, "point length overflow"))?;
    if point_bytes.len() != expected {
        return Err(Error::new(
            ERR_LENGTH,
            format!("MLE point must contain exactly {expected_len} scalars"),
        ));
    }
    point_bytes
        .chunks_exact(SCALAR_BYTES)
        .map(decode_scalar)
        .collect()
}

/// Evaluations of `eq(x, r)` over the boolean hypercube, with `r[0]` bound
/// to the most significant index bit.
fn eq_evals(r: &[Fr]) -> Vec<Fr> {
    let mut evals = vec![Fr::ZERO; 1usize << r.len()];
    evals[0] = Fr::ONE;
    let mut size = 1;
    for coordinate in r.iter().rev() {
        let (low, high) = evals.split_at_mut(size);
        for (x, y) in low.iter_mut().zip(high[..size].iter_mut()) {
            *y = *x * coordinate;
            *x -= *y;
        }
        size *= 2;
    }
    evals
}

/// MLE of a row-major int8 matrix at `point` (row coordinates first,
/// most significant bit first).
fn evaluate_i8_matrix_values(
    values: &[i8],
    rows: usize,
    columns: usize,
    point: &[Fr],
) -> Result<Fr, Error> {
    let row_bits = exact_power_of_two(rows, "matrix row count")?;
    let column_bits = exact_power_of_two(columns, "matrix column count")?;
    let expected_values = rows
        .checked_mul(columns)
        .ok_or_else(|| Error::new(ERR_LENGTH, "matrix size overflow"))?;
    if values.len() != expected_values || values.len() > MAX_MATRIX_BYTES {
        return Err(Error::new(
            ERR_LENGTH,
            "signed-i8 matrix buffer has the wrong length",
        ));
    }
    if point.len() != row_bits + column_bits {
        return Err(Error::new(
            ERR_LENGTH,
            "matrix MLE point has the wrong length",
        ));
    }

    let row_weights = eq_evals(&point[..row_bits]);
    let column_weights = eq_evals(&point[row_bits..]);
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

/// Wire length: 10-byte header, one claimed Y per block, four round
/// evaluations per inner bit, then final X and W evaluations per block.
fn gemm_batch_proof_wire_len(block_count: usize, inner: usize) -> Result<usize, Error> {
    if block_count == 0 || block_count > MAX_GEMM_BATCH_BLOCKS || block_count > u16::MAX as usize {
        return Err(Error::new(
            ERR_LENGTH,
            format!("GEMM batch block count must be in 1..={MAX_GEMM_BATCH_BLOCKS}"),
        ));
    }
    if inner == 0 || inner > MAX_INNER {
        return Err(Error::new(
            ERR_LENGTH,
            format!("GEMM batch inner dimension must be in 1..={MAX_INNER}"),
        ));
    }
    if !inner.is_power_of_two() {
        return Err(Error::new(
            ERR_LENGTH,
            "GEMM batch inner dimension must be a power of two",
        ));
    }
    let rounds = inner.trailing_zeros() as usize;
    10usize
        .checked_add(
            (3usize
                .checked_mul(block_count)
                .and_then(|value| value.checked_add(4 * rounds))
                .ok_or_else(|| Error::new(ERR_LENGTH, "GEMM batch proof size overflow"))?)
            .checked_mul(SCALAR_BYTES)
            .ok_or_else(|| Error::new(ERR_LENGTH, "GEMM batch proof size overflow"))?,
        )
        .ok_or_else(|| Error::new(ERR_LENGTH, "GEMM batch proof size overflow"))
}

fn exact_power_of_two(value: usize, label: &str) -> Result<usize, Error> {
    if value == 0 || !value.is_power_of_two() {
        return Err(Error::new(
            ERR_LENGTH,
            format!("{label} must be a positive power of two"),
        ));
    }
    Ok(value.trailing_zeros() as usize)
}

/// Length-prefixed encoding: `u32 label length, label, u64 value length, value`
/// per field, little-endian.
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

/// Reduces a little-endian 256-bit candidate modulo the Pallas scalar
/// modulus if it is below `CHALLENGE_LIMIT`, otherwise rejects it.
fn challenge_scalar(digest: &[u8; 32]) -> Option<Fr> {
    let limbs: [u64; 4] = std::array::from_fn(|index| {
        u64::from_le_bytes(digest[index * 8..(index + 1) * 8].try_into().unwrap())
    });
    if limbs.iter().rev().cmp(CHALLENGE_LIMIT.iter().rev()) != std::cmp::Ordering::Less {
        return None;
    }
    let mut wide = [0u8; 64];
    wide[..32].copy_from_slice(digest);
    Some(Fr::from_uniform_bytes(&wide))
}

/// SHA-256 hash-chain Fiat-Shamir transcript.
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

    fn squeeze_scalar(&mut self, label: &[u8]) -> Result<Fr, Error> {
        for attempt in 0..=u32::MAX {
            let attempt_bytes = attempt.to_le_bytes();
            let challenge_record =
                gemm_batch_record(&[(b"challenge_label", label), (b"attempt", &attempt_bytes)]);
            let digest = sha256_array(&[&self.state, &challenge_record]);
            if let Some(scalar) = challenge_scalar(&digest) {
                let scalar_bytes = encode_scalar(&scalar);
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
        Err(Error::new(
            ERR_INTERNAL,
            "GEMM batch challenge rejection sampling did not terminate",
        ))
    }
}

fn indexed_label(prefix: &[u8], index: usize) -> Vec<u8> {
    let mut label = Vec::with_capacity(prefix.len() + 4);
    label.extend_from_slice(prefix);
    label.extend_from_slice(&(index as u32).to_le_bytes());
    label
}

/// Evaluates the MLE of `values` at `point`, most significant bit first.
fn fold_mle(mut values: Vec<Fr>, point: &[Fr]) -> Result<Fr, Error> {
    if values.len() != 1usize << point.len() {
        return Err(Error::new(
            ERR_LENGTH,
            "GEMM MLE table does not match its point",
        ));
    }
    for challenge in point {
        fold_table_once(&mut values, *challenge);
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

/// Proves `Y_b = X_b @ W_b` for every block `b` with one batched sumcheck
/// over the inner dimension.
///
/// Row and column points `r_i`, `r_j` are drawn from the transcript, each
/// block's claim `Y_b(r_i, r_j)` is absorbed, and the claims are combined
/// with transcript coefficients (the first is one). The sumcheck runs over
/// `sum_k X_b(r_i, k) * W_b(k, r_j) * mask_b(k)`, where `mask_b` selects the
/// first `valid_inner[b]` positions; X and W padding beyond that must be zero.
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
) -> Result<Vec<u8>, Error> {
    let row_bits = exact_power_of_two(rows, "GEMM batch rows")?;
    let inner_bits = exact_power_of_two(inner, "GEMM batch inner dimension")?;
    let column_bits = exact_power_of_two(columns, "GEMM batch columns")?;
    gemm_batch_proof_wire_len(block_count, inner)?;
    if valid_inner.len() != block_count {
        return Err(Error::new(ERR_LENGTH, "wrong valid-inner count"));
    }
    let x_per_block = rows
        .checked_mul(inner)
        .ok_or_else(|| Error::new(ERR_LENGTH, "GEMM X size overflow"))?;
    let w_per_block = inner
        .checked_mul(columns)
        .ok_or_else(|| Error::new(ERR_LENGTH, "GEMM W size overflow"))?;
    let y_per_block = rows
        .checked_mul(columns)
        .ok_or_else(|| Error::new(ERR_LENGTH, "GEMM Y size overflow"))?;
    if x_values.len() != block_count * x_per_block
        || w_values.len() != block_count * w_per_block
        || y_values.len() != block_count * y_per_block
    {
        return Err(Error::new(
            ERR_LENGTH,
            "GEMM batch witness buffers have the wrong length",
        ));
    }
    let total_bytes = x_values
        .len()
        .checked_add(w_values.len())
        .and_then(|value| value.checked_add(y_values.len() * std::mem::size_of::<i64>()))
        .and_then(|value| value.checked_add(valid_inner.len() * 4))
        .ok_or_else(|| Error::new(ERR_LENGTH, "GEMM batch witness size overflow"))?;
    if total_bytes > MAX_GEMM_BATCH_BYTES {
        return Err(Error::new(
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
                return Err(Error::new(
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
                    return Err(Error::new(ERR_FORMAT, "GEMM X padding is nonzero"));
                }
            }
            if w_raw[valid * columns..].iter().any(|value| *value != 0) {
                return Err(Error::new(ERR_FORMAT, "GEMM W padding is nonzero"));
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
        .collect::<Result<Vec<_>, Error>>()?;
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
        // Round polynomial evaluated at 0, 1, 2, 3.
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
            return Err(Error::new(
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
    output.extend_from_slice(&GEMM_BATCH_PROOF_VERSION.to_le_bytes());
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

/// Runs `operation`, records its error message, and contains panics.
fn ffi_status<F>(operation: F) -> i32
where
    F: FnOnce() -> Result<(), Error>,
{
    clear_last_error();
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            set_last_error(error.message);
            error.code
        }
        Err(_) => {
            set_last_error("native panic contained at FFI boundary");
            ERR_INTERNAL
        }
    }
}

unsafe fn required_input<'a, T>(
    pointer: *const T,
    length: usize,
    label: &str,
) -> Result<&'a [T], Error> {
    if pointer.is_null() {
        return Err(Error::new(ERR_NULL, format!("null {label} pointer")));
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
) -> Result<&'a [T], Error> {
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
) -> Result<&'a mut [u8], Error> {
    if pointer.is_null() {
        return Err(Error::new(ERR_NULL, format!("null {label} pointer")));
    }
    // SAFETY: the C ABI requires a writable allocation of `length` bytes.
    Ok(unsafe { std::slice::from_raw_parts_mut(pointer, length) })
}

#[no_mangle]
pub extern "C" fn verathos_gemm_abi_version() -> u32 {
    ABI_VERSION
}

/// Copies the calling thread's last error as a NUL-terminated UTF-8 string.
/// Returns the full message length excluding the terminator.
///
/// # Safety
///
/// If `capacity` is non-zero, `output` must point to `capacity` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn verathos_gemm_last_error(output: *mut u8, capacity: usize) -> usize {
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

/// Exact proof length for `block_count` blocks of inner dimension `inner`,
/// or 0 if the shape is invalid.
#[no_mangle]
pub extern "C" fn verathos_gemm_batch_proof_size(block_count: usize, inner: usize) -> usize {
    gemm_batch_proof_wire_len(block_count, inner).unwrap_or(0)
}

/// Produces one transcript-bound aggregate sumcheck for equal padded GEMM blocks.
///
/// X is packed block-major as `blocks * rows * inner` signed bytes, W as
/// `blocks * inner * columns`, and Y as `blocks * rows * columns` signed i64s.
/// All dimensions and explicit buffer lengths are checked before dereferencing.
///
/// # Safety
///
/// `batch_digest` must point to 32 readable bytes, each input pointer to its
/// stated element count, `output_proof` to `proof_capacity` writable bytes and
/// `output_proof_len` to one writable `usize`.
#[no_mangle]
pub unsafe extern "C" fn verathos_gemm_batch_prove_i8(
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
            return Err(Error::new(
                ERR_BUFFER,
                "GEMM batch proof output buffer is too small",
            ));
        }
        if output_proof_len.is_null() {
            return Err(Error::new(
                ERR_NULL,
                "null GEMM batch proof-length output",
            ));
        }
        let expected_x = block_count
            .checked_mul(rows)
            .and_then(|value| value.checked_mul(inner))
            .ok_or_else(|| Error::new(ERR_LENGTH, "GEMM X size overflow"))?;
        let expected_w = block_count
            .checked_mul(inner)
            .and_then(|value| value.checked_mul(columns))
            .ok_or_else(|| Error::new(ERR_LENGTH, "GEMM W size overflow"))?;
        let expected_y = block_count
            .checked_mul(rows)
            .and_then(|value| value.checked_mul(columns))
            .ok_or_else(|| Error::new(ERR_LENGTH, "GEMM Y size overflow"))?;
        if x_value_count != expected_x
            || w_value_count != expected_w
            || y_value_count != expected_y
            || valid_inner_count != block_count
        {
            return Err(Error::new(
                ERR_LENGTH,
                "GEMM batch explicit buffer lengths are inconsistent",
            ));
        }
        let digest_raw = unsafe { required_input(batch_digest, 32, "batch digest")? };
        let digest: [u8; 32] = digest_raw
            .try_into()
            .map_err(|_| Error::new(ERR_LENGTH, "batch digest must be exactly 32 bytes"))?;
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

/// Evaluates the MLE of a row-major `rows x columns` int8 matrix at a point
/// of `log2(rows) + log2(columns)` scalars (row coordinates first, most
/// significant bit first).
///
/// # Safety
///
/// `values` must point to `values_len` readable signed bytes,
/// `point_bytes` must point to `point_count * 32` readable bytes (or may be
/// null when `point_count` is zero), and `output_evaluation` must point to
/// 32 writable bytes.
#[no_mangle]
pub unsafe extern "C" fn verathos_gemm_evaluate_i8_matrix(
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
            return Err(Error::new(
                ERR_LENGTH,
                "matrix MLE point has the wrong length",
            ));
        }
        let expected_values = rows
            .checked_mul(columns)
            .ok_or_else(|| Error::new(ERR_LENGTH, "matrix size overflow"))?;
        if values_len != expected_values || values_len > MAX_MATRIX_BYTES {
            return Err(Error::new(
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

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(value: u64) -> Fr {
        Fr::from(value)
    }

    fn p_minus_one_le() -> [u8; 32] {
        encode_scalar(&-Fr::ONE)
    }

    fn add_le(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
        let mut out = [0u8; 32];
        let mut carry = 0u16;
        for index in 0..32 {
            let sum = left[index] as u16 + right[index] as u16 + carry;
            out[index] = sum as u8;
            carry = sum >> 8;
        }
        assert_eq!(carry, 0);
        out
    }

    fn one_le() -> [u8; 32] {
        let mut out = [0u8; 32];
        out[0] = 1;
        out
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
    fn challenge_sampling_reduces_below_three_moduli_and_rejects_above() {
        let p_minus_one = p_minus_one_le();
        let p = add_le(&p_minus_one, &one_le());
        let two_p = add_le(&p, &p);
        let three_p = add_le(&two_p, &p);
        let three_p_minus_one = add_le(&two_p, &p_minus_one);
        let limit: [u8; 32] = std::array::from_fn(|index| {
            CHALLENGE_LIMIT[index / 8].to_le_bytes()[index % 8]
        });
        assert_eq!(three_p, limit);
        assert_eq!(challenge_scalar(&[0u8; 32]), Some(Fr::ZERO));
        assert_eq!(challenge_scalar(&p_minus_one), Some(-Fr::ONE));
        assert_eq!(challenge_scalar(&p), Some(Fr::ZERO));
        assert_eq!(challenge_scalar(&add_le(&p, &one_le())), Some(Fr::ONE));
        assert_eq!(challenge_scalar(&two_p), Some(Fr::ZERO));
        assert_eq!(challenge_scalar(&three_p_minus_one), Some(-Fr::ONE));
        assert_eq!(challenge_scalar(&three_p), None);
        assert_eq!(challenge_scalar(&[0xff; 32]), None);
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
        let flat = values
            .iter()
            .map(|value| signed_i64_to_scalar(*value as i64))
            .collect::<Vec<_>>();
        assert_eq!(
            evaluate_i8_matrix_values(&values, rows, columns, &point).unwrap(),
            fold_mle(flat, &point).unwrap(),
        );
    }

    #[test]
    fn signed_i64_boolean_evaluations_follow_msb_first_order() {
        let values = [-3i64, 5, 7, -11];
        let flat = values.iter().map(|value| signed_i64_to_scalar(*value)).collect::<Vec<_>>();
        for (index, expected) in values.iter().enumerate() {
            let point = [scalar((index >> 1) as u64), scalar((index & 1) as u64)];
            assert_eq!(fold_mle(flat.clone(), &point).unwrap(), signed_i64_to_scalar(*expected));
            let weights = eq_evals(&point);
            assert_eq!(weights[index], Fr::ONE);
            assert_eq!(weights.iter().filter(|weight| **weight == Fr::ZERO).count(), 3);
        }
    }

    struct Witness {
        x: Vec<i8>,
        w: Vec<i8>,
        y: Vec<i64>,
        valid: Vec<u32>,
    }

    fn witness(block_count: usize, rows: usize, inner: usize, columns: usize) -> Witness {
        let mut x = vec![0i8; block_count * rows * inner];
        let mut w = vec![0i8; block_count * inner * columns];
        let mut y = vec![0i64; block_count * rows * columns];
        let mut valid = Vec::with_capacity(block_count);
        for block in 0..block_count {
            let valid_inner = inner - block % 3;
            valid.push(valid_inner as u32);
            for row in 0..rows {
                for k in 0..valid_inner {
                    x[block * rows * inner + row * inner + k] =
                        ((block * 31 + row * 17 + k * 7) as u8) as i8;
                }
            }
            for k in 0..valid_inner {
                for column in 0..columns {
                    w[block * inner * columns + k * columns + column] =
                        ((block * 11 + k * 13 + column * 5 + 3) as u8) as i8;
                }
            }
            for row in 0..rows {
                for column in 0..columns {
                    y[block * rows * columns + row * columns + column] = (0..inner)
                        .map(|k| {
                            x[block * rows * inner + row * inner + k] as i64
                                * w[block * inner * columns + k * columns + column] as i64
                        })
                        .sum();
                }
            }
        }
        Witness { x, w, y, valid }
    }

    fn cubic_at(evaluations: &[Fr; 4], at: Fr) -> Fr {
        let nodes = [0u64, 1, 2, 3].map(scalar);
        let mut total = Fr::ZERO;
        for (i, value) in evaluations.iter().enumerate() {
            let mut numerator = Fr::ONE;
            let mut denominator = Fr::ONE;
            for (j, node) in nodes.iter().enumerate() {
                if i != j {
                    numerator *= at - node;
                    denominator *= nodes[i] - node;
                }
            }
            total += *value * numerator * denominator.invert().unwrap();
        }
        total
    }

    fn read_scalars(bytes: &[u8]) -> Vec<Fr> {
        bytes.chunks_exact(SCALAR_BYTES).map(|chunk| decode_scalar(chunk).unwrap()).collect()
    }

    #[test]
    fn gemm_batch_proof_satisfies_sumcheck_relation() {
        let (block_count, rows, inner, columns) = (3usize, 4usize, 16usize, 8usize);
        let witness = witness(block_count, rows, inner, columns);
        let digest = [7u8; 32];
        let proof = gemm_batch_prove_i8_impl(
            &digest, &witness.x, &witness.w, &witness.y, &witness.valid, block_count, rows,
            inner, columns,
        )
        .unwrap();
        assert_eq!(proof.len(), gemm_batch_proof_wire_len(block_count, inner).unwrap());
        assert_eq!(&proof[..4], GEMM_BATCH_PROOF_MAGIC);
        assert_eq!(&proof[4..10], &[1, 0, 3, 0, 4, 0]);

        let rounds_bits = inner.trailing_zeros() as usize;
        let body = read_scalars(&proof[10..]);
        let (claimed_y, rest) = body.split_at(block_count);
        let (round_values, finals) = rest.split_at(4 * rounds_bits);
        let (final_x, final_w) = finals.split_at(block_count);

        let mut transcript = GemmBatchTranscript::new(&digest);
        let row_point = (0..rows.trailing_zeros() as usize)
            .map(|index| transcript.squeeze_scalar(&indexed_label(b"r_i", index)).unwrap())
            .collect::<Vec<_>>();
        let column_point = (0..columns.trailing_zeros() as usize)
            .map(|index| transcript.squeeze_scalar(&indexed_label(b"r_j", index)).unwrap())
            .collect::<Vec<_>>();
        let mut coefficients = vec![Fr::ONE];
        for (index, value) in claimed_y.iter().enumerate() {
            let block = &witness.y[index * rows * columns..(index + 1) * rows * columns];
            let flat = block.iter().map(|value| signed_i64_to_scalar(*value)).collect();
            let point = [row_point.clone(), column_point.clone()].concat();
            assert_eq!(*value, fold_mle(flat, &point).unwrap());
            transcript.absorb_scalar(&indexed_label(b"claimed_y_eval", index), value);
        }
        for index in 1..block_count {
            coefficients
                .push(transcript.squeeze_scalar(&indexed_label(b"relation_alpha", index)).unwrap());
        }
        let mut claim = coefficients
            .iter()
            .zip(claimed_y)
            .fold(Fr::ZERO, |sum, (coefficient, value)| sum + *coefficient * value);
        let mut inner_point = Vec::new();
        for (round_index, chunk) in round_values.chunks_exact(4).enumerate() {
            let evaluations: [Fr; 4] = chunk.try_into().unwrap();
            assert_eq!(evaluations[0] + evaluations[1], claim);
            for (label, value) in [b"round_e0", b"round_e1", b"round_e2", b"round_e3"]
                .iter()
                .zip(&evaluations)
            {
                transcript.absorb_scalar(&indexed_label(*label, round_index), value);
            }
            let challenge = transcript.squeeze_scalar(&indexed_label(b"r_k", round_index)).unwrap();
            claim = cubic_at(&evaluations, challenge);
            inner_point.push(challenge);
        }
        let terminal = (0..block_count).fold(Fr::ZERO, |sum, block| {
            let mask = (0..inner)
                .map(|k| if k < witness.valid[block] as usize { Fr::ONE } else { Fr::ZERO })
                .collect();
            let mask_value = fold_mle(mask, &inner_point).unwrap();
            sum + coefficients[block] * final_x[block] * final_w[block] * mask_value
        });
        assert_eq!(terminal, claim);
    }

    #[test]
    fn gemm_batch_rejects_wrong_product_and_nonzero_padding() {
        let (block_count, rows, inner, columns) = (2usize, 2usize, 8usize, 4usize);
        let mut witness = witness(block_count, rows, inner, columns);
        witness.y[3] += 1;
        let error = gemm_batch_prove_i8_impl(
            &[0u8; 32], &witness.x, &witness.w, &witness.y, &witness.valid, block_count, rows,
            inner, columns,
        )
        .unwrap_err();
        assert_eq!(error.code, ERR_CRYPTO);

        let mut witness = self::witness(block_count, rows, inner, columns);
        witness.valid[0] = 4;
        let error = gemm_batch_prove_i8_impl(
            &[0u8; 32], &witness.x, &witness.w, &witness.y, &witness.valid, block_count, rows,
            inner, columns,
        )
        .unwrap_err();
        assert_eq!(error.code, ERR_FORMAT);
    }

    #[test]
    fn gemm_batch_ffi_rejects_inconsistent_lengths() {
        let (block_count, rows, inner, columns) = (1usize, 2usize, 4usize, 2usize);
        let witness = witness(block_count, rows, inner, columns);
        let capacity = verathos_gemm_batch_proof_size(block_count, inner);
        assert_eq!(capacity, 10 + (3 + 4 * 2) * SCALAR_BYTES);
        assert_eq!(verathos_gemm_batch_proof_size(0, inner), 0);
        assert_eq!(verathos_gemm_batch_proof_size(1, 3), 0);
        let mut output = vec![0u8; capacity];
        let mut written = 0usize;
        let status = unsafe {
            verathos_gemm_batch_prove_i8(
                [0u8; 32].as_ptr(),
                witness.x.as_ptr(),
                witness.x.len() - 1,
                witness.w.as_ptr(),
                witness.w.len(),
                witness.y.as_ptr(),
                witness.y.len(),
                witness.valid.as_ptr(),
                witness.valid.len(),
                block_count,
                rows,
                inner,
                columns,
                output.as_mut_ptr(),
                capacity,
                &mut written,
            )
        };
        assert_eq!(status, ERR_LENGTH);
        let message_len = unsafe { verathos_gemm_last_error(ptr::null_mut(), 0) };
        let mut message = vec![0u8; message_len + 1];
        unsafe { verathos_gemm_last_error(message.as_mut_ptr(), message.len()) };
        assert_eq!(
            &message[..message_len],
            b"GEMM batch explicit buffer lengths are inconsistent"
        );
        assert_eq!(written, 0);
    }
}
