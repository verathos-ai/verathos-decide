//! The run plan read from stdin and its validation.

use crate::{hex, tls::Pin};
use hyper::{
    header::{self, HeaderMap, HeaderName, HeaderValue},
    Uri,
};
use serde::Deserialize;
use serde_json::value::RawValue;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    ops::RangeInclusive,
    time::Duration,
};

pub const MAX_TARGETS: usize = 100_000;
pub const MAX_RATE_RPS: u32 = 1_000_000;
/// Longest warmup plus test.
pub const MAX_RUN_S: u32 = 3_600;
pub const MAX_TIMEOUT_S: u32 = 3_600;
pub const MAX_STATE_WORDS: u32 = 100_000;
pub const MAX_SAMPLES: usize = 10_000;
pub const MAX_START_DELAY_MS: u64 = 3_600_000;
pub const MIN_SEED_BYTES: usize = 16;
pub const MAX_RESPONSE_BYTES: usize = 16 << 20;

pub const REQUEST_ID_HEADER: HeaderName = HeaderName::from_static("x-verathos-request-id");

/// Headers the generator sets itself or that would break HTTP/1.1 framing.
const RESERVED_HEADERS: [HeaderName; 6] = [
    header::CONNECTION,
    header::CONTENT_LENGTH,
    header::CONTENT_TYPE,
    header::HOST,
    header::TRANSFER_ENCODING,
    REQUEST_ID_HEADER,
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PlanInput {
    start_at_ms: u64,
    warmup_s: f64,
    test_s: f64,
    grace_s: f64,
    timeout_s: f64,
    max_response_bytes: usize,
    words: Vec<String>,
    seed: String,
    targets: Vec<TargetInput>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TargetInput {
    id: String,
    url: String,
    tls: String,
    rate_rps: f64,
    max_outstanding: usize,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    model: String,
    state_words: (u32, u32),
    questions: Box<RawValue>,
    samples: usize,
}

#[derive(Debug)]
pub struct Plan {
    pub start_at_ms: u64,
    pub warmup: Duration,
    pub test: Duration,
    /// Before the window: how early a request may be scheduled and its reply inside the window still
    /// count. At most the warmup.
    pub grace: Duration,
    pub timeout: Duration,
    /// Largest reply body read; a larger 200 reply is `too_large`.
    pub max_response_bytes: usize,
    pub words: Vec<String>,
    /// ChaCha key for all generated randomness: SHA-256 of the seed bytes.
    pub key: [u8; 32],
    pub targets: Vec<Target>,
}

#[derive(Debug)]
pub struct Target {
    pub id: String,
    pub uri: Uri,
    /// The pin of the endpoint's certificate: the SHA-256 of its SubjectPublicKeyInfo.
    pub pin: Pin,
    pub rate_rps: f64,
    pub max_outstanding: usize,
    pub headers: HeaderMap,
    pub model: String,
    pub state_words: RangeInclusive<u32>,
    pub questions: Box<RawValue>,
    pub samples: usize,
}

impl Plan {
    /// Parses and validates a plan; `now_ms` bounds how far in the future it may start.
    pub fn parse(input: &[u8], now_ms: u64) -> Result<Self, String> {
        let input: PlanInput =
            serde_json::from_slice(input).map_err(|error| format!("invalid plan JSON: {error}"))?;
        let warmup = seconds("warmup_s", input.warmup_s, 0.0, f64::from(MAX_RUN_S))?;
        let test = seconds(
            "test_s",
            input.test_s,
            f64::MIN_POSITIVE,
            f64::from(MAX_RUN_S),
        )?;
        if warmup + test > Duration::from_secs(MAX_RUN_S.into()) {
            return Err(format!("warmup_s + test_s must be at most {MAX_RUN_S}"));
        }
        let grace = seconds("grace_s", input.grace_s, 0.0, input.warmup_s)?;
        if !(1..=MAX_RESPONSE_BYTES).contains(&input.max_response_bytes) {
            return Err(format!(
                "max_response_bytes must be between 1 and {MAX_RESPONSE_BYTES}"
            ));
        }
        let timeout = seconds(
            "timeout_s",
            input.timeout_s,
            f64::MIN_POSITIVE,
            f64::from(MAX_TIMEOUT_S),
        )?;
        if input.start_at_ms > now_ms.saturating_add(MAX_START_DELAY_MS) {
            return Err(format!(
                "start_at_ms is more than {} s in the future",
                MAX_START_DELAY_MS / 1000
            ));
        }
        if input.words.is_empty() || input.words.iter().any(String::is_empty) {
            return Err("words must be a non-empty list of non-empty strings".into());
        }
        let key = seed_key(&input.seed)?;
        if input.targets.is_empty() || input.targets.len() > MAX_TARGETS {
            return Err(format!("targets must hold 1 to {MAX_TARGETS} entries"));
        }
        let mut ids = HashSet::new();
        let targets = input
            .targets
            .into_iter()
            .enumerate()
            .map(|(index, target)| {
                if !ids.insert(target.id.clone()) {
                    return Err(format!("targets[{index}]: duplicate id {:?}", target.id));
                }
                let id = format!("{:?}", target.id);
                Target::validate(target)
                    .map_err(|error| format!("targets[{index}] ({id}): {error}"))
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            start_at_ms: input.start_at_ms,
            warmup,
            test,
            grace,
            timeout,
            max_response_bytes: input.max_response_bytes,
            words: input.words,
            key,
            targets,
        })
    }
}

impl Target {
    fn validate(input: TargetInput) -> Result<Self, String> {
        if input.id.is_empty() {
            return Err("id must not be empty".into());
        }
        let uri = https_uri(&input.url)?;
        let pin = tls_pin(&input.tls)?;
        if !(input.rate_rps.is_finite()
            && (0.0..=f64::from(MAX_RATE_RPS)).contains(&input.rate_rps))
        {
            return Err(format!("rate_rps must be between 0 and {MAX_RATE_RPS}"));
        }
        if input.max_outstanding == 0 {
            return Err("max_outstanding must be at least 1".into());
        }
        if input.model.is_empty() {
            return Err("model must not be empty".into());
        }
        let (low, high) = input.state_words;
        if !(1 <= low && low <= high && high <= MAX_STATE_WORDS) {
            return Err(format!(
                "state_words must be a range [low, high] with 1 <= low <= high <= {MAX_STATE_WORDS}"
            ));
        }
        if !input.questions.get().starts_with('{') {
            return Err("questions must be a JSON object".into());
        }
        if input.samples > MAX_SAMPLES {
            return Err(format!("samples must be at most {MAX_SAMPLES}"));
        }
        Ok(Self {
            headers: static_headers(&input.headers)?,
            id: input.id,
            uri,
            pin,
            rate_rps: input.rate_rps,
            max_outstanding: input.max_outstanding,
            model: input.model,
            state_words: low..=high,
            questions: input.questions,
            samples: input.samples,
        })
    }
}

fn seconds(name: &str, value: f64, min: f64, max: f64) -> Result<Duration, String> {
    if value.is_finite() && (min..=max).contains(&value) {
        Ok(Duration::from_secs_f64(value))
    } else if min > 0.0 {
        Err(format!("{name} must be greater than 0 and at most {max}"))
    } else {
        Err(format!("{name} must be between {min} and {max}"))
    }
}

fn seed_key(seed: &str) -> Result<[u8; 32], String> {
    match hex::decode(seed) {
        Some(bytes) if bytes.len() >= MIN_SEED_BYTES => Ok(Sha256::digest(&bytes).into()),
        _ => Err(format!(
            "seed must be a hex string of at least {MIN_SEED_BYTES} bytes ({} hex digits)",
            2 * MIN_SEED_BYTES
        )),
    }
}

fn https_uri(url: &str) -> Result<Uri, String> {
    let uri: Uri = url
        .parse()
        .map_err(|error| format!("url {url:?} is invalid: {error}"))?;
    if uri.scheme_str() != Some("https") || uri.host().is_none_or(str::is_empty) {
        return Err(format!("url {url:?} must be an absolute https:// URL"));
    }
    Ok(uri)
}

fn tls_pin(pin: &str) -> Result<Pin, String> {
    match hex::decode(pin) {
        Some(bytes) if pin.len() == 64 && pin == pin.to_ascii_lowercase() => {
            Ok(bytes.try_into().expect("32 bytes"))
        }
        _ => Err(format!(
            "tls {pin:?} must be a certificate pin: 64 lowercase hex digits"
        )),
    }
}

fn static_headers(input: &BTreeMap<String, String>) -> Result<HeaderMap, String> {
    let mut headers = HeaderMap::with_capacity(input.len());
    for (name, value) in input {
        let parsed = HeaderName::from_bytes(name.as_bytes())
            .map_err(|_| format!("header name {name:?} is invalid"))?;
        if RESERVED_HEADERS.contains(&parsed) {
            return Err(format!("header {name:?} is set by the load generator"));
        }
        let value = HeaderValue::from_str(value)
            .map_err(|_| format!("header {name:?} has an invalid value"))?;
        if headers.insert(parsed, value).is_some() {
            return Err(format!("header {name:?} is given twice"));
        }
    }
    Ok(headers)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::{json, Value};

    pub const NOW_MS: u64 = 1_791_310_000_000;

    pub fn plan_json() -> Value {
        json!({
            "start_at_ms": NOW_MS + 1_000,
            "warmup_s": 2.0,
            "test_s": 8.0,
            "grace_s": 2.0,
            "timeout_s": 20.0,
            "max_response_bytes": 65536,
            "words": ["invoice", "refund", "late", "card"],
            "seed": "00112233445566778899aabbccddeeff",
            "targets": [{
                "id": "hotkey/gpu0",
                "url": "https://203.0.113.5:8091/e/gpu0/v1/systemone",
                "tls": "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff",
                "rate_rps": 42.5,
                "max_outstanding": 2000,
                "headers": {"X-Verathos-Session": "abc", "X-Verathos-Session-Signature": "0xdef"},
                "model": "kev-4b",
                "state_words": [150, 1200],
                "questions": {"intent": {"type": "choice", "options": ["refund", "other"]}},
                "samples": 10
            }]
        })
    }

    fn parse(plan: &Value) -> Result<Plan, String> {
        Plan::parse(plan.to_string().as_bytes(), NOW_MS)
    }

    fn rejects(edit: impl FnOnce(&mut Value), expected: &str) {
        let mut plan = plan_json();
        edit(&mut plan);
        let error = parse(&plan).expect_err("plan should be rejected");
        assert!(
            error.contains(expected),
            "{error:?} should mention {expected:?}"
        );
    }

    #[test]
    fn accepts_a_valid_plan() {
        let plan = parse(&plan_json()).unwrap();
        assert_eq!(plan.warmup, Duration::from_secs(2));
        assert_eq!(plan.test, Duration::from_secs(8));
        assert_eq!(plan.grace, Duration::from_secs(2));
        assert_eq!(plan.timeout, Duration::from_secs(20));
        assert_eq!(
            plan.key,
            <[u8; 32]>::from(Sha256::digest(
                hex::decode("00112233445566778899aabbccddeeff").unwrap()
            ))
        );
        let target = &plan.targets[0];
        assert_eq!(target.uri.host(), Some("203.0.113.5"));
        assert_eq!(target.uri.port_u16(), Some(8091));
        assert_eq!(target.uri.path(), "/e/gpu0/v1/systemone");
        assert_eq!(target.pin[..2], [0x00, 0x11]);
        assert_eq!(target.state_words, 150..=1200);
        assert_eq!(target.headers["x-verathos-session"], "abc");
        assert_eq!(target.headers.len(), 2);
    }

    #[test]
    fn keeps_questions_verbatim() {
        let questions = r#"{"z": {"type": "choice", "options": ["b", "a"]},  "a": 1.50}"#;
        let text = plan_json().to_string().replace(
            &plan_json()["targets"][0]["questions"].to_string(),
            questions,
        );
        assert!(text.contains(questions));
        let plan = Plan::parse(text.as_bytes(), NOW_MS).unwrap();
        assert_eq!(plan.targets[0].questions.get(), questions);
    }

    #[test]
    fn rejects_invalid_timing() {
        rejects(|p| p["warmup_s"] = json!(-1.0), "warmup_s");
        rejects(|p| p["test_s"] = json!(0.0), "test_s");
        rejects(|p| p["timeout_s"] = json!(0), "timeout_s");
        rejects(|p| p["grace_s"] = json!(-1.0), "grace_s");
        rejects(|p| p["grace_s"] = json!(2.5), "grace_s");
        rejects(|p| p["max_response_bytes"] = json!(0), "max_response_bytes");
        rejects(|p| p["test_s"] = json!(3_600.0), "warmup_s + test_s");
        rejects(
            |p| p["start_at_ms"] = json!(NOW_MS + 2 * MAX_START_DELAY_MS),
            "start_at_ms",
        );
        rejects(|p| p["start_at_ms"] = json!(-5), "invalid plan JSON");
    }

    #[test]
    fn rejects_invalid_vocabulary_seed_and_shape() {
        rejects(|p| p["words"] = json!([]), "words");
        rejects(|p| p["words"] = json!(["ok", ""]), "words");
        rejects(|p| p["seed"] = json!("0011"), "seed");
        rejects(
            |p| p["seed"] = json!("zz112233445566778899aabbccddeeff"),
            "seed",
        );
        rejects(|p| p["targets"] = json!([]), "targets");
        rejects(|p| p["extra"] = json!(1), "unknown field");
        rejects(|p| p["targets"][0]["rate"] = json!(1), "unknown field");
        rejects(
            |p| {
                p.as_object_mut().unwrap().remove("seed");
            },
            "missing field",
        );
    }

    #[test]
    fn rejects_invalid_targets() {
        let target =
            |field: &'static str, value: Value| move |p: &mut Value| p["targets"][0][field] = value;
        rejects(target("rate_rps", json!(-0.5)), "rate_rps");
        rejects(target("rate_rps", json!(2e6)), "rate_rps");
        rejects(target("max_outstanding", json!(0)), "max_outstanding");
        rejects(target("state_words", json!([0, 10])), "state_words");
        rejects(target("state_words", json!([10, 9])), "state_words");
        rejects(target("state_words", json!([1, 2, 3])), "invalid plan JSON");
        rejects(target("questions", json!(["a"])), "questions");
        rejects(target("samples", json!(MAX_SAMPLES + 1)), "samples");
        rejects(target("model", json!("")), "model");
        rejects(target("id", json!("")), "id");
        rejects(
            target("url", json!("http://203.0.113.5/e/gpu0/v1/systemone")),
            "https://",
        );
        rejects(target("url", json!("/e/gpu0/v1/systemone")), "https://");
        rejects(target("tls", json!("0011")), "certificate pin");
        rejects(
            target("tls", json!("00".repeat(31) + "AB")),
            "certificate pin",
        );
        rejects(target("tls", json!("zz".repeat(32))), "certificate pin");
        rejects(
            |p| {
                p["targets"][0].as_object_mut().unwrap().remove("tls");
            },
            "missing field",
        );
        rejects(target("url", json!("http://exa mple/")), "invalid");
        rejects(
            target("headers", json!({"Content-Type": "text/plain"})),
            "set by the load generator",
        );
        rejects(
            target("headers", json!({"x-verathos-request-id": "1"})),
            "set by the load generator",
        );
        rejects(
            target("headers", json!({"X-A": "1", "x-a": "2"})),
            "given twice",
        );
        rejects(target("headers", json!({"bad name": "1"})), "header name");
        rejects(
            target("headers", json!({"x-a": "line\nbreak"})),
            "invalid value",
        );
    }

    #[test]
    fn rejects_duplicate_ids_and_names_the_target() {
        let mut plan = plan_json();
        let target = plan["targets"][0].clone();
        plan["targets"].as_array_mut().unwrap().push(target);
        assert!(parse(&plan)
            .unwrap_err()
            .contains("targets[1]: duplicate id"));

        let mut plan = plan_json();
        plan["targets"][0]["rate_rps"] = json!(-1);
        assert!(parse(&plan)
            .unwrap_err()
            .starts_with("targets[0] (\"hotkey/gpu0\"): rate_rps"));
    }

    #[test]
    fn headers_default_to_empty_and_zero_rate_is_allowed() {
        let mut plan = plan_json();
        plan["targets"][0]
            .as_object_mut()
            .unwrap()
            .remove("headers");
        plan["targets"][0]["rate_rps"] = json!(0);
        let plan = parse(&plan).unwrap();
        assert!(plan.targets[0].headers.is_empty());
        assert_eq!(plan.targets[0].rate_rps, 0.0);
    }
}
