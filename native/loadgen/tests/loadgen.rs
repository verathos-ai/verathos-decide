//! End-to-end runs of the binary against local HTTP servers.

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::{body::Incoming, server::conn::http1, service::service_fn, Request, Response};
use hyper_util::rt::TokioIo;
use serde_json::{json, Value};
use std::{
    convert::Infallible,
    net::SocketAddr,
    process::{ExitStatus, Stdio},
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpSocket},
    process::Command,
    time::{sleep_until, Instant},
};

const SESSION: &str = "session-token";

#[derive(Clone, Copy)]
enum Overload {
    /// Serve every request, at most `capacity` per second, in arrival order.
    Queue,
    /// Answer 429 to requests above `capacity` per second.
    Reject,
}

#[derive(Clone, Copy)]
enum Reply {
    Answer,
    Status(u16),
    Body(&'static str),
    Oversized,
}

#[derive(Clone, Copy)]
struct Behavior {
    delay: Duration,
    capacity: Option<(f64, Overload)>,
    reply: Reply,
}

impl Behavior {
    fn answer(delay_ms: u64) -> Self {
        Self {
            delay: Duration::from_millis(delay_ms),
            capacity: None,
            reply: Reply::Answer,
        }
    }

    fn capacity(self, rps: f64, overload: Overload) -> Self {
        Self {
            capacity: Some((rps, overload)),
            ..self
        }
    }

    fn reply(self, reply: Reply) -> Self {
        Self { reply, ..self }
    }
}

/// Fixed-rate admission (GCRA) with a burst of a few requests.
struct Limiter {
    interval: Duration,
    overload: Overload,
    next_free: Instant,
}

impl Limiter {
    const BURST: u32 = 3;

    /// When to start serving a request that arrived at `now`, or `None` to reject it.
    fn admit(&mut self, now: Instant) -> Option<Instant> {
        let slot = self.next_free.max(now);
        match self.overload {
            Overload::Queue => {
                self.next_free = slot + self.interval;
                Some(slot)
            }
            Overload::Reject if slot <= now + Self::BURST * self.interval => {
                self.next_free = slot + self.interval;
                Some(now)
            }
            Overload::Reject => None,
        }
    }
}

async fn handle(
    request: Request<Incoming>,
    behavior: Behavior,
    limiter: Option<Arc<Mutex<Limiter>>>,
) -> Result<Response<Full<Bytes>>, Infallible> {
    let arrived = Instant::now();
    let header = |name: &str| {
        request
            .headers()
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned)
    };
    let (request_id, session, content_type) = (
        header("x-verathos-request-id"),
        header("x-verathos-session"),
        header("content-type"),
    );
    let body = request
        .into_body()
        .collect()
        .await
        .map(|body| body.to_bytes())
        .unwrap_or_default();
    let body: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let start = match &limiter {
        None => Some(arrived),
        Some(limiter) => limiter.lock().unwrap().admit(arrived),
    };
    let Some(start) = start else {
        return Ok(response(429, "{\"error\":\"over capacity\"}".into()));
    };
    sleep_until(start + behavior.delay).await;
    Ok(match behavior.reply {
        Reply::Answer => {
            let answer = json!({
                "answers": {"intent": {"refund": 0.75, "other": 0.25}},
                "request_id": request_id,
                "session": session,
                "content_type": content_type,
                "state": body["state"],
            });
            response(200, answer.to_string())
        }
        Reply::Status(status) => response(status, "{}".into()),
        Reply::Body(text) => response(200, text.into()),
        Reply::Oversized => response(200, format!("{{\"pad\":\"{}\"}}", "x".repeat(2 << 20))),
    })
}

fn response(status: u16, body: String) -> Response<Full<Bytes>> {
    let mut response = Response::new(Full::new(Bytes::from(body)));
    *response.status_mut() = status.try_into().unwrap();
    response
}

async fn serve(behavior: Behavior) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let limiter = behavior.capacity.map(|(rps, overload)| {
        Arc::new(Mutex::new(Limiter {
            interval: Duration::from_secs_f64(1.0 / rps),
            overload,
            next_free: Instant::now(),
        }))
    });
    tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                continue;
            };
            let limiter = limiter.clone();
            tokio::spawn(async move {
                let service = service_fn(move |request| handle(request, behavior, limiter.clone()));
                let _ = http1::Builder::new()
                    .serve_connection(TokioIo::new(stream), service)
                    .await;
            });
        }
    });
    address
}

/// Reads part of each request, then closes the connection without answering.
async fn serve_hangup() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                continue;
            };
            tokio::spawn(async move {
                let mut buffer = [0u8; 64];
                let _ = stream.read(&mut buffer).await;
            });
        }
    });
    address
}

/// A bound socket that never listens: connections to its port are refused, and no other test can
/// take the port.
fn closed_port() -> (TcpSocket, SocketAddr) {
    let socket = TcpSocket::new_v4().unwrap();
    socket.bind("127.0.0.1:0".parse().unwrap()).unwrap();
    let address = socket.local_addr().unwrap();
    (socket, address)
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

fn questions() -> Value {
    json!({"intent": {"type": "choice", "options": ["refund", "other"]}})
}

fn target(id: &str, address: SocketAddr, rate_rps: f64, samples: usize) -> Value {
    json!({
        "id": id,
        "url": format!("http://{address}/e/gpu0/v1/systemone"),
        "rate_rps": rate_rps,
        "max_outstanding": 2000,
        "headers": {"X-Verathos-Session": SESSION},
        "model": "kev-4b",
        "state_words": [150, 1200],
        "questions": questions(),
        "samples": samples,
    })
}

fn plan(
    start_at_ms: i64,
    warmup_s: f64,
    test_s: f64,
    timeout_s: f64,
    targets: Vec<Value>,
) -> Value {
    json!({
        "start_at_ms": start_at_ms,
        "warmup_s": warmup_s,
        "test_s": test_s,
        "timeout_s": timeout_s,
        "words": ["invoice", "refund", "late", "card", "password", "shipment"],
        "seed": "8f1e4c0b9a7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f",
        "targets": targets,
    })
}

async fn run(stdin: &[u8]) -> (ExitStatus, String, String) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_verathos-loadgen"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    input.write_all(stdin).await.unwrap();
    drop(input);
    let output = child.wait_with_output().await.unwrap();
    let text = |bytes: Vec<u8>| String::from_utf8(bytes).unwrap();
    (output.status, text(output.stdout), text(output.stderr))
}

async fn run_plan(plan: &Value) -> Value {
    let (status, stdout, stderr) = run(plan.to_string().as_bytes()).await;
    assert!(status.success(), "loadgen failed: {stderr}");
    serde_json::from_str(&stdout).unwrap()
}

fn assert_near(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance * expected,
        "{what}: {actual} is not within {}% of {expected}",
        tolerance * 100.0
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn delivered_rate_tracks_the_offered_rate_up_to_capacity() {
    let fast = serve(Behavior::answer(20)).await;
    let queued = serve(Behavior::answer(20).capacity(150.0, Overload::Queue)).await;
    let rejecting = serve(Behavior::answer(20).capacity(100.0, Overload::Reject)).await;
    let (warmup_s, test_s) = (1.0, 2.5);
    let start_at_ms = now_ms() + 300;
    let result = run_plan(&plan(
        start_at_ms,
        warmup_s,
        test_s,
        20.0,
        vec![
            target("fast", fast, 120.0, 5),
            target("queued", queued, 300.0, 0),
            target("rejecting", rejecting, 200.0, 0),
        ],
    ))
    .await;

    assert_eq!(result["started_at_ms"], start_at_ms);
    assert_eq!(result["late_start_ms"], 0);
    let targets = result["targets"].as_array().unwrap();
    let ids = targets
        .iter()
        .map(|target| target["id"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(ids, ["fast", "queued", "rejecting"]);
    for (target, rate, capacity) in [
        (&targets[0], 120.0, 1e9),
        (&targets[1], 300.0, 150.0),
        (&targets[2], 200.0, 100.0),
    ] {
        let id = target["id"].as_str().unwrap();
        assert_near(
            target["sent"].as_f64().unwrap(),
            rate * test_s,
            0.01,
            &format!("{id} sent"),
        );
        assert_eq!(target["skipped"], 0, "{id}");
        assert_eq!(target["late_sends"], 0, "{id}: {target}");
        assert!(target["max_send_lag_ms"].as_f64().unwrap() < 50.0, "{id}");
        let delivered = target["delivered_rps"].as_f64().unwrap();
        assert_near(
            delivered,
            f64::min(rate, capacity),
            0.10,
            &format!("{id} delivered_rps"),
        );
        assert_eq!(delivered, target["completed"].as_f64().unwrap() / test_s);
    }
    assert_eq!(targets[0]["errors"], json!({}));
    assert_eq!(targets[1]["errors"], json!({}));
    let rejected = targets[2]["errors"]["http_429"].as_f64().unwrap();
    assert_near(
        rejected,
        (200.0 - 100.0) * test_s,
        0.15,
        "rejected requests",
    );
    assert_eq!(targets[2]["errors"].as_object().unwrap().len(), 1);

    let latency = &targets[0]["latency_ms"];
    let (p50, p95, max) = (
        latency["p50"].as_f64().unwrap(),
        latency["p95"].as_f64().unwrap(),
        latency["max"].as_f64().unwrap(),
    );
    assert!(
        20.0 <= p50 && p50 <= p95 && p95 <= max && max < 1_000.0,
        "{latency}"
    );
    assert!(
        targets[1]["latency_ms"]["max"].as_f64().unwrap() > 1_000.0,
        "the queue should build up"
    );

    let samples = targets[0]["samples"].as_array().unwrap();
    assert_eq!(samples.len(), 5);
    let window_ms = (start_at_ms + 1_000, start_at_ms + 3_500);
    let mut request_ids = std::collections::HashSet::new();
    for sample in samples {
        let request_id = sample["request_id"].as_str().unwrap();
        assert!(request_id.len() == 32 && request_id.bytes().all(|c| c.is_ascii_hexdigit()));
        assert!(request_ids.insert(request_id));
        assert_eq!(sample["status"], 200);
        let request = &sample["request"];
        assert_eq!(request.as_object().unwrap().len(), 3);
        assert_eq!(request["model"], "kev-4b");
        assert_eq!(request["questions"], questions());
        assert!(request["state"].as_str().unwrap().starts_with("Ticket "));
        let response = &sample["response"];
        assert_eq!(response["request_id"], request_id);
        assert_eq!(response["state"], request["state"]);
        assert_eq!(response["session"], SESSION);
        assert_eq!(response["content_type"], "application/json");
        let (sent, completed) = (
            sample["sent_at_ms"].as_i64().unwrap(),
            sample["completed_at_ms"].as_i64().unwrap(),
        );
        assert!(completed - sent >= 19, "{sample}");
        assert!(
            window_ms.0 <= completed && completed <= window_ms.1,
            "{completed} outside {window_ms:?}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn failures_are_counted_by_kind() {
    let rate = 20.0;
    let (warmup_s, test_s) = (0.5, 1.5);
    let (_refusing, refused) = closed_port();
    let targets = vec![
        ("refused", refused, "connect"),
        (
            "status",
            serve(Behavior::answer(5).reply(Reply::Status(500))).await,
            "http_500",
        ),
        (
            "array",
            serve(Behavior::answer(5).reply(Reply::Body("[1, 2]"))).await,
            "invalid_json",
        ),
        (
            "truncated",
            serve(Behavior::answer(5).reply(Reply::Body("{\"a\": "))).await,
            "invalid_json",
        ),
        (
            "oversized",
            serve(Behavior::answer(5).reply(Reply::Oversized)).await,
            "too_large",
        ),
        ("slow", serve(Behavior::answer(2_000)).await, "timeout"),
        ("hangup", serve_hangup().await, "transport"),
    ];
    let start_at_ms = now_ms() - 5_000;
    let plan = plan(
        start_at_ms,
        warmup_s,
        test_s,
        0.4,
        targets
            .iter()
            .map(|(id, address, _)| target(id, *address, rate, 3))
            .collect(),
    );
    let result = run_plan(&plan).await;

    assert!(result["late_start_ms"].as_i64().unwrap() >= 5_000);
    assert!(result["started_at_ms"].as_i64().unwrap() >= start_at_ms + 5_000);
    for ((id, _, kind), reported) in targets.iter().zip(result["targets"].as_array().unwrap()) {
        assert_eq!(reported["id"], *id);
        assert_eq!(reported["completed"], 0, "{reported}");
        assert_eq!(reported["delivered_rps"], 0.0);
        assert_eq!(
            reported["latency_ms"],
            json!({"p50": null, "p95": null, "max": null})
        );
        assert_eq!(reported["samples"], json!([]));
        let errors = reported["errors"].as_object().unwrap();
        assert_eq!(
            errors.keys().map(String::as_str).collect::<Vec<_>>(),
            [*kind],
            "{id}: {errors:?}"
        );
        assert_near(errors[*kind].as_f64().unwrap(), rate * test_s, 0.2, id);
    }
}

#[tokio::test]
async fn invalid_plans_exit_with_status_2() {
    let mut bad_rate = plan(
        now_ms(),
        1.0,
        1.0,
        1.0,
        vec![target("t", closed_port().1, 1.0, 0)],
    );
    bad_rate["targets"][0]["rate_rps"] = json!(-1.0);
    for (input, expected) in [
        (bad_rate.to_string(), "rate_rps"),
        ("not json".to_owned(), "invalid plan JSON"),
    ] {
        let (status, stdout, stderr) = run(input.as_bytes()).await;
        assert_eq!(status.code(), Some(2), "{stderr}");
        assert!(stdout.is_empty());
        assert!(
            stderr.contains("invalid plan") && stderr.contains(expected),
            "{stderr}"
        );
    }
}
