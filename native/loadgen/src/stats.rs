//! Per-target accounting over the measurement window.
//!
//! A request counts as completed if its 200 reply arrives inside the window and it was scheduled no
//! earlier than the grace before the window opened. Completions inside the window are bounded by
//! what the target serves while the window is open, so declaring more than it serves gains nothing;
//! a backlog of older requests served first does not count either. An honest target whose latency
//! stays within the grace has every reply inside the window counted. Send-side counts (`sent`,
//! `skipped`, send lag) are attributed by scheduled time inside the window.

use crate::reservoir::Reservoir;
use bytes::Bytes;
use std::{
    collections::BTreeMap,
    fmt,
    time::{Duration, Instant},
};

/// A send this far behind its scheduled time means the generator itself fell behind.
pub const LATE_SEND: Duration = Duration::from_millis(50);

#[derive(Clone, Copy, Debug)]
pub struct Window {
    pub start: Instant,
    pub end: Instant,
    /// The start minus the grace: replies to requests scheduled earlier do not count.
    pub earliest_send: Instant,
}

impl Window {
    /// Half-open: an event timed after the final read of the stats can never fall inside.
    pub fn contains(&self, at: Instant) -> bool {
        self.start <= at && at < self.end
    }

    pub fn counts_send(&self, scheduled: Instant) -> bool {
        self.earliest_send <= scheduled && scheduled < self.end
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorKind {
    Http(u16),
    Timeout,
    Connect,
    InvalidJson,
    TooLarge,
    Transport,
    /// The generator's own host ran out of sockets, file descriptors or local ports.
    Local,
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http(status) => write!(f, "http_{status}"),
            Self::Timeout => f.write_str("timeout"),
            Self::Connect => f.write_str("connect"),
            Self::InvalidJson => f.write_str("invalid_json"),
            Self::TooLarge => f.write_str("too_large"),
            Self::Transport => f.write_str("transport"),
            Self::Local => f.write_str("local"),
        }
    }
}

/// A finished request.
pub struct Exchange {
    pub request_id: String,
    pub request: Bytes,
    pub scheduled: Instant,
    pub sent_at: Instant,
    /// The body of a 200 reply that is a JSON object, or why the request failed.
    pub outcome: Result<Bytes, ErrorKind>,
}

pub struct Sample {
    pub request_id: String,
    pub request: Bytes,
    pub response: Bytes,
    pub sent_at: Instant,
    pub completed_at: Instant,
}

#[derive(Debug, PartialEq)]
pub struct Latency {
    pub p50: Duration,
    pub p95: Duration,
    pub max: Duration,
}

pub struct Summary {
    pub sent: u64,
    pub skipped: u64,
    pub completed: u64,
    pub latency: Option<Latency>,
    pub errors: BTreeMap<ErrorKind, u64>,
    pub max_send_lag: Duration,
    pub late_sends: u64,
    /// Ordered by send time.
    pub samples: Vec<Sample>,
}

pub struct TargetStats {
    window: Window,
    sent: u64,
    skipped: u64,
    late_sends: u64,
    max_send_lag: Duration,
    latencies: Vec<Duration>,
    errors: BTreeMap<ErrorKind, u64>,
    samples: Reservoir<Sample>,
}

impl TargetStats {
    pub fn new(window: Window, samples: Reservoir<Sample>) -> Self {
        Self {
            window,
            sent: 0,
            skipped: 0,
            late_sends: 0,
            max_send_lag: Duration::ZERO,
            latencies: Vec::new(),
            errors: BTreeMap::new(),
            samples,
        }
    }

    /// A scheduled send skipped because `max_outstanding` requests were already in flight.
    pub fn record_skip(&mut self, scheduled: Instant) {
        if self.window.contains(scheduled) {
            self.skipped += 1;
        }
    }

    pub fn record_send(&mut self, scheduled: Instant, sent_at: Instant) {
        if !self.window.contains(scheduled) {
            return;
        }
        self.sent += 1;
        let lag = sent_at.saturating_duration_since(scheduled);
        self.max_send_lag = self.max_send_lag.max(lag);
        if lag > LATE_SEND {
            self.late_sends += 1;
        }
    }

    pub fn record_exchange(&mut self, exchange: Exchange, completed_at: Instant) {
        if !(self.window.counts_send(exchange.scheduled) && self.window.contains(completed_at)) {
            return;
        }
        match exchange.outcome {
            Ok(response) => {
                self.latencies
                    .push(completed_at.saturating_duration_since(exchange.sent_at));
                // The response is copied so that a kept sample does not pin the connection's
                // read buffer it was sliced from.
                self.samples.offer(|| Sample {
                    request_id: exchange.request_id,
                    request: exchange.request,
                    response: Bytes::copy_from_slice(&response),
                    sent_at: exchange.sent_at,
                    completed_at,
                });
            }
            Err(kind) => *self.errors.entry(kind).or_default() += 1,
        }
    }

    pub fn summary(&mut self) -> Summary {
        self.latencies.sort_unstable();
        let mut samples = self.samples.take();
        samples.sort_by_key(|sample| sample.sent_at);
        Summary {
            sent: self.sent,
            skipped: self.skipped,
            completed: self.latencies.len() as u64,
            latency: latency(&self.latencies),
            errors: self.errors.clone(),
            max_send_lag: self.max_send_lag,
            late_sends: self.late_sends,
            samples,
        }
    }
}

/// Nearest-rank percentiles of sorted latencies.
fn latency(sorted: &[Duration]) -> Option<Latency> {
    let max = *sorted.last()?;
    let rank = |percent: usize| sorted[(sorted.len() * percent).div_ceil(100) - 1];
    Some(Latency {
        p50: rank(50),
        p95: rank(95),
        max,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha12Rng;

    const MS: Duration = Duration::from_millis(1);

    struct Fixture {
        origin: Instant,
        stats: TargetStats,
    }

    impl Fixture {
        /// Window from 2 s to 10 s after the origin, counting requests sent from 1 s on.
        fn new(samples: usize) -> Self {
            let origin = Instant::now();
            let window = Window {
                start: origin + 2_000 * MS,
                end: origin + 10_000 * MS,
                earliest_send: origin + 1_000 * MS,
            };
            let reservoir = Reservoir::new(samples, ChaCha12Rng::seed_from_u64(0));
            Self {
                origin,
                stats: TargetStats::new(window, reservoir),
            }
        }

        fn at(&self, ms: u64) -> Instant {
            self.origin + Duration::from_millis(ms)
        }

        fn finish(&mut self, sent_ms: u64, completed_ms: u64, outcome: Result<Bytes, ErrorKind>) {
            let exchange = Exchange {
                request_id: format!("{sent_ms}"),
                request: Bytes::from_static(b"{}"),
                scheduled: self.at(sent_ms),
                sent_at: self.at(sent_ms),
                outcome,
            };
            self.stats.record_exchange(exchange, self.at(completed_ms));
        }
    }

    fn ok() -> Result<Bytes, ErrorKind> {
        Ok(Bytes::from_static(b"{\"answers\":{}}"))
    }

    #[test]
    fn outcomes_count_replies_inside_the_window_to_requests_sent_from_the_grace_on() {
        let mut fixture = Fixture::new(100);
        fixture.finish(500, 2_500, ok()); // scheduled before the grace: a held-back warmup answer
        fixture.finish(1_000, 2_100, ok()); // scheduled at the start of the grace
        fixture.finish(1_500, 1_999, ok()); // answered before the window opened
        fixture.finish(5_000, 5_500, ok());
        fixture.finish(9_000, 9_999, ok());
        fixture.finish(9_000, 10_000, ok()); // answered as the window closes
        fixture.finish(3_000, 3_001, Err(ErrorKind::Http(503)));
        fixture.finish(3_000, 3_100, Err(ErrorKind::Local));
        fixture.finish(500, 3_000, Err(ErrorKind::Timeout)); // scheduled before the grace
        let summary = fixture.stats.summary();
        assert_eq!(summary.completed, 3);
        assert_eq!(
            summary.latency,
            Some(Latency {
                p50: 999 * MS,
                p95: 1_100 * MS,
                max: 1_100 * MS
            })
        );
        assert_eq!(
            summary
                .errors
                .iter()
                .map(|(kind, n)| (kind.to_string(), *n))
                .collect::<Vec<_>>(),
            vec![("http_503".to_owned(), 1), ("local".to_owned(), 1)]
        );
        let sent = summary
            .samples
            .iter()
            .map(|sample| sample.request_id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(sent, ["1000", "5000", "9000"]);
    }

    #[test]
    fn sends_count_by_scheduled_time_and_track_lag() {
        let mut fixture = Fixture::new(0);
        let at = |ms| fixture.at(ms);
        let (warmup, early, late, after) = (at(1_990), at(2_000), at(6_000), at(10_000));
        fixture.stats.record_send(warmup, warmup + 500 * MS); // scheduled in warmup: ignored
        fixture.stats.record_send(early, early + 10 * MS);
        fixture.stats.record_send(late, late + 51 * MS);
        fixture.stats.record_send(late, late + 50 * MS); // exactly at the threshold is not late
        fixture.stats.record_send(after, after);
        fixture.stats.record_skip(warmup);
        fixture.stats.record_skip(late);
        fixture.stats.record_skip(after);
        let summary = fixture.stats.summary();
        assert_eq!(
            (summary.sent, summary.skipped, summary.late_sends),
            (3, 1, 1)
        );
        assert_eq!(summary.max_send_lag, 51 * MS);
        assert_eq!(summary.completed, 0);
        assert_eq!(summary.latency, None);
        assert!(summary.samples.is_empty());
    }

    #[test]
    fn percentiles_use_nearest_rank() {
        let values =
            |range: std::ops::RangeInclusive<u32>| range.map(|ms| ms * MS).collect::<Vec<_>>();
        assert_eq!(latency(&[]), None);
        assert_eq!(
            latency(&values(7..=7)),
            Some(Latency {
                p50: 7 * MS,
                p95: 7 * MS,
                max: 7 * MS
            })
        );
        assert_eq!(
            latency(&values(1..=100)),
            Some(Latency {
                p50: 50 * MS,
                p95: 95 * MS,
                max: 100 * MS
            })
        );
        assert_eq!(
            latency(&values(1..=10)),
            Some(Latency {
                p50: 5 * MS,
                p95: 10 * MS,
                max: 10 * MS
            })
        );
    }

    #[test]
    fn error_kinds_have_stable_names() {
        let names = [
            ErrorKind::Http(429),
            ErrorKind::Timeout,
            ErrorKind::Connect,
            ErrorKind::InvalidJson,
            ErrorKind::TooLarge,
            ErrorKind::Transport,
            ErrorKind::Local,
        ]
        .map(|kind| kind.to_string());
        assert_eq!(
            names,
            [
                "http_429",
                "timeout",
                "connect",
                "invalid_json",
                "too_large",
                "transport",
                "local"
            ]
        );
    }
}
