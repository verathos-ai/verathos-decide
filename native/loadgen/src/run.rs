//! Drives a plan: one scheduler task per target sends open loop on that target's timetable, every
//! request runs in its own task, and the per-target stats are read once the window has closed.
//!
//! Target `i` sends request `k` at `start + (k + phase_i) / rate_i`, with `phase_i` in [0, 1) drawn
//! from the seed. The phases keep targets of equal rates from firing in lockstep, which would turn
//! the total load into bursts of one request per target.

use crate::{
    client::{self, Client},
    plan::{Plan, Target},
    reservoir::Reservoir,
    stats::{Exchange, Summary, TargetStats, Window},
    workload::Generator,
};
use rand::RngExt;
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex, MutexGuard, PoisonError,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::{sync::mpsc, time::sleep_until};

/// Maps monotonic instants to wall-clock milliseconds through a single reading of both clocks.
#[derive(Clone, Copy, Debug)]
pub struct Clock {
    anchor: Instant,
    anchor_ms: u64,
}

impl Clock {
    pub fn now() -> Self {
        Self {
            anchor: Instant::now(),
            anchor_ms: unix_ms(),
        }
    }

    pub fn unix_ms(&self, at: Instant) -> u64 {
        self.anchor_ms + at.saturating_duration_since(self.anchor).as_millis() as u64
    }
}

pub fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_millis() as u64)
}

pub struct Outcome {
    pub clock: Clock,
    pub started_at: Instant,
    pub late_start_ms: u64,
    pub test: Duration,
    pub targets: Vec<(String, Summary)>,
}

struct Shared {
    generator: Generator,
    client: Client,
    start: Instant,
    /// Warmup plus test: no request is scheduled at or after `start + duration`.
    duration: Duration,
    timeout: Duration,
}

struct TargetState {
    index: usize,
    plan: Target,
    /// Fraction of the send interval by which this target's timetable is shifted.
    phase: f64,
    stats: Mutex<TargetStats>,
    outstanding: AtomicUsize,
}

/// Counts a request as outstanding for as long as it lives.
struct InFlight(Arc<TargetState>);

impl InFlight {
    fn new(target: Arc<TargetState>) -> Self {
        target.outstanding.fetch_add(1, Ordering::AcqRel);
        Self(target)
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        self.0.outstanding.fetch_sub(1, Ordering::AcqRel);
    }
}

pub async fn run(plan: Plan) -> Result<Outcome, String> {
    let clock = Clock::now();
    let (start, late_start_ms) = match plan.start_at_ms.checked_sub(clock.anchor_ms) {
        Some(wait_ms) => (clock.anchor + Duration::from_millis(wait_ms), 0),
        None => (clock.anchor, clock.anchor_ms - plan.start_at_ms),
    };
    let window = Window {
        start: start + plan.warmup,
        end: start + plan.warmup + plan.test,
        earliest_send: start + plan.warmup - plan.grace,
    };
    let generator = Generator::new(plan.key, plan.words);
    let targets = plan
        .targets
        .into_iter()
        .enumerate()
        .map(|(index, plan)| {
            let mut rng = generator.target_rng(index);
            let phase = rng.random::<f64>();
            let samples = Reservoir::new(plan.samples, rng);
            Arc::new(TargetState {
                index,
                plan,
                phase,
                stats: Mutex::new(TargetStats::new(window, samples)),
                outstanding: AtomicUsize::new(0),
            })
        })
        .collect::<Vec<_>>();
    let shared = Arc::new(Shared {
        generator,
        client: Client::new(plan.max_response_bytes),
        start,
        duration: plan.warmup + plan.test,
        timeout: plan.timeout,
    });

    let total_rps = targets
        .iter()
        .map(|target| target.plan.rate_rps)
        .sum::<f64>();
    if late_start_ms > 0 {
        eprintln!("verathos-loadgen: start time passed {late_start_ms} ms ago; starting now");
    }
    eprintln!(
        "verathos-loadgen: {} targets, {total_rps:.1} requests/s in total, \
         {:.1} s warmup + {:.1} s test, starting in {} ms",
        targets.len(),
        plan.warmup.as_secs_f64(),
        plan.test.as_secs_f64(),
        start.saturating_duration_since(Instant::now()).as_millis(),
    );

    // Every request holds a sender until it has recorded its send, so the channel closes only after
    // all dispatched requests have recorded their send lag.
    let (sends_recorded, mut all_sends_recorded) = mpsc::channel::<()>(1);
    let schedulers = targets
        .iter()
        .map(|target| {
            tokio::spawn(schedule(
                shared.clone(),
                target.clone(),
                sends_recorded.clone(),
            ))
        })
        .collect::<Vec<_>>();
    drop(sends_recorded);
    for scheduler in schedulers {
        scheduler
            .await
            .map_err(|error| format!("a scheduler task failed: {error}"))?;
    }
    while all_sends_recorded.recv().await.is_some() {}
    sleep_until(window.end.into()).await;

    let summaries = targets
        .iter()
        .map(|target| (target.plan.id.clone(), lock(&target.stats).summary()))
        .collect();
    Ok(Outcome {
        clock,
        started_at: start,
        late_start_ms,
        test: plan.test,
        targets: summaries,
    })
}

async fn schedule(shared: Arc<Shared>, target: Arc<TargetState>, sends_recorded: mpsc::Sender<()>) {
    let rate = target.plan.rate_rps;
    if rate <= 0.0 {
        return;
    }
    for position in 0u64.. {
        let offset = Duration::from_secs_f64((position as f64 + target.phase) / rate);
        if offset >= shared.duration {
            break;
        }
        let scheduled = shared.start + offset;
        if scheduled > Instant::now() {
            sleep_until(scheduled.into()).await;
        }
        if target.outstanding.load(Ordering::Acquire) >= target.plan.max_outstanding {
            lock(&target.stats).record_skip(scheduled);
            continue;
        }
        let in_flight = InFlight::new(target.clone());
        tokio::spawn(send(
            shared.clone(),
            in_flight,
            position,
            scheduled,
            sends_recorded.clone(),
        ));
    }
}

async fn send(
    shared: Arc<Shared>,
    in_flight: InFlight,
    position: u64,
    scheduled: Instant,
    sends_recorded: mpsc::Sender<()>,
) {
    let target = &in_flight.0;
    let request = shared
        .generator
        .request(target.index, position, &target.plan);
    let post = client::post(&target.plan, &request);
    let sent_at = Instant::now();
    lock(&target.stats).record_send(scheduled, sent_at);
    drop(sends_recorded);
    let outcome = shared.client.exchange(post, sent_at + shared.timeout).await;
    let exchange = Exchange {
        request_id: request.id,
        request: request.body,
        scheduled,
        sent_at,
        outcome,
    };
    let mut stats = lock(&target.stats);
    // The completion time is read under the lock, so any record made after the final read of the
    // stats is timed at or after the window end: nothing inside the window can be missed.
    stats.record_exchange(exchange, Instant::now());
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}
