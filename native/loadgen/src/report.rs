//! The result object written to stdout.

use crate::{
    run::Outcome,
    stats::{Latency, Sample, Summary},
};
use serde::Serialize;
use serde_json::value::RawValue;
use std::{collections::BTreeMap, io::Write, time::Duration};

#[derive(Serialize)]
struct Report<'a> {
    started_at_ms: u64,
    late_start_ms: u64,
    targets: Vec<TargetReport<'a>>,
}

#[derive(Serialize)]
struct TargetReport<'a> {
    id: &'a str,
    sent: u64,
    skipped: u64,
    completed: u64,
    delivered_rps: f64,
    latency_ms: LatencyReport,
    errors: BTreeMap<String, u64>,
    max_send_lag_ms: f64,
    late_sends: u64,
    samples: Vec<SampleReport<'a>>,
}

/// All fields are null when nothing completed in the window.
#[derive(Serialize)]
struct LatencyReport {
    p50: Option<f64>,
    p95: Option<f64>,
    max: Option<f64>,
}

#[derive(Serialize)]
struct SampleReport<'a> {
    request: &'a RawValue,
    request_id: &'a str,
    status: u16,
    response: &'a RawValue,
    sent_at_ms: u64,
    completed_at_ms: u64,
}

pub fn write(outcome: &Outcome, mut out: impl Write) -> Result<(), String> {
    let report = Report {
        started_at_ms: outcome.clock.unix_ms(outcome.started_at),
        late_start_ms: outcome.late_start_ms,
        targets: outcome
            .targets
            .iter()
            .map(|(id, summary)| target_report(id, summary, outcome))
            .collect::<Result<_, _>>()?,
    };
    serde_json::to_writer(&mut out, &report)
        .map_err(|error| format!("writing the result: {error}"))?;
    writeln!(out)
        .and_then(|()| out.flush())
        .map_err(|error| format!("writing the result: {error}"))
}

fn target_report<'a>(
    id: &'a str,
    summary: &'a Summary,
    outcome: &Outcome,
) -> Result<TargetReport<'a>, String> {
    let latency = summary.latency.as_ref();
    let percentile =
        |pick: fn(&Latency) -> Duration| latency.map(|latency| milliseconds(pick(latency)));
    Ok(TargetReport {
        id,
        sent: summary.sent,
        skipped: summary.skipped,
        completed: summary.completed,
        delivered_rps: summary.completed as f64 / outcome.test.as_secs_f64(),
        latency_ms: LatencyReport {
            p50: percentile(|latency| latency.p50),
            p95: percentile(|latency| latency.p95),
            max: percentile(|latency| latency.max),
        },
        errors: summary
            .errors
            .iter()
            .map(|(kind, count)| (kind.to_string(), *count))
            .collect(),
        max_send_lag_ms: milliseconds(summary.max_send_lag),
        late_sends: summary.late_sends,
        samples: summary
            .samples
            .iter()
            .map(|sample| sample_report(sample, outcome))
            .collect::<Result<_, _>>()?,
    })
}

fn sample_report<'a>(sample: &'a Sample, outcome: &Outcome) -> Result<SampleReport<'a>, String> {
    let raw = |body: &'a [u8]| {
        serde_json::from_slice::<&RawValue>(body)
            .map_err(|error| format!("a kept body is not JSON: {error}"))
    };
    Ok(SampleReport {
        request: raw(&sample.request)?,
        request_id: &sample.request_id,
        status: 200,
        response: raw(&sample.response)?,
        sent_at_ms: outcome.clock.unix_ms(sample.sent_at),
        completed_at_ms: outcome.clock.unix_ms(sample.completed_at),
    })
}

/// Milliseconds with microsecond resolution.
fn milliseconds(duration: Duration) -> f64 {
    duration.as_micros() as f64 / 1000.0
}
