//! Open-loop HTTP load generator for the throughput audit.
//!
//! Reads one JSON plan from stdin, drives every target at its declared rate at the same time, and
//! writes one JSON result object to stdout. Logs go to stderr. Exits 0 on success, 2 for an invalid
//! plan and 1 for an internal failure.

mod client;
mod hex;
mod plan;
mod report;
mod reservoir;
mod run;
mod stats;
mod workload;

use plan::Plan;
use stats::Summary;
use std::{
    io::{self, BufWriter, Read},
    process::ExitCode,
};

enum Failure {
    InvalidInput(String),
    Internal(String),
}

fn main() -> ExitCode {
    match try_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(Failure::InvalidInput(message)) => {
            eprintln!("verathos-loadgen: invalid plan: {message}");
            ExitCode::from(2)
        }
        Err(Failure::Internal(message)) => {
            eprintln!("verathos-loadgen: {message}");
            ExitCode::FAILURE
        }
    }
}

fn try_main() -> Result<(), Failure> {
    let mut input = Vec::new();
    io::stdin()
        .read_to_end(&mut input)
        .map_err(|error| Failure::Internal(format!("reading the plan from stdin: {error}")))?;
    let plan = Plan::parse(&input, run::unix_ms()).map_err(Failure::InvalidInput)?;
    raise_open_file_limit();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| Failure::Internal(format!("starting the runtime: {error}")))?;
    let outcome = runtime
        .block_on(run::run(plan))
        .map_err(Failure::Internal)?;
    report::write(&outcome, BufWriter::new(io::stdout().lock())).map_err(Failure::Internal)?;
    let total = |count: fn(&Summary) -> u64| {
        outcome
            .targets
            .iter()
            .map(|(_, summary)| count(summary))
            .sum::<u64>()
    };
    eprintln!(
        "verathos-loadgen: done: {} sent, {} completed, {} late sends",
        total(|summary| summary.sent),
        total(|summary| summary.completed),
        total(|summary| summary.late_sends),
    );
    // Requests still in flight after the window are abandoned rather than awaited.
    runtime.shutdown_background();
    Ok(())
}

/// Every in-flight request holds a socket, so the soft open-file limit is raised to the hard one.
fn raise_open_file_limit() {
    match rlimit::increase_nofile_limit(u64::MAX) {
        Ok(limit) => eprintln!("verathos-loadgen: open file limit {limit}"),
        Err(error) => eprintln!("verathos-loadgen: could not raise the open file limit: {error}"),
    }
}
