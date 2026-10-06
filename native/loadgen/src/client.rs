//! The shared HTTP/1.1 keep-alive client and the classification of every exchange.

use crate::{
    plan::{Target, REQUEST_ID_HEADER},
    stats::ErrorKind,
    workload::GeneratedRequest,
};
use bytes::Bytes;
use http_body_util::{BodyExt, Full, LengthLimitError, Limited};
use hyper::{
    header::{HeaderValue, CONTENT_TYPE},
    Method, Request, StatusCode,
};
use hyper_util::{
    client::legacy::{connect::HttpConnector, Client as PooledClient},
    rt::TokioExecutor,
};
use serde_json::value::RawValue;
use std::time::{Duration, Instant};
use tokio::time::timeout_at;

/// Shorter than the 5 s keep-alive of common Python servers (uvicorn), so an idle connection is
/// never reused just as the server closes it. Enforced when a connection is taken from the pool;
/// the pool's background reaper stays off because it evicts every expired connection at once under
/// the pool's global lock, which stalls all sends when thousands expire together.
const POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(3);

/// `errno` values that mean this host, not the target, ran out of a resource.
const LOCAL_ERRNOS: [i32; 5] = [
    23,  // ENFILE: system file table full
    24,  // EMFILE: process file descriptors exhausted
    98,  // EADDRINUSE
    99,  // EADDRNOTAVAIL: no local port left
    105, // ENOBUFS
];

#[derive(Clone)]
pub struct Client {
    pool: PooledClient<HttpConnector, Full<Bytes>>,
    max_response_bytes: usize,
}

impl Client {
    pub fn new(max_response_bytes: usize) -> Self {
        let mut connector = HttpConnector::new();
        connector.set_nodelay(true);
        let pool = PooledClient::builder(TokioExecutor::new())
            .pool_idle_timeout(POOL_IDLE_TIMEOUT)
            .pool_max_idle_per_host(usize::MAX)
            .build(connector);
        Self {
            pool,
            max_response_bytes,
        }
    }

    /// Sends one request. `Ok` carries the body of a 200 reply that is a JSON object.
    pub async fn exchange(
        &self,
        request: Request<Full<Bytes>>,
        deadline: Instant,
    ) -> Result<Bytes, ErrorKind> {
        let deadline = tokio::time::Instant::from_std(deadline);
        let response = match timeout_at(deadline, self.pool.request(request)).await {
            Err(_) => return Err(ErrorKind::Timeout),
            Ok(Err(error)) if local_failure(&error) => return Err(ErrorKind::Local),
            Ok(Err(error)) if error.is_connect() => return Err(ErrorKind::Connect),
            Ok(Err(_)) => return Err(ErrorKind::Transport),
            Ok(Ok(response)) => response,
        };
        let status = response.status();
        let body = timeout_at(
            deadline,
            Limited::new(response.into_body(), self.max_response_bytes).collect(),
        )
        .await;
        if status != StatusCode::OK {
            // The body was read anyway so that the connection can return to the pool.
            return Err(ErrorKind::Http(status.as_u16()));
        }
        let body = match body {
            Err(_) => return Err(ErrorKind::Timeout),
            Ok(Err(error)) if error.is::<LengthLimitError>() => return Err(ErrorKind::TooLarge),
            Ok(Err(_)) => return Err(ErrorKind::Transport),
            Ok(Ok(collected)) => collected.to_bytes(),
        };
        if is_json_object(&body) {
            Ok(body)
        } else {
            Err(ErrorKind::InvalidJson)
        }
    }
}

pub fn post(target: &Target, request: &GeneratedRequest) -> Request<Full<Bytes>> {
    let mut post = Request::new(Full::new(request.body.clone()));
    *post.method_mut() = Method::POST;
    *post.uri_mut() = target.uri.clone();
    let headers = post.headers_mut();
    headers.clone_from(&target.headers);
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(
        REQUEST_ID_HEADER,
        HeaderValue::from_str(&request.id).expect("a hex request id is a valid header value"),
    );
    post
}

/// Whether an error comes from this host running out of sockets, descriptors or ports.
fn local_failure(error: &(dyn std::error::Error + 'static)) -> bool {
    let mut source = Some(error);
    while let Some(error) = source {
        if let Some(io) = error.downcast_ref::<std::io::Error>() {
            if io
                .raw_os_error()
                .is_some_and(|code| LOCAL_ERRNOS.contains(&code))
            {
                return true;
            }
        }
        source = error.source();
    }
    false
}

/// Full syntax check without building the value.
pub fn is_json_object(body: &[u8]) -> bool {
    serde_json::from_slice::<&RawValue>(body).is_ok_and(|value| value.get().starts_with('{'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_only_json_objects() {
        assert!(is_json_object(br#"{"answers": {"a": 0.5}}"#));
        assert!(is_json_object(b" {} \n"));
        assert!(!is_json_object(b"[1, 2]"));
        assert!(!is_json_object(b"\"text\""));
        assert!(!is_json_object(b"{\"a\": }"));
        assert!(!is_json_object(b"{} {}"));
        assert!(!is_json_object(b"{\"a\": \"\xff\"}"));
        assert!(!is_json_object(b""));
    }
}
