//! OpenAI-compatible HTTP API.
//!
//! ```text
//! POST /v1/chat/completions   (stream: true → server-sent events)
//! POST /v1/completions
//! GET  /v1/models
//! GET  /health
//! GET  /metrics               (Prometheus text; ?format=json for JSON)
//! GET  /                      the web dashboard (embedded in the binary)
//! GET  /api/info              model, plan, placement and hardware (static)
//! GET  /api/live              memory, residency, caches, turn history (?experts=1 adds the expert map)
//! ```
//!
//! The dashboard and `/api/live` read the session's monitor, which never
//! takes the session lock, so they stay responsive while a request generates.
//!
//! Generation runs on a blocking thread; requests are served one at a time
//! per model (the MVP does not batch concurrent requests).

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use kestrel_backends::{InferenceSession, Monitor};
use kestrel_engine::{ChatMessage, GenParams, GenStats, SamplerConfig};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

pub type SharedSession = Arc<Mutex<Box<dyn InferenceSession>>>;

#[derive(Clone)]
struct AppState {
    session: SharedSession,
    model_id: String,
    default_max_tokens: usize,
    monitor: Option<Monitor>,
    info: Arc<Value>,
}

pub struct ServeOptions {
    pub default_max_tokens: usize,
    /// Merged into `/api/info` (the CLI adds the hardware profile).
    pub extra_info: Value,
}

const DASHBOARD: &str = include_str!("../web/index.html");
const LOGO: &[u8] = include_bytes!("../../../assets/logo.png");

pub fn router(session: SharedSession, opts: ServeOptions) -> Router {
    let (model_id, monitor, mut info) = {
        let s = session.lock().unwrap();
        (s.model_name().to_string(), s.monitor(), s.info())
    };
    if let (Some(dst), Some(src)) = (info.as_object_mut(), opts.extra_info.as_object()) {
        for (k, v) in src {
            dst.insert(k.clone(), v.clone());
        }
    }
    info["server"] = json!({"version": env!("CARGO_PKG_VERSION"), "model_id": model_id});
    let state = AppState { session, model_id, default_max_tokens: opts.default_max_tokens, monitor, info: Arc::new(info) };
    Router::new()
        .route("/", get(|| async { ([("content-type", "text/html; charset=utf-8"), ("cache-control", "no-cache")], DASHBOARD) }))
        .route("/logo.png", get(|| async { ([("content-type", "image/png"), ("cache-control", "max-age=86400")], LOGO) }))
        .route("/api/info", get(api_info))
        .route("/api/live", get(api_live))
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
        .route("/metrics", get(metrics))
        .with_state(state)
}

pub async fn serve(session: SharedSession, addr: &str, opts: ServeOptions) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router(session, opts)).with_graceful_shutdown(async {
        let _ = tokio::signal::ctrl_c().await;
    })
    .await?;
    Ok(())
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn err(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(json!({"error": {"message": msg.into(), "type": "invalid_request_error"}}))).into_response()
}

async fn models(State(s): State<AppState>) -> Json<Value> {
    Json(json!({"object": "list", "data": [{"id": s.model_id, "object": "model", "created": now(), "owned_by": "local"}]}))
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Stop {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
struct Common {
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    min_p: Option<f32>,
    seed: Option<u64>,
    stop: Option<Stop>,
    #[serde(default)]
    stream: bool,
    frequency_penalty: Option<f32>,
    repeat_penalty: Option<f32>,
}

impl Common {
    fn params(&self, default_max: usize) -> GenParams {
        let mut sc = SamplerConfig::default();
        if let Some(t) = self.temperature {
            sc.temperature = t;
        }
        if let Some(p) = self.top_p {
            sc.top_p = p;
        }
        if let Some(k) = self.top_k {
            sc.top_k = k;
        }
        if let Some(m) = self.min_p {
            sc.min_p = m;
        }
        if let Some(s) = self.seed {
            sc.seed = s;
        }
        if let Some(r) = self.repeat_penalty.or(self.frequency_penalty.map(|f| 1.0 + f.max(0.0) * 0.5)) {
            sc.repeat_penalty = r;
        }
        GenParams {
            max_tokens: self.max_completion_tokens.or(self.max_tokens).unwrap_or(default_max),
            sampler: sc,
            ignore_eos: false,
            stop: match &self.stop {
                Some(Stop::One(s)) => vec![s.clone()],
                Some(Stop::Many(v)) => v.clone(),
                None => vec![],
            },
        }
    }
}

#[derive(Deserialize)]
struct ChatReq {
    messages: Vec<InMessage>,
    #[serde(flatten)]
    common: Common,
}

#[derive(Deserialize)]
struct InMessage {
    role: String,
    content: Content,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Content {
    Text(String),
    Parts(Vec<Value>),
}

impl Content {
    fn text(&self) -> String {
        match self {
            Content::Text(s) => s.clone(),
            Content::Parts(p) => p.iter().filter_map(|x| x.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join(""),
        }
    }
}

#[derive(Deserialize)]
struct CompletionReq {
    prompt: PromptIn,
    #[serde(flatten)]
    common: Common,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum PromptIn {
    One(String),
    Many(Vec<String>),
}

fn usage(st: &GenStats) -> Value {
    json!({"prompt_tokens": st.prompt_tokens, "completion_tokens": st.generated, "total_tokens": st.prompt_tokens + st.generated})
}

fn finish(st: &GenStats) -> &'static str {
    if st.stop_reason == "length" {
        "length"
    } else {
        "stop"
    }
}

fn timings(st: &GenStats) -> Value {
    json!({"prompt_ms": st.prefill_s * 1e3, "ttft_ms": st.ttft_s * 1e3, "predicted_ms": st.decode_s * 1e3, "predicted_per_second": st.decode_tok_s, "prompt_per_second": st.prompt_tok_s, "memory": st.memory})
}

async fn chat(State(s): State<AppState>, Json(req): Json<ChatReq>) -> Response {
    let messages: Vec<ChatMessage> = req.messages.iter().map(|m| ChatMessage { role: m.role.clone(), content: m.content.text() }).collect();
    let prompt = {
        let sess = s.session.lock().unwrap();
        match sess.render_chat(&messages) {
            Ok(p) => sess.tokenize(&p),
            Err(e) => return err(StatusCode::BAD_REQUEST, e.to_string()),
        }
    };
    run(s, prompt, req.common, true).await
}

async fn completions(State(s): State<AppState>, Json(req): Json<CompletionReq>) -> Response {
    let text = match &req.prompt {
        PromptIn::One(p) => p.clone(),
        PromptIn::Many(v) if v.len() == 1 => v[0].clone(),
        PromptIn::Many(_) => return err(StatusCode::BAD_REQUEST, "batched prompts are not supported"),
    };
    let prompt = s.session.lock().unwrap().tokenize(&text);
    run(s, prompt, req.common, false).await
}

async fn run(s: AppState, prompt: Vec<u32>, common: Common, is_chat: bool) -> Response {
    let params = common.params(s.default_max_tokens);
    let id = format!("{}-{}", if is_chat { "chatcmpl" } else { "cmpl" }, now());
    let created = now();
    let model = s.model_id.clone();
    if !common.stream {
        let session = s.session.clone();
        let res = tokio::task::spawn_blocking(move || {
            let mut text = String::new();
            let mut sess = session.lock().unwrap();
            let st = sess.generate(&prompt, &params, &mut |piece| {
                text.push_str(piece);
                true
            });
            st.map(|st| (text, st))
        })
        .await;
        return match res {
            Ok(Ok((text, st))) => {
                let choice = if is_chat {
                    json!({"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish(&st)})
                } else {
                    json!({"index": 0, "text": text, "finish_reason": finish(&st)})
                };
                Json(json!({
                    "id": id, "object": if is_chat {"chat.completion"} else {"text_completion"},
                    "created": created, "model": model, "choices": [choice], "usage": usage(&st), "timings": timings(&st),
                }))
                .into_response()
            }
            Ok(Err(e)) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        };
    }

    // Streaming: pieces flow from the blocking generator through a channel.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Value>();
    let session = s.session.clone();
    let (id2, model2) = (id.clone(), model.clone());
    tokio::task::spawn_blocking(move || {
        let chunk = |delta: Value, fin: Value| {
            if is_chat {
                json!({"id": id2, "object": "chat.completion.chunk", "created": created, "model": model2, "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]})
            } else {
                json!({"id": id2, "object": "text_completion", "created": created, "model": model2, "choices": [{"index": 0, "text": delta.get("content").cloned().unwrap_or(json!("")), "finish_reason": fin}]})
            }
        };
        if is_chat {
            let _ = tx.send(chunk(json!({"role": "assistant", "content": ""}), Value::Null));
        }
        let mut sess = session.lock().unwrap();
        let res = sess.generate(&prompt, &params, &mut |piece| tx.send(chunk(json!({"content": piece}), Value::Null)).is_ok());
        match res {
            Ok(st) => {
                let mut last = chunk(json!({}), json!(finish(&st)));
                last["usage"] = usage(&st);
                last["timings"] = timings(&st);
                let _ = tx.send(last);
            }
            Err(e) => {
                let _ = tx.send(json!({"error": {"message": e.to_string()}}));
            }
        }
    });
    let stream = UnboundedReceiverStream::new(rx)
        .map(|v| Ok::<_, std::convert::Infallible>(Event::default().data(v.to_string())))
        .chain(tokio_stream::once(Ok(Event::default().data("[DONE]"))));
    Sse::new(stream).into_response()
}

async fn api_info(State(s): State<AppState>) -> Json<Value> {
    Json((*s.info).clone())
}

async fn api_live(State(s): State<AppState>, Query(q): Query<HashMap<String, String>>) -> Response {
    let with_experts = q.get("experts").is_some_and(|v| v == "1" || v == "true");
    match &s.monitor {
        Some(m) => {
            let m = m.clone();
            match tokio::task::spawn_blocking(move || m(with_experts)).await {
                Ok(v) => Json(v).into_response(),
                Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            }
        }
        None => Json(json!({"busy": s.session.try_lock().is_err()})).into_response(),
    }
}

async fn metrics(State(s): State<AppState>, Query(q): Query<HashMap<String, String>>) -> Response {
    let mut status = match s.session.try_lock() {
        Ok(sess) => sess.status(),
        Err(_) => json!({"busy": true}),
    };
    // While a request generates, the session is locked: take memory and
    // store figures from the lock-free monitor instead.
    if let (Some(m), Some(obj)) = (&s.monitor, status.as_object_mut()) {
        let live = m(false);
        obj.entry("rss_bytes").or_insert(live["rss"].clone());
        obj.entry("ram_budget").or_insert(live["ram"].clone());
        obj.entry("store").or_insert(live["store"].clone());
    }
    if q.get("format").map(String::as_str) == Some("json") {
        return Json(status).into_response();
    }
    let mut out = String::new();
    let mut put = |name: &str, help: &str, v: Option<f64>| {
        if let Some(v) = v {
            out.push_str(&format!("# HELP kestrel_{name} {help}\n# TYPE kestrel_{name} gauge\nkestrel_{name} {v}\n"));
        }
    };
    let f = |p: &str| status.pointer(p).and_then(Value::as_f64);
    put("rss_bytes", "Process resident set size", f("/rss_bytes"));
    put("ram_reserved_bytes", "RAM reserved in the ledger", f("/ram_budget/reserved"));
    put("ram_limit_bytes", "RAM budget", f("/ram_budget/limit"));
    put("weight_leases_total", "Weight group leases", f("/store/leases"));
    put("hit_rate", "Fraction of leases served without waiting for I/O", f("/store/hit_rate"));
    put("prefetch_accuracy", "Prefetches used / issued", f("/store/prefetch_accuracy"));
    put("stall_seconds_total", "Compute time blocked on weight I/O", f("/store/stall_s"));
    put("stream_bytes_total", "Bytes streamed from disk", f("/store/stream_bytes"));
    put("disk_bandwidth_bytes", "Effective streaming bandwidth", f("/store/disk_bw"));
    put("generated_tokens_total", "Generated tokens", f("/totals/generated_tokens"));
    put("prompt_tokens_total", "Prompt tokens", f("/totals/prompt_tokens"));
    put("decode_tokens_per_second", "Decode speed of the last request", f("/totals/last_decode_tok_s"));
    ([("content-type", "text/plain; version=0.0.4")], out).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use tower::ServiceExt;

    struct Mock;
    impl InferenceSession for Mock {
        fn model_name(&self) -> &str {
            "mock-model"
        }
        fn render_chat(&self, messages: &[ChatMessage]) -> anyhow::Result<String> {
            Ok(messages.iter().map(|m| m.content.clone()).collect::<Vec<_>>().join("\n"))
        }
        fn tokenize(&self, text: &str) -> Vec<u32> {
            text.bytes().map(u32::from).collect()
        }
        fn generate(&mut self, _: &[u32], _: &GenParams, on_text: &mut dyn FnMut(&str) -> bool) -> anyhow::Result<GenStats> {
            for p in ["he", "llo"] {
                if !on_text(p) {
                    break;
                }
            }
            Ok(GenStats { generated: 2, ..Default::default() })
        }
        fn status(&self) -> Value {
            json!({"rss_bytes": 1})
        }
        fn info(&self) -> Value {
            json!({"model": {"name": "mock-model"}})
        }
        fn monitor(&self) -> Option<Monitor> {
            Some(Arc::new(|experts| json!({"busy": false, "experts_requested": experts})))
        }
    }

    fn app() -> (SharedSession, Router) {
        let s: SharedSession = Arc::new(Mutex::new(Box::new(Mock)));
        let r = router(s.clone(), ServeOptions { default_max_tokens: 16, extra_info: json!({"hardware": {"cores": 8}}) });
        (s, r)
    }

    async fn get(r: &Router, uri: &str) -> (StatusCode, String, String) {
        let resp = r.clone().oneshot(Request::get(uri).body(Body::empty()).unwrap()).await.unwrap();
        let ct = resp.headers().get("content-type").map(|v| v.to_str().unwrap().to_string()).unwrap_or_default();
        let status = resp.status();
        let body = to_bytes(resp.into_body(), 1 << 24).await.unwrap();
        (status, ct, String::from_utf8_lossy(&body).into_owned())
    }

    #[tokio::test]
    async fn serves_the_dashboard_and_logo() {
        let (_, r) = app();
        let (st, ct, body) = get(&r, "/").await;
        assert_eq!(st, StatusCode::OK);
        assert!(ct.starts_with("text/html"));
        assert!(body.contains("<title>Kestrel</title>") && body.contains("/api/live") && body.trim_end().ends_with("</html>"));
        let (st, ct, _) = get(&r, "/logo.png").await;
        assert_eq!((st, ct.as_str()), (StatusCode::OK, "image/png"));
    }

    #[tokio::test]
    async fn info_merges_hardware_and_server() {
        let (_, r) = app();
        let (_, _, body) = get(&r, "/api/info").await;
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["model"]["name"], "mock-model");
        assert_eq!(v["hardware"]["cores"], 8);
        assert_eq!(v["server"]["model_id"], "mock-model");
    }

    #[tokio::test]
    async fn live_answers_while_the_session_is_generating() {
        let (s, r) = app();
        // Hold the session lock, as a running generation does.
        let guard = s.lock().unwrap();
        let (st, _, body) = tokio::time::timeout(std::time::Duration::from_secs(5), get(&r, "/api/live?experts=1")).await.expect("/api/live must not wait for the session");
        drop(guard);
        assert_eq!(st, StatusCode::OK);
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["experts_requested"], true);
    }

    #[tokio::test]
    async fn chat_completion_still_works() {
        let (_, r) = app();
        let req = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"messages":[{"role":"user","content":"hi"}]}"#))
            .unwrap();
        let resp = r.oneshot(req).await.unwrap();
        let body = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "hello");
    }
}
