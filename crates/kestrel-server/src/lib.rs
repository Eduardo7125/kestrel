//! OpenAI-compatible HTTP API.
//!
//! ```text
//! POST /v1/chat/completions   (stream: true → server-sent events)
//! POST /v1/completions
//! GET  /v1/models
//! GET  /health
//! GET  /metrics               (Prometheus text; ?format=json for JSON)
//! ```
//!
//! Generation runs on a blocking thread; requests are served one at a time
//! per model (the MVP does not batch concurrent requests).

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use kestrel_backends::InferenceSession;
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
}

pub fn router(session: SharedSession, default_max_tokens: usize) -> Router {
    let model_id = session.lock().unwrap().model_name().to_string();
    let state = AppState { session, model_id, default_max_tokens };
    Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
        .route("/metrics", get(metrics))
        .with_state(state)
}

pub async fn serve(session: SharedSession, addr: &str, default_max_tokens: usize) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router(session, default_max_tokens)).with_graceful_shutdown(async {
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
    json!({"prompt_ms": st.prefill_s * 1e3, "predicted_ms": st.decode_s * 1e3, "predicted_per_second": st.decode_tok_s, "prompt_per_second": st.prompt_tok_s, "memory": st.memory})
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

async fn metrics(State(s): State<AppState>, Query(q): Query<HashMap<String, String>>) -> Response {
    let status = match s.session.try_lock() {
        Ok(sess) => sess.status(),
        Err(_) => json!({"busy": true}),
    };
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
