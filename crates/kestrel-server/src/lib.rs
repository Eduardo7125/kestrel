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
//! With a [`Control`] (the CLI provides one), the dashboard can also manage
//! the model: preview a plan for new settings, reload with them, unload to
//! free memory, run the autotuner, and shut the server down.
//!
//! ```text
//! GET  /api/control           state: loaded / loading / unloaded / tuning / error, settings, tune log
//! POST /api/plan              plan for the posted settings, without loading
//! POST /api/apply             reload the model with the posted settings
//! POST /api/unload            free the model's memory (the server keeps running)
//! POST /api/load              load it again with the current settings
//! POST /api/tune              unload, run the autotuner, reload with its result
//! POST /api/shutdown          stop the server
//! ```
//!
//! Control requests must carry `Content-Type: application/json` and, when
//! they come from a browser, an `Origin` matching the server: another site
//! cannot unload or stop a local Kestrel.
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
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use axum::http::HeaderMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

/// The loaded model, or `None` while it is unloaded (memory freed).
pub type SharedSession = Arc<Mutex<Option<Box<dyn InferenceSession>>>>;

pub fn shared(session: Box<dyn InferenceSession>) -> SharedSession {
    Arc::new(Mutex::new(Some(session)))
}

/// Load a session with the given dashboard settings.
pub type LoadFn = Arc<dyn Fn(&Value) -> anyhow::Result<Box<dyn InferenceSession>> + Send + Sync>;
/// Plan for the given settings without loading anything.
pub type PlanFn = Arc<dyn Fn(&Value) -> anyhow::Result<Value> + Send + Sync>;
/// Run the autotuner, reporting progress lines; returns its conclusion.
pub type TuneFn = Arc<dyn Fn(&mut dyn FnMut(String)) -> anyhow::Result<String> + Send + Sync>;

/// How the dashboard manages the model. Settings are a JSON object whose
/// meaning belongs to the provider (the CLI maps it onto its options).
pub struct Control {
    pub load: LoadFn,
    pub plan: PlanFn,
    pub tune: TuneFn,
    /// Settings the current session was loaded with.
    pub settings: Value,
}

#[derive(Clone, Serialize)]
struct ControlState {
    /// loaded · loading · unloading · unloaded · tuning · error
    phase: &'static str,
    message: Option<String>,
    settings: Value,
    tune_running: bool,
    tune_log: Vec<String>,
    tune_result: Option<String>,
}

/// What the dashboard reads from the current session.
struct Loaded {
    monitor: Option<Monitor>,
    info: Arc<Value>,
}

#[derive(Clone)]
struct AppState {
    session: SharedSession,
    model_id: String,
    default_max_tokens: usize,
    loaded: Arc<RwLock<Loaded>>,
    extra_info: Arc<Value>,
    control: Option<Arc<Control>>,
    ctl: Arc<Mutex<ControlState>>,
    shutdown: Arc<tokio::sync::Notify>,
}

pub struct ServeOptions {
    pub default_max_tokens: usize,
    /// Merged into `/api/info` (the CLI adds the hardware profile).
    pub extra_info: Value,
    /// Lets the dashboard reload, unload, tune and stop the model.
    pub control: Option<Control>,
}

fn describe(sess: &dyn InferenceSession, extra: &Value) -> Loaded {
    let mut info = sess.info();
    if let (Some(dst), Some(src)) = (info.as_object_mut(), extra.as_object()) {
        for (k, v) in src {
            dst.insert(k.clone(), v.clone());
        }
    }
    info["server"] = json!({"version": env!("CARGO_PKG_VERSION"), "model_id": sess.model_name()});
    Loaded { monitor: sess.monitor(), info: Arc::new(info) }
}

const DASHBOARD: &str = include_str!("../web/index.html");
const LOGO: &[u8] = include_bytes!("../../../assets/logo.png");

pub fn router(session: SharedSession, opts: ServeOptions) -> Router {
    build(session, opts).0
}

fn build(session: SharedSession, opts: ServeOptions) -> (Router, Arc<tokio::sync::Notify>) {
    let (model_id, loaded) = {
        let s = session.lock().unwrap();
        match s.as_deref() {
            Some(sess) => (sess.model_name().to_string(), describe(sess, &opts.extra_info)),
            None => ("local".to_string(), Loaded { monitor: None, info: Arc::new(opts.extra_info.clone()) }),
        }
    };
    let settings = opts.control.as_ref().map(|c| c.settings.clone()).unwrap_or(Value::Null);
    let shutdown = Arc::new(tokio::sync::Notify::new());
    let state = AppState {
        session,
        model_id,
        default_max_tokens: opts.default_max_tokens,
        loaded: Arc::new(RwLock::new(loaded)),
        extra_info: Arc::new(opts.extra_info),
        control: opts.control.map(Arc::new),
        ctl: Arc::new(Mutex::new(ControlState { phase: "loaded", message: None, settings, tune_running: false, tune_log: Vec::new(), tune_result: None })),
        shutdown: shutdown.clone(),
    };
    let r = Router::new()
        .route("/", get(|| async { ([("content-type", "text/html; charset=utf-8"), ("cache-control", "no-cache")], DASHBOARD) }))
        .route("/logo.png", get(|| async { ([("content-type", "image/png"), ("cache-control", "max-age=86400")], LOGO) }))
        .route("/api/info", get(api_info))
        .route("/api/live", get(api_live))
        .route("/api/control", get(api_control))
        .route("/api/plan", post(api_plan))
        .route("/api/apply", post(api_apply))
        .route("/api/unload", post(api_unload))
        .route("/api/load", post(api_load))
        .route("/api/tune", post(api_tune))
        .route("/api/shutdown", post(api_shutdown))
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
        .route("/metrics", get(metrics))
        .with_state(state);
    (r, shutdown)
}

/// Serve until Ctrl+C or `POST /api/shutdown`.
pub async fn serve(session: SharedSession, addr: &str, opts: ServeOptions) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let (app, stop) = build(session, opts);
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = stop.notified() => {}
            }
        })
        .await?;
    Ok(())
}

// ---------------------------------------------------------------- model control

/// Reject control requests a browser sends on behalf of another site.
// Handlers return the rejection itself; it is built once per request.
#[allow(clippy::result_large_err)]
fn check_origin(h: &HeaderMap) -> Result<(), Response> {
    let json = h.get("content-type").and_then(|v| v.to_str().ok()).is_some_and(|v| v.starts_with("application/json"));
    if !json {
        return Err(err(StatusCode::UNSUPPORTED_MEDIA_TYPE, "control requests must be application/json"));
    }
    if let Some(origin) = h.get("origin").and_then(|v| v.to_str().ok()) {
        let host = h.get("host").and_then(|v| v.to_str().ok()).unwrap_or("");
        let origin_host = origin.split("://").nth(1).unwrap_or(origin);
        if origin_host != host {
            return Err(err(StatusCode::FORBIDDEN, "cross-origin control request refused"));
        }
    }
    Ok(())
}

// Handlers return the rejection itself; it is built once per request.
#[allow(clippy::result_large_err)]
fn control_of(s: &AppState) -> Result<Arc<Control>, Response> {
    s.control.clone().ok_or_else(|| err(StatusCode::NOT_IMPLEMENTED, "this server was started without model control"))
}

/// Mark the start of a long operation, or refuse if one is running.
// Handlers return the rejection itself; it is built once per request.
#[allow(clippy::result_large_err)]
fn begin(s: &AppState, phase: &'static str) -> Result<(), Response> {
    let mut c = s.ctl.lock().unwrap();
    if matches!(c.phase, "loading" | "unloading" | "tuning") {
        return Err(err(StatusCode::CONFLICT, format!("busy: {}", c.phase)));
    }
    c.phase = phase;
    c.message = None;
    Ok(())
}

fn set_phase(s: &AppState, phase: &'static str, message: Option<String>) {
    let mut c = s.ctl.lock().unwrap();
    c.phase = phase;
    c.message = message;
}

/// Drop the session (waits for a running generation) so its memory is freed.
fn unload_blocking(s: &AppState) {
    set_phase(s, "unloading", None);
    // Never hold the read guard while taking the write guard.
    let info = s.loaded.read().unwrap().info.clone();
    *s.loaded.write().unwrap() = Loaded { monitor: None, info };
    let old = s.session.lock().unwrap().take();
    drop(old);
}

fn load_blocking(s: &AppState, ctl: &Control, settings: Value) {
    set_phase(s, "loading", None);
    match (ctl.load)(&settings) {
        Ok(sess) => {
            *s.loaded.write().unwrap() = describe(sess.as_ref(), &s.extra_info);
            *s.session.lock().unwrap() = Some(sess);
            let mut c = s.ctl.lock().unwrap();
            c.phase = "loaded";
            c.message = None;
            c.settings = settings;
        }
        Err(e) => set_phase(s, "error", Some(format!("{e:#}"))),
    }
}

fn accepted(s: &AppState) -> Response {
    (StatusCode::ACCEPTED, Json(serde_json::to_value(s.ctl.lock().unwrap().clone()).unwrap_or_default())).into_response()
}

async fn api_control(State(s): State<AppState>) -> Json<Value> {
    let mut v = serde_json::to_value(s.ctl.lock().unwrap().clone()).unwrap_or_default();
    v["controllable"] = json!(s.control.is_some());
    v["busy"] = json!(s.session.try_lock().is_err());
    Json(v)
}

async fn api_plan(State(s): State<AppState>, h: HeaderMap, Json(settings): Json<Value>) -> Response {
    if let Err(r) = check_origin(&h) {
        return r;
    }
    let ctl = match control_of(&s) {
        Ok(c) => c,
        Err(r) => return r,
    };
    match tokio::task::spawn_blocking(move || (ctl.plan)(&settings)).await {
        Ok(Ok(v)) => Json(v).into_response(),
        Ok(Err(e)) => err(StatusCode::UNPROCESSABLE_ENTITY, format!("{e:#}")),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

async fn api_apply(State(s): State<AppState>, h: HeaderMap, Json(settings): Json<Value>) -> Response {
    if let Err(r) = check_origin(&h) {
        return r;
    }
    let ctl = match control_of(&s) {
        Ok(c) => c,
        Err(r) => return r,
    };
    if let Err(r) = begin(&s, "loading") {
        return r;
    }
    let st = s.clone();
    tokio::task::spawn_blocking(move || {
        unload_blocking(&st);
        load_blocking(&st, &ctl, settings);
    });
    accepted(&s)
}

async fn api_unload(State(s): State<AppState>, h: HeaderMap) -> Response {
    if let Err(r) = check_origin(&h) {
        return r;
    }
    if let Err(r) = control_of(&s) {
        return r;
    }
    if let Err(r) = begin(&s, "unloading") {
        return r;
    }
    let st = s.clone();
    let _ = tokio::task::spawn_blocking(move || {
        unload_blocking(&st);
        set_phase(&st, "unloaded", None);
    })
    .await;
    accepted(&s)
}

async fn api_load(State(s): State<AppState>, h: HeaderMap) -> Response {
    if let Err(r) = check_origin(&h) {
        return r;
    }
    let ctl = match control_of(&s) {
        Ok(c) => c,
        Err(r) => return r,
    };
    if s.session.lock().unwrap().is_some() {
        return accepted(&s);
    }
    if let Err(r) = begin(&s, "loading") {
        return r;
    }
    let settings = s.ctl.lock().unwrap().settings.clone();
    let st = s.clone();
    tokio::task::spawn_blocking(move || load_blocking(&st, &ctl, settings));
    accepted(&s)
}

async fn api_tune(State(s): State<AppState>, h: HeaderMap) -> Response {
    if let Err(r) = check_origin(&h) {
        return r;
    }
    let ctl = match control_of(&s) {
        Ok(c) => c,
        Err(r) => return r,
    };
    if let Err(r) = begin(&s, "tuning") {
        return r;
    }
    {
        let mut c = s.ctl.lock().unwrap();
        c.tune_running = true;
        c.tune_log.clear();
        c.tune_result = None;
    }
    let st = s.clone();
    tokio::task::spawn_blocking(move || {
        // The trials load the model themselves: free this copy first.
        unload_blocking(&st);
        set_phase(&st, "tuning", None);
        let log = st.ctl.clone();
        let res = (ctl.tune)(&mut |line| {
            let mut c = log.lock().unwrap();
            c.tune_log.push(line);
            let n = c.tune_log.len();
            if n > 500 {
                c.tune_log.drain(..n - 500);
            }
        });
        let mut settings = {
            let mut c = st.ctl.lock().unwrap();
            c.tune_running = false;
            c.tune_result = Some(match &res {
                Ok(r) => r.clone(),
                Err(e) => format!("autotune failed: {e:#}"),
            });
            c.settings.clone()
        };
        // Reload, letting the new tuning profile fill the knobs left on auto.
        if let Some(o) = settings.as_object_mut() {
            o.insert("tune_profile".into(), json!(true));
        }
        load_blocking(&st, &ctl, settings);
    });
    accepted(&s)
}

async fn api_shutdown(State(s): State<AppState>, h: HeaderMap) -> Response {
    if let Err(r) = check_origin(&h) {
        return r;
    }
    set_phase(&s, "stopping", None);
    let stop = s.shutdown.clone();
    tokio::spawn(async move {
        // Let this response go out, then stop; force the exit if an open
        // stream keeps the graceful shutdown waiting.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        stop.notify_one();
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        std::process::exit(0);
    });
    Json(json!({"phase": "stopping"})).into_response()
}

fn not_loaded(s: &AppState) -> Response {
    let c = s.ctl.lock().unwrap();
    let msg = match c.phase {
        "loading" => "the model is loading; try again in a moment".to_string(),
        "tuning" => "the model is unloaded while the autotuner runs".to_string(),
        "error" => format!("the model failed to load: {}", c.message.clone().unwrap_or_default()),
        _ => "the model is unloaded; load it from the dashboard's Settings page".to_string(),
    };
    err(StatusCode::SERVICE_UNAVAILABLE, msg)
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
    let session = s.session.clone();
    let prompt = tokio::task::spawn_blocking(move || {
        let guard = session.lock().unwrap();
        guard.as_deref().map(|sess| sess.render_chat(&messages).map(|p| sess.tokenize(&p)).map_err(|e| e.to_string()))
    })
    .await;
    let prompt = match prompt {
        Ok(Some(Ok(p))) => p,
        Ok(Some(Err(e))) => return err(StatusCode::BAD_REQUEST, e),
        Ok(None) => return not_loaded(&s),
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    run(s, prompt, req.common, true).await
}

async fn completions(State(s): State<AppState>, Json(req): Json<CompletionReq>) -> Response {
    let text = match &req.prompt {
        PromptIn::One(p) => p.clone(),
        PromptIn::Many(v) if v.len() == 1 => v[0].clone(),
        PromptIn::Many(_) => return err(StatusCode::BAD_REQUEST, "batched prompts are not supported"),
    };
    // Tokenize off the async workers: the session lock is held for the
    // whole of a running generation.
    let session = s.session.clone();
    let prompt = match tokio::task::spawn_blocking(move || session.lock().unwrap().as_deref().map(|sess| sess.tokenize(&text))).await {
        Ok(Some(p)) => p,
        Ok(None) => return not_loaded(&s),
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
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
            let mut guard = session.lock().unwrap();
            let Some(sess) = guard.as_mut() else { return Err(anyhow::anyhow!("the model was unloaded")) };
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
        let mut guard = session.lock().unwrap();
        let res = match guard.as_mut() {
            Some(sess) => sess.generate(&prompt, &params, &mut |piece| tx.send(chunk(json!({"content": piece}), Value::Null)).is_ok()),
            None => Err(anyhow::anyhow!("the model was unloaded")),
        };
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
    let mut v = (*s.loaded.read().unwrap().info).clone();
    v["phase"] = json!(s.ctl.lock().unwrap().phase);
    Json(v)
}

async fn api_live(State(s): State<AppState>, Query(q): Query<HashMap<String, String>>) -> Response {
    let with_experts = q.get("experts").is_some_and(|v| v == "1" || v == "true");
    let phase = s.ctl.lock().unwrap().phase;
    let monitor = s.loaded.read().unwrap().monitor.clone();
    match monitor {
        Some(m) => match tokio::task::spawn_blocking(move || m(with_experts)).await {
            Ok(mut v) => {
                v["phase"] = json!(phase);
                Json(v).into_response()
            }
            Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        },
        None => Json(json!({"busy": false, "phase": phase, "rss": kestrel_hw::process_rss()})).into_response(),
    }
}

async fn metrics(State(s): State<AppState>, Query(q): Query<HashMap<String, String>>) -> Response {
    let mut status = match s.session.try_lock() {
        Ok(guard) => guard.as_deref().map(|sess| sess.status()).unwrap_or_else(|| json!({"loaded": false})),
        Err(_) => json!({"busy": true}),
    };
    // While a request generates, the session is locked: take memory and
    // store figures from the lock-free monitor instead.
    let monitor = s.loaded.read().unwrap().monitor.clone();
    if let (Some(m), Some(obj)) = (&monitor, status.as_object_mut()) {
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
        let s: SharedSession = shared(Box::new(Mock));
        let r = router(s.clone(), ServeOptions { default_max_tokens: 16, extra_info: json!({"hardware": {"cores": 8}}), control: None });
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

    async fn post(r: &Router, uri: &str, body: &str) -> (StatusCode, Value) {
        let req = Request::post(uri).header("content-type", "application/json").body(Body::from(body.to_string())).unwrap();
        let resp = r.clone().oneshot(req).await.unwrap();
        let st = resp.status();
        let b = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        (st, serde_json::from_slice(&b).unwrap_or(Value::Null))
    }

    async fn wait_phase(r: &Router, phase: &str) -> Value {
        for _ in 0..200 {
            let (_, _, b) = get(r, "/api/control").await;
            let v: Value = serde_json::from_str(&b).unwrap();
            if v["phase"] == phase {
                return v;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!("never reached {phase}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dashboard_controls_unload_load_tune() {
        let loads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let l2 = loads.clone();
        let control = Control {
            load: Arc::new(move |v: &Value| {
                if v["fail"] == true {
                    anyhow::bail!("no room");
                }
                l2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(Box::new(Mock) as Box<dyn InferenceSession>)
            }),
            plan: Arc::new(|v: &Value| Ok(json!({"threads": v["threads"], "tok_s": 5.0}))),
            tune: Arc::new(|on_line: &mut dyn FnMut(String)| {
                on_line("trial 1".into());
                on_line("trial 2".into());
                Ok("tuned: threads 8".into())
            }),
            settings: json!({"threads": null}),
        };
        let s = shared(Box::new(Mock));
        let r = router(s.clone(), ServeOptions { default_max_tokens: 16, extra_info: json!({}), control: Some(control) });

        let (st, v) = post(&r, "/api/plan", r#"{"threads": 4}"#).await;
        assert_eq!((st, v["threads"].as_u64()), (StatusCode::OK, Some(4)));

        let (st, _) = post(&r, "/api/unload", "{}").await;
        assert_eq!(st, StatusCode::ACCEPTED);
        wait_phase(&r, "unloaded").await;
        assert!(s.lock().unwrap().is_none(), "the session was dropped");
        // The rest of the server keeps answering while unloaded.
        assert_eq!(get(&r, "/health").await.0, StatusCode::OK);
        assert_eq!(get(&r, "/api/live").await.0, StatusCode::OK);
        let (st, v) = post(&r, "/v1/chat/completions", r#"{"messages":[{"role":"user","content":"hi"}]}"#).await;
        assert_eq!(st, StatusCode::SERVICE_UNAVAILABLE);
        assert!(v["error"]["message"].as_str().unwrap().contains("unloaded"));

        post(&r, "/api/load", "{}").await;
        wait_phase(&r, "loaded").await;
        assert_eq!(get(&r, "/api/info").await.0, StatusCode::OK);

        post(&r, "/api/apply", r#"{"threads": 8}"#).await;
        let v = wait_phase(&r, "loaded").await;
        assert_eq!(v["settings"]["threads"], 8);

        // A failed load is reported, and the server stays usable.
        post(&r, "/api/apply", r#"{"fail": true}"#).await;
        let v = wait_phase(&r, "error").await;
        assert!(v["message"].as_str().unwrap().contains("no room"));
        assert_eq!(get(&r, "/health").await.0, StatusCode::OK);

        post(&r, "/api/tune", "{}").await;
        let v = loop {
            let v = wait_phase(&r, "loaded").await;
            if v["tune_result"].is_string() {
                break v;
            }
        };
        assert_eq!(v["tune_result"], "tuned: threads 8");
        assert_eq!(v["tune_log"].as_array().unwrap().len(), 2);
        assert_eq!(v["settings"]["tune_profile"], true);
        assert!(loads.load(std::sync::atomic::Ordering::SeqCst) >= 3);
    }

    #[tokio::test]
    async fn control_requires_same_origin_json() {
        let (_, r) = app();
        let req = Request::post("/api/unload").header("content-type", "application/json").header("origin", "http://evil.example").header("host", "127.0.0.1:8080").body(Body::from("{}")).unwrap();
        assert_eq!(r.clone().oneshot(req).await.unwrap().status(), StatusCode::FORBIDDEN);
        let req = Request::post("/api/shutdown").header("content-type", "text/plain").body(Body::from("x")).unwrap();
        assert_eq!(r.clone().oneshot(req).await.unwrap().status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
        // Without a Control, the dashboard cannot manage the model.
        let (st, _) = post(&r, "/api/unload", "{}").await;
        assert_eq!(st, StatusCode::NOT_IMPLEMENTED);
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
