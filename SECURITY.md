# Security policy

## Supported versions

Kestrel is in alpha. Security fixes are made on the `main` branch and are
included in the next release. Older versions are not patched.

## Reporting a vulnerability

Please **do not open a public issue** for security problems.

Report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/eduardo7125/kestrel/security/advisories/new).
Include the affected version or commit, the platform, steps to reproduce
(a crafted model file, a request, a command) and the impact you observed.

You can expect an acknowledgement within 7 days and an assessment within
30 days. We will coordinate a disclosure date with you and credit you in the
advisory unless you prefer otherwise.

## Scope and threat model

In scope:

- **Untrusted model files.** The GGUF parser and the prepared-container
  reader must reject malformed input without memory-unsafe behaviour or
  unbounded allocation.
- **The HTTP API** (`kestrel serve`): request handling, streaming, metrics.
- **Memory safety** in the I/O engine, the weight and expert stores, and the
  native kernels (`unsafe` code).

Out of scope:

- Network exposure of `kestrel serve` without an authenticating proxy. The
  server binds to `127.0.0.1` by default and has no authentication by design.
- The content a model generates.
- Vulnerabilities in llama.cpp or other external binaries that Kestrel
  launches; report those to their projects.

## Hardening notes for deployments

- Keep `kestrel serve` on localhost, or put it behind a reverse proxy that
  terminates TLS and authenticates clients.
- Run it as an unprivileged user with read-only access to the model
  directory.
- Only load model files from sources you trust: a malicious chat template or
  tokenizer can still produce misleading output even when parsing is safe.
