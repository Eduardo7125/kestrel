#!/usr/bin/env sh
# Kestrel: from a fresh checkout to a running model in one step.
#
#   ./start-here.sh            pick a model that fits this machine, download it, chat
#   ./start-here.sh --serve    the same, then serve the OpenAI-compatible API
#   ./start-here.sh --list     show every model against this machine
#
# Run it again any time: it rebuilds only what changed, resumes an
# interrupted download, and once a model is set up it starts straight away.
# Every option is passed to `kestrel setup` (see `kestrel setup --help`).
set -e
cd "$(dirname "$0")"

yes=0
for a in "$@"; do
  case "$a" in -y|--yes) yes=1 ;; esac
done

ask() { # ask "question" → 0 for yes
  if [ "$yes" = 1 ]; then return 0; fi
  if [ ! -t 0 ]; then return 1; fi
  printf '%s [Y/n] ' "$1"
  read -r ans
  case "$ans" in n|N|no|NO) return 1 ;; *) return 0 ;; esac
}

[ -x "$HOME/.cargo/bin/cargo" ] && PATH="$HOME/.cargo/bin:$PATH"

if ! command -v cargo >/dev/null 2>&1; then
  echo "Kestrel is written in Rust and is built on this machine once."
  echo "The Rust toolchain is not installed."
  if ask "Install it now with rustup (https://rustup.rs, into ~/.cargo)?"; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
    PATH="$HOME/.cargo/bin:$PATH"
  else
    echo "Install Rust from https://rustup.rs, then run ./start-here.sh again."
    exit 1
  fi
fi

if ! command -v cc >/dev/null 2>&1; then
  echo "A C compiler is needed to build Kestrel's dependencies. Install it, then run ./start-here.sh again:"
  case "$(uname -s)" in
    Darwin) echo "  xcode-select --install" ;;
    *) echo "  sudo apt install build-essential     # Debian, Ubuntu"
       echo "  sudo dnf install gcc                  # Fedora"
       echo "  sudo pacman -S base-devel             # Arch" ;;
  esac
  exit 1
fi

if [ ! -x target/release/kestrel ]; then
  echo "Building Kestrel (the first build takes a few minutes)…"
fi
cargo build --release --quiet -p kestrel-cli
exec ./target/release/kestrel setup "$@"
