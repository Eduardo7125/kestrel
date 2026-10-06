#!/usr/bin/env sh
# Build the llama.cpp oracle against a llama.cpp checkout+build.
#   LLAMA_CPP=/path/to/llama.cpp tools/oracle/build.sh
set -e
: "${LLAMA_CPP:?set LLAMA_CPP to a llama.cpp checkout built with cmake (build/bin)}"
out="${1:-$(dirname "$0")/llama_oracle}"
c++ -O2 -std=c++17 "$(dirname "$0")/llama_oracle.cpp" -o "$out" \
  -I"$LLAMA_CPP/include" -I"$LLAMA_CPP/ggml/include" \
  -L"$LLAMA_CPP/build/bin" -lllama -lggml -lggml-base \
  -Wl,-rpath,"$LLAMA_CPP/build/bin"
echo "built $out"
