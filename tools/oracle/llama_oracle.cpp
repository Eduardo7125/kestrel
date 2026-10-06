// Reference oracle: run a GGUF model through llama.cpp and print, as JSON,
// the prompt tokens, the logits of the last prompt position, and a greedy
// continuation. Used by Kestrel's correctness tests:
//
//   llama_oracle <model.gguf> <prompt> <n_greedy> [parse_special=1]
//
// Build: see tools/oracle/build.sh
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model prompt n_greedy [parse_special]\n", argv[0]); return 2; }
    const char * path = argv[1];
    std::string prompt = argv[2];
    int n_greedy = atoi(argv[3]);
    bool special = argc < 5 || atoi(argv[4]) != 0;

    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 512; cp.n_threads = 4;
    llama_context * ctx = llama_init_from_model(model, cp);

    std::vector<llama_token> toks(prompt.size() + 16);
    int n = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(), toks.data(), (int)toks.size(), true, special);
    if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    toks.resize(n);

    if (llama_decode(ctx, llama_batch_get_one(toks.data(), n))) { fprintf(stderr, "decode failed\n"); return 1; }
    const int nv = llama_vocab_n_tokens(vocab);
    const float * lg = llama_get_logits_ith(ctx, -1);
    printf("{\"tokens\":[");
    for (int i = 0; i < n; i++) printf("%s%d", i ? "," : "", toks[i]);
    printf("],\"logits\":[");
    for (int i = 0; i < nv; i++) printf("%s%.6g", i ? "," : "", lg[i]);
    printf("],\"greedy\":[");
    for (int g = 0; g < n_greedy; g++) {
        const float * l = llama_get_logits_ith(ctx, -1);
        int best = 0;
        for (int i = 1; i < nv; i++) if (l[i] > l[best]) best = i;
        printf("%s%d", g ? "," : "", best);
        llama_token t = best;
        if (llama_decode(ctx, llama_batch_get_one(&t, 1))) break;
    }
    printf("]}\n");
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
