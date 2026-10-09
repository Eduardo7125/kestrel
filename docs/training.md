# Teaching the model: knowledge, instructions, fine-tuning

There are three ways to make a local model answer the way you need. They
change different things, and only one of them needs training.

| | What it changes | Use it for | Cost | In Kestrel |
|---|---|---|---|---|
| **Knowledge** (retrieval) | What the model reads before answering | Your notes, documents and facts, especially ones that change | Instant; updates when you edit a note | Built in: the dashboard's **Knowledge** page, `kestrel knowledge` |
| **Custom instructions** | Role, tone and rules for every answer | "Answer in Spanish", "be brief", "you are my study assistant" | Instant | Built in: **Training → Custom instructions** (the chat's system prompt) |
| **Fine-tuning** (LoRA) | The model's weights | A consistent style or output format, a narrow task done many times | A GPU, hours of compute, and good examples | Build the dataset in the dashboard, train with an external tool, run the result in Kestrel |

**If you want the model to know your notes, use Knowledge, not
fine-tuning.** Fine-tuning on a pile of notes teaches the model to imitate
them. It does not reliably memorize them, it cannot tell you where an answer
came from, and it has to be redone every time the notes change. Retrieval
reads the current notes for every question and cites them.

## Knowledge: your Obsidian vault and files

Connect a vault in the dashboard (**Knowledge → Connect a vault or folder**,
then paste its path) or from a terminal:

```bash
kestrel knowledge add ~/Documents/MyVault     # an Obsidian vault or any folder
kestrel knowledge list                        # sources, files, passages
kestrel knowledge search "when is Ana's birthday"
```

What happens on every chat message:

1. Kestrel searches the connected sources for the message's words (BM25,
   accents and plurals folded, English and Spanish stop words removed).
   Notes are split into passages along their headings. A note's title,
   headings, tags and aliases count double.
2. The best passages (4 by default, at most 2 per note, within a size
   budget) are added to the system message, with an instruction to cite
   notes as `[[Title]]` and to say when the notes do not contain the answer.
3. The answer lists the notes it was given. For an Obsidian vault each one
   links to `obsidian://open?...`, which opens the note in Obsidian.

Obsidian specifics:
- **Understood:** wiki-links (`[[Note|alias]]` becomes "alias"), front matter
  tags and aliases, and headings.
- **Skipped:** embeds (`![[...]]`), comments (`%% ... %%`), and the
  `.obsidian` and `.trash` folders.
- **Edits:** changes are picked up within 20 seconds, without a manual sync.

Limits of this version:
- **Lexical search.** It matches words, not meaning. A question phrased
  with different words than the note ("salary" vs "pay") can miss it.
  Semantic search with an embedding model is on the roadmap.
- **Text files only:** Markdown, text, CSV, JSON and code. Export PDF and
  Word documents to text or Markdown first.
- **Context use.** Retrieved passages use part of the model's context
  (3000 characters, about 750 tokens, by default).

API clients get the same behaviour. A request can opt out with
`"kestrel": {"knowledge": false}`. Responses carry the passages used in
`kestrel_sources`: in the first chunk when streaming, at the top level
otherwise.

## Custom instructions

**Training → Custom instructions** is the system prompt sent with every chat
message from the dashboard. Use it for durable rules:
- "Answer in Spanish."
- "When you use my notes, say which note."
- "Explain like a tutor and end with one practice question."

## Fine-tuning with LoRA

Fine-tuning is worth it when instructions are not enough. Typical cases: a
fixed output format, a house style, or a narrow task done thousands of
times. It is not the way to add knowledge.

### 1. Build the dataset in the dashboard

- In the chat, press **Save as example** under an answer you would like
  the model to give every time. In **Training → Fine-tuning dataset**,
  edit it until it is exactly right, or add examples by hand.
- Aim for 50 to 500 consistent, high-quality examples. A few hundred good
  examples beat thousands of mediocre ones.
- **Export JSONL** writes one conversation per line in the OpenAI chat
  format, with your custom instructions as the system message:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

The dataset lives in the browser's local storage. Export it to keep a copy.

### 2. Train a LoRA adapter

Kestrel runs models; it does not train them. Use a training tool such as
[Unsloth](https://github.com/unslothai/unsloth) (free, single GPU), or
[Axolotl](https://github.com/axolotl-ai-cloud/axolotl).

What fits:
- **8 GB GPU** (for example an RTX 4060 Laptop): 4-bit QLoRA on models up
  to about 7–8B parameters, such as Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct.
- **30B and larger** (including Qwen3-30B-A3B): needs much more GPU memory.
  Rent a GPU, or fine-tune a smaller model.
- **No GPU:** CPU-only training is impractically slow for these sizes.

With Unsloth the steps are:
1. Load the base model in 4-bit.
2. Attach a LoRA adapter (rank 8–32 is typical).
3. Train with the exported JSONL for one to three epochs.

Unsloth publishes notebooks for each model family. Start from the one for
your base model, because the API changes between versions.

### 3. Export to GGUF and run it in Kestrel

Merge the adapter into the base model and export a GGUF with a `Q4_K_M`
quantization; Unsloth does both in one step (`save_pretrained_gguf`). Then:

```bash
kestrel setup --model-file ./my-model-Q4_K_M.gguf
kestrel web
```

Kestrel's native executor runs llama, qwen2, qwen3, qwen2moe and qwen3moe
models with Q4_K_M and the other K-quants it supports (`kestrel inspect`
says whether it can run a file). Loading LoRA adapters without merging
them is not supported yet.
