# pi-local-council

**An LLM Council for the [pi coding agent](https://github.com/badlogic/pi-mono) — built for local models. $0 per verdict.**

Andrej Karpathy's [llm-council](https://github.com/karpathy/llm-council) sends your question to a
panel of frontier models over OpenRouter, has them peer-review each other's answers anonymously,
and lets a chairman synthesize the final verdict.

This extension puts that exact mechanism **inside your coding agent**, running entirely on
**your own hardware** — llama.cpp, Ollama, MLX, or anything else in your pi `models.json`.
Your agent can convene the council on its own before a big decision, or you can convene it
with `/council`.

```
Stage 1  every member answers independently        (parallel, isolated pi processes)
Stage 2  members rank the ANONYMIZED answers        (no model can play favorites)
Stage 3  the chairman synthesizes the verdict       (weighs rankings, may overrule)
```

The output is a verdict, a peer-review leaderboard (Borda count), and a full markdown
transcript receipt in `.pi/council/`.

## Why local

- **$0 per question.** A full council round is ~7 inference calls. On APIs that adds up; on
  your own machine it's free and private.
- **Different model families actually disagree.** A council of one vendor's models is an echo
  chamber. Locally you can seat Qwen next to Gemma next to GLM.
- **Your code never leaves the machine.** Pass diffs, plans, and error output as context
  without thinking twice.

## Install

```bash
pi install git:github.com/ravsau/pi-local-council
```

Then generate a config from the local providers pi already knows about:

```
/council init
```

This writes `~/.pi/agent/council.json`, auto-picking up to three **distinct model families**
from your `models.json`. Edit to taste:

```json
{
  "members": [
    { "name": "qwen",  "model": "llamacpp-qwen36-mtp/qwen3.6-27b-q6k-mtp" },
    { "name": "gemma", "model": "ollama/gemma4:26b-mlx" },
    { "name": "glm",   "model": "ollama/glm-4.7-flash:latest" }
  ],
  "chairman": { "name": "qwen", "model": "llamacpp-qwen36-mtp/qwen3.6-27b-q6k-mtp" },
  "concurrency": 2,
  "memberTools": [],
  "timeoutMs": 900000,
  "saveTranscripts": true
}
```

A project-local `.pi/council.json` overrides the user config, so each repo can seat its own
council.

## Use

Ask your agent naturally — the council is a tool it can decide to call:

```
Before you implement this, convene the council on whether SQLite or Postgres
fits this project better.
```

Or convene it yourself:

```
/council Should I use a single 27B model or a team of smaller models for agentic coding?
```

You get back:

```
## 🏛️ Council Verdict (chairman: qwen)
...synthesized answer, agreements, disagreements, who the chairman sided with...

### Peer-review leaderboard (Borda count, anonymized votes)
| Rank | Member | Model | Points | Answer time |
|---|---|---|---|---|
| 1 | gemma | ollama/gemma4:26b-mlx | 4 | 38.2s |
| 2 | qwen  | llamacpp-qwen36-mtp/qwen3.6-27b-q6k-mtp | 3 | 51.7s |
| 3 | glm   | ollama/glm-4.7-flash:latest | 1 | 29.9s |

📜 Full transcript: .pi/council/council-2026-07-14T18-30-00.md
```

## Config reference

| Key | Default | What it does |
|---|---|---|
| `members` | — | 2-10 models. Use different families so the council actually disagrees. |
| `chairman` | — | The synthesizer. Seat your strongest local model. |
| `concurrency` | `2` | Parallel member processes. Raise if you have RAM headroom, lower if you're swapping. |
| `memberTools` | `[]` | Read-only tools members may use to inspect the repo (e.g. `["read","grep","find","ls"]`). Empty = pure opinion, much faster. |
| `timeoutMs` | `900000` | Per-call timeout. Local models are slow; be generous. |
| `saveTranscripts` | `true` | Write full receipts to `.pi/council/`. |
| `maxAnswerWords` | `400` | Word budget hint for stage-1 answers. |

## Memory math (the part that matters on one machine)

Every member is an ephemeral `pi -p` process hitting the model server you configured.
Model **weights** are what you budget for, and `concurrency` is your throttle:

- Members served by **Ollama** load/unload on demand; N distinct models resident means the
  sum of their weights in RAM.
- Members served by a persistent **llama-server** stay resident regardless.
- Example that fits comfortably in 128GB of unified memory: 27B Q6 (~23GB) + 26B (~17GB) +
  19B (~19GB) ≈ 59GB of weights plus KV — with room to spare.

If the machine starts swapping, drop `concurrency` to 1 — the council becomes sequential
and peak memory is a single member at a time.

## Prior art & credit

- [karpathy/llm-council](https://github.com/karpathy/llm-council) — the mechanism (independent
  answers → anonymized peer review → chairman). This extension is that idea, local-first and
  in-harness.
- [sshkeda/pi-council](https://github.com/sshkeda/pi-council) — a pi council of **cloud**
  frontier models (Claude/GPT/Gemini/Grok) with background RPC members. Different design goals:
  it treats differing opinions as the product; this project runs the full review-and-synthesize
  protocol on local models.
- [Mario Zechner's pi](https://mariozechner.at/posts/2025-11-30-pi-coding-agent/) — the minimal
  harness whose extension system makes this a single TypeScript file.

## License

MIT
