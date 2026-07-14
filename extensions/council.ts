/**
 * pi-local-council — an LLM Council for pi, built for local models.
 *
 * Implements Andrej Karpathy's llm-council mechanism (github.com/karpathy/llm-council)
 * inside the pi coding agent, using local models (llama.cpp, Ollama, MLX, or any
 * provider in your models.json):
 *
 *   Stage 1 — every council member answers the question independently (parallel)
 *   Stage 2 — each member reviews and ranks the ANONYMIZED answers (no favoritism)
 *   Stage 3 — the chairman synthesizes the final verdict from answers + reviews
 *
 * The main pi agent gets a `council` tool it can convene on its own (before a big
 * decision, on request, or whenever a second opinion is worth 7 local inference
 * calls). A `/council` command lets you convene it by hand.
 *
 * Config: .pi/council.json (project) or ~/.pi/agent/council.json (user).
 * Run `/council init` to auto-generate one from your local providers.
 */

import { spawn } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { Type } from "@mariozechner/pi-ai";
import { type ExtensionAPI, getAgentDir } from "@mariozechner/pi-coding-agent";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

interface CouncilMember {
	/** Display name, e.g. "qwen". Used in transcripts and the leaderboard. */
	name: string;
	/** pi model pattern, e.g. "ollama/gemma4:26b-mlx" or "llamacpp-qwen36-mtp/qwen3.6-27b-q6k-mtp". */
	model: string;
}

interface CouncilConfig {
	members: CouncilMember[];
	chairman: CouncilMember;
	/** How many member processes run at once. Tune to your RAM. Default: 2. */
	concurrency?: number;
	/** Read-only tools members may use to inspect the repo, e.g. ["read","grep","find","ls"]. Default: none (pure opinion). */
	memberTools?: string[];
	/** Per-call timeout in ms. Local models are slow; default 900000 (15 min). */
	timeoutMs?: number;
	/** Write a full transcript to .pi/council/<timestamp>.md. Default: true. */
	saveTranscripts?: boolean;
	/** Word budget hint for stage-1 answers. Default: 400. */
	maxAnswerWords?: number;
}

const CONFIG_BASENAME = "council.json";

function findProjectConfig(cwd: string): string | null {
	let dir = cwd;
	while (true) {
		const candidate = path.join(dir, ".pi", CONFIG_BASENAME);
		if (fs.existsSync(candidate)) return candidate;
		const parent = path.dirname(dir);
		if (parent === dir) return null;
		dir = parent;
	}
}

function userConfigPath(): string {
	return path.join(getAgentDir(), CONFIG_BASENAME);
}

function loadConfig(cwd: string): { config: CouncilConfig; source: string } | { error: string } {
	const projectPath = findProjectConfig(cwd);
	const configPath = projectPath ?? (fs.existsSync(userConfigPath()) ? userConfigPath() : null);
	if (!configPath) {
		return {
			error:
				`No council config found. Create ${userConfigPath()} (or <project>/.pi/${CONFIG_BASENAME}), ` +
				`or run /council init to generate one from your local providers.`,
		};
	}
	let raw: CouncilConfig;
	try {
		raw = JSON.parse(fs.readFileSync(configPath, "utf-8"));
	} catch (e) {
		return { error: `Failed to parse ${configPath}: ${e instanceof Error ? e.message : String(e)}` };
	}
	if (!Array.isArray(raw.members) || raw.members.length < 2) {
		return { error: `${configPath}: "members" must list at least 2 models. A council of one is just a model.` };
	}
	if (!raw.chairman?.model) {
		return { error: `${configPath}: "chairman.model" is required.` };
	}
	for (const m of raw.members) {
		if (!m.model) return { error: `${configPath}: every member needs a "model".` };
		if (!m.name) m.name = m.model.split("/").pop() ?? m.model;
	}
	if (!raw.chairman.name) raw.chairman.name = raw.chairman.model.split("/").pop() ?? raw.chairman.model;
	return { config: raw, source: configPath };
}

// ---------------------------------------------------------------------------
// One-shot pi runs (each council call is an isolated, ephemeral pi process)
// ---------------------------------------------------------------------------

interface OneShotResult {
	ok: boolean;
	text: string;
	error?: string;
	seconds: number;
	tokensIn: number;
	tokensOut: number;
}

function getPiInvocation(args: string[]): { command: string; args: string[] } {
	const currentScript = process.argv[1];
	if (currentScript && fs.existsSync(currentScript)) {
		return { command: process.execPath, args: [currentScript, ...args] };
	}
	return { command: "pi", args };
}

async function runPiOneShot(
	model: string,
	systemPrompt: string,
	prompt: string,
	tools: string[],
	timeoutMs: number,
	cwd: string,
	signal: AbortSignal | undefined,
): Promise<OneShotResult> {
	const started = Date.now();
	const result: OneShotResult = { ok: false, text: "", seconds: 0, tokensIn: 0, tokensOut: 0 };

	const args = ["--mode", "json", "-p", "--no-session", "--model", model];
	if (tools.length > 0) args.push("--tools", tools.join(","));
	else args.push("--no-tools");

	const tmpDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), "pi-council-"));
	const promptPath = path.join(tmpDir, "system.md");
	await fs.promises.writeFile(promptPath, systemPrompt, { encoding: "utf-8", mode: 0o600 });
	args.push("--append-system-prompt", promptPath);
	args.push(prompt);

	try {
		await new Promise<void>((resolve) => {
			const invocation = getPiInvocation(args);
			const proc = spawn(invocation.command, invocation.args, {
				cwd,
				shell: false,
				stdio: ["ignore", "pipe", "pipe"],
			});
			let buffer = "";
			let stderr = "";

			const processLine = (line: string) => {
				if (!line.trim()) return;
				let event: any;
				try {
					event = JSON.parse(line);
				} catch {
					return;
				}
				if (event.type === "message_end" && event.message?.role === "assistant") {
					const msg = event.message;
					for (const part of msg.content ?? []) {
						if (part.type === "text" && part.text) result.text = part.text;
					}
					if (msg.usage) {
						result.tokensIn += msg.usage.input || 0;
						result.tokensOut += msg.usage.output || 0;
					}
					if (msg.errorMessage) result.error = msg.errorMessage;
				}
			};

			const timer = setTimeout(() => {
				result.error = `timed out after ${Math.round(timeoutMs / 1000)}s`;
				proc.kill("SIGTERM");
				setTimeout(() => {
					if (!proc.killed) proc.kill("SIGKILL");
				}, 5000);
			}, timeoutMs);

			proc.stdout.on("data", (data) => {
				buffer += data.toString();
				const lines = buffer.split("\n");
				buffer = lines.pop() || "";
				for (const line of lines) processLine(line);
			});
			proc.stderr.on("data", (data) => {
				stderr += data.toString();
			});
			proc.on("close", (code) => {
				clearTimeout(timer);
				if (buffer.trim()) processLine(buffer);
				if (code !== 0 && !result.error) {
					result.error = stderr.trim().slice(-500) || `pi exited with code ${code}`;
				}
				resolve();
			});
			proc.on("error", (err) => {
				clearTimeout(timer);
				result.error = err.message;
				resolve();
			});

			if (signal) {
				const kill = () => {
					result.error = "aborted";
					proc.kill("SIGTERM");
				};
				if (signal.aborted) kill();
				else signal.addEventListener("abort", kill, { once: true });
			}
		});
	} finally {
		try {
			await fs.promises.rm(tmpDir, { recursive: true, force: true });
		} catch {
			/* ignore */
		}
	}

	result.seconds = Math.round((Date.now() - started) / 10) / 100;
	result.ok = !result.error && result.text.trim().length > 0;
	if (!result.ok && !result.error) result.error = "empty response";
	return result;
}

async function mapWithConcurrencyLimit<TIn, TOut>(
	items: TIn[],
	concurrency: number,
	fn: (item: TIn, index: number) => Promise<TOut>,
): Promise<TOut[]> {
	if (items.length === 0) return [];
	const limit = Math.max(1, Math.min(concurrency, items.length));
	const results: TOut[] = new Array(items.length);
	let nextIndex = 0;
	const workers = new Array(limit).fill(null).map(async () => {
		while (true) {
			const current = nextIndex++;
			if (current >= items.length) return;
			results[current] = await fn(items[current], current);
		}
	});
	await Promise.all(workers);
	return results;
}

// ---------------------------------------------------------------------------
// The council: 3 stages
// ---------------------------------------------------------------------------

interface MemberAnswer {
	member: CouncilMember;
	result: OneShotResult;
	/** Anonymized label ("A", "B", ...) assigned after shuffling. */
	label?: string;
}

interface MemberReview {
	member: CouncilMember;
	result: OneShotResult;
	ranking: string[]; // labels, best first
}

interface CouncilOutcome {
	question: string;
	answers: MemberAnswer[];
	reviews: MemberReview[];
	tally: { label: string; member: CouncilMember; points: number }[];
	chairman: { member: CouncilMember; result: OneShotResult };
	transcriptPath?: string;
	report: string;
}

const LABELS = "ABCDEFGHIJ".split("");

function parseRanking(text: string, validLabels: string[]): string[] {
	const matches = [...text.matchAll(/RANKING:\s*([A-J](?:\s*>\s*[A-J])*)/gi)];
	if (matches.length === 0) return [];
	const last = matches[matches.length - 1][1];
	const seen = new Set<string>();
	const ranking: string[] = [];
	for (const raw of last.split(">")) {
		const label = raw.trim().toUpperCase();
		if (validLabels.includes(label) && !seen.has(label)) {
			seen.add(label);
			ranking.push(label);
		}
	}
	return ranking;
}

type ProgressFn = (text: string) => void;

async function runCouncil(
	config: CouncilConfig,
	question: string,
	context: string | undefined,
	cwd: string,
	signal: AbortSignal | undefined,
	onProgress: ProgressFn,
): Promise<CouncilOutcome> {
	const concurrency = config.concurrency ?? 2;
	const timeoutMs = config.timeoutMs ?? 900_000;
	const memberTools = config.memberTools ?? [];
	const maxWords = config.maxAnswerWords ?? 400;
	const fullQuestion = context ? `${question}\n\nContext:\n${context}` : question;

	// ---- Stage 1: independent answers (parallel) ----
	const stage1Status = new Map<string, string>(config.members.map((m) => [m.name, "waiting"]));
	const emitStage1 = () =>
		onProgress(
			`Stage 1/3 — independent answers: ` +
				config.members.map((m) => `${m.name} ${stage1Status.get(m.name)}`).join(" · "),
		);
	emitStage1();

	const answers: MemberAnswer[] = await mapWithConcurrencyLimit(config.members, concurrency, async (member) => {
		stage1Status.set(member.name, "thinking…");
		emitStage1();
		const system =
			`You are "${member.name}", one member of a council of AI models advising a software engineer. ` +
			`Answer the question with your own independent judgment. Be concrete and direct. ` +
			`Commit to a clear recommendation — no fence-sitting. Keep it under ${maxWords} words.`;
		const result = await runPiOneShot(member.model, system, fullQuestion, memberTools, timeoutMs, cwd, signal);
		stage1Status.set(member.name, result.ok ? `done (${result.seconds}s)` : `FAILED (${result.error})`);
		emitStage1();
		return { member, result };
	});

	const okAnswers = answers.filter((a) => a.result.ok);
	if (okAnswers.length < 2) {
		const failures = answers
			.filter((a) => !a.result.ok)
			.map((a) => `${a.member.name}: ${a.result.error}`)
			.join("; ");
		throw new Error(`Council needs at least 2 answers, got ${okAnswers.length}. Failures — ${failures}`);
	}

	// ---- Anonymize: shuffle, assign labels ----
	const shuffled = [...okAnswers];
	for (let i = shuffled.length - 1; i > 0; i--) {
		const j = Math.floor(Math.random() * (i + 1));
		[shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
	}
	shuffled.forEach((a, i) => {
		a.label = LABELS[i];
	});
	const validLabels = shuffled.map((a) => a.label!) as string[];
	const anonymizedBlock = shuffled
		.map((a) => `### Response ${a.label}\n\n${a.result.text.trim()}`)
		.join("\n\n---\n\n");

	// ---- Stage 2: anonymized peer review (parallel) ----
	const stage2Status = new Map<string, string>(okAnswers.map((a) => [a.member.name, "waiting"]));
	const emitStage2 = () =>
		onProgress(
			`Stage 2/3 — anonymized peer review: ` +
				okAnswers.map((a) => `${a.member.name} ${stage2Status.get(a.member.name)}`).join(" · "),
		);
	emitStage2();

	const reviews: MemberReview[] = await mapWithConcurrencyLimit(okAnswers, concurrency, async (answer) => {
		stage2Status.set(answer.member.name, "reviewing…");
		emitStage2();
		const system =
			`You are "${answer.member.name}", a council member reviewing the council's anonymized answers. ` +
			`You do not know which model wrote which answer (one of them is yours — judge it on merit like the rest). ` +
			`Rank ALL responses from best to worst by correctness, depth of insight, and practical usefulness.`;
		const prompt =
			`Question the council was asked:\n${fullQuestion}\n\n` +
			`${anonymizedBlock}\n\n` +
			`Review each response in 1-2 sentences, then end with the final line EXACTLY in this format:\n` +
			`RANKING: ${validLabels.join(" > ")}\n` +
			`(reorder the letters to your actual ranking, best first)`;
		const result = await runPiOneShot(answer.member.model, system, prompt, [], timeoutMs, cwd, signal);
		const ranking = result.ok ? parseRanking(result.text, validLabels) : [];
		stage2Status.set(
			answer.member.name,
			result.ok ? (ranking.length > 0 ? `done (${result.seconds}s)` : "done (no parseable ranking)") : `FAILED`,
		);
		emitStage2();
		return { member: answer.member, result, ranking };
	});

	// ---- Tally: Borda count ----
	const points = new Map<string, number>(validLabels.map((l) => [l, 0]));
	for (const review of reviews) {
		const n = review.ranking.length;
		review.ranking.forEach((label, i) => {
			points.set(label, (points.get(label) ?? 0) + (n - 1 - i));
		});
	}
	const tally = shuffled
		.map((a) => ({ label: a.label!, member: a.member, points: points.get(a.label!) ?? 0 }))
		.sort((x, y) => y.points - x.points);

	// ---- Stage 3: chairman synthesis ----
	onProgress(`Stage 3/3 — chairman (${config.chairman.name}) is synthesizing the verdict…`);
	const revealedBlock = shuffled
		.map((a) => `### Response ${a.label} — by ${a.member.name} (${a.member.model})\n\n${a.result.text.trim()}`)
		.join("\n\n---\n\n");
	const reviewsBlock = reviews
		.map(
			(r) =>
				`### Review by ${r.member.name}\n\n${r.result.ok ? r.result.text.trim() : `(review failed: ${r.result.error})`}`,
		)
		.join("\n\n---\n\n");
	const tallyBlock = tally
		.map((t, i) => `${i + 1}. Response ${t.label} (${t.member.name}) — ${t.points} points`)
		.join("\n");

	const chairmanSystem =
		`You are the Chairman of an LLM council. The members answered a question independently, ` +
		`then peer-reviewed the anonymized answers. Your job is to synthesize the single best final answer. ` +
		`Weigh the peer rankings but use your own judgment — the majority can be wrong.`;
	const chairmanPrompt =
		`Question:\n${fullQuestion}\n\n` +
		`## Member answers (identities revealed)\n\n${revealedBlock}\n\n` +
		`## Peer reviews\n\n${reviewsBlock}\n\n` +
		`## Peer ranking tally (Borda count)\n\n${tallyBlock}\n\n` +
		`Write the council's final answer:\n` +
		`1. **Verdict** — the direct answer or recommendation.\n` +
		`2. **Where the council agreed.**\n` +
		`3. **Where it disagreed, which side you took, and why.**\n` +
		`Be decisive and keep it tight.`;
	const chairmanResult = await runPiOneShot(
		config.chairman.model,
		chairmanSystem,
		chairmanPrompt,
		[],
		timeoutMs,
		cwd,
		signal,
	);
	if (!chairmanResult.ok) {
		throw new Error(`Chairman (${config.chairman.model}) failed: ${chairmanResult.error}`);
	}

	// ---- Report ----
	const failedMembers = answers.filter((a) => !a.result.ok);
	const totalSeconds = Math.round(
		answers.reduce((s, a) => s + a.result.seconds, 0) +
			reviews.reduce((s, r) => s + r.result.seconds, 0) +
			chairmanResult.seconds,
	);
	const leaderboard = tally
		.map(
			(t, i) =>
				`| ${i + 1} | ${t.member.name} | \`${t.member.model}\` | ${t.points} | ${
					answers.find((a) => a.member.name === t.member.name)?.result.seconds ?? "-"
				}s |`,
		)
		.join("\n");

	let report =
		`## 🏛️ Council Verdict (chairman: ${config.chairman.name})\n\n` +
		`${chairmanResult.text.trim()}\n\n` +
		`### Peer-review leaderboard (Borda count, anonymized votes)\n\n` +
		`| Rank | Member | Model | Points | Answer time |\n|---|---|---|---|---|\n${leaderboard}\n\n` +
		`_${okAnswers.length} members answered, ${reviews.filter((r) => r.ranking.length > 0).length} valid rankings, ` +
		`~${totalSeconds}s total inference, $0.00 spent._`;
	if (failedMembers.length > 0) {
		report += `\n\n⚠️ Failed members: ${failedMembers.map((a) => `${a.member.name} (${a.result.error})`).join(", ")}`;
	}

	const outcome: CouncilOutcome = {
		question,
		answers,
		reviews,
		tally,
		chairman: { member: config.chairman, result: chairmanResult },
		report,
	};

	// ---- Transcript ----
	if (config.saveTranscripts !== false) {
		try {
			const dir = path.join(cwd, ".pi", "council");
			await fs.promises.mkdir(dir, { recursive: true });
			const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
			const file = path.join(dir, `council-${stamp}.md`);
			const transcript =
				`# Council transcript — ${new Date().toISOString()}\n\n` +
				`**Question:** ${question}\n\n${context ? `**Context:**\n${context}\n\n` : ""}` +
				`## Stage 1 — independent answers\n\n${revealedBlock}\n\n` +
				`## Stage 2 — anonymized peer reviews\n\n${reviewsBlock}\n\n` +
				`## Tally\n\n${tallyBlock}\n\n` +
				`## Stage 3 — chairman verdict (${config.chairman.name}, ${config.chairman.model})\n\n` +
				`${chairmanResult.text.trim()}\n`;
			await fs.promises.writeFile(file, transcript, "utf-8");
			outcome.transcriptPath = file;
			outcome.report += `\n\n📜 Full transcript: ${path.relative(cwd, file)}`;
		} catch {
			/* transcripts are best-effort */
		}
	}

	return outcome;
}

// ---------------------------------------------------------------------------
// /council init — generate a config from local providers in models.json
// ---------------------------------------------------------------------------

function generateConfigTemplate(): { config: CouncilConfig; note: string } {
	const candidates: CouncilMember[] = [];
	try {
		const modelsPath = path.join(getAgentDir(), "models.json");
		const models = JSON.parse(fs.readFileSync(modelsPath, "utf-8"));
		for (const [providerName, provider] of Object.entries<any>(models.providers ?? {})) {
			const baseUrl: string = provider.baseUrl ?? "";
			const isLocal = baseUrl.includes("localhost") || baseUrl.includes("127.0.0.1");
			if (!isLocal) continue;
			for (const model of provider.models ?? []) {
				candidates.push({ name: model.id.split(":")[0].split("/").pop(), model: `${providerName}/${model.id}` });
			}
		}
	} catch {
		/* fall through to placeholders */
	}

	// Prefer distinct model families so the council actually disagrees.
	const distinct: CouncilMember[] = [];
	const seenFamilies = new Set<string>();
	for (const c of candidates) {
		const family = c.name.replace(/[\d.:].*$/, "");
		if (seenFamilies.has(family)) continue;
		seenFamilies.add(family);
		distinct.push(c);
		if (distinct.length === 3) break;
	}
	const members =
		distinct.length >= 2
			? distinct
			: [
					{ name: "member-1", model: "ollama/<model-id>" },
					{ name: "member-2", model: "ollama/<model-id>" },
					{ name: "member-3", model: "<provider>/<model-id>" },
				];
	const chairman = candidates[0] ?? { name: "chairman", model: "<provider>/<strongest-local-model>" };
	const note =
		distinct.length >= 2
			? `Detected ${candidates.length} local models; picked ${members.length} distinct families. Edit to taste.`
			: `No local providers detected in models.json — fill in the placeholders.`;
	return {
		config: { members, chairman, concurrency: 2, memberTools: [], timeoutMs: 900_000, saveTranscripts: true },
		note,
	};
}

// ---------------------------------------------------------------------------
// Extension entry point
// ---------------------------------------------------------------------------

export default function (pi: ExtensionAPI) {
	pi.registerTool({
		name: "council",
		label: "Council",
		description:
			"Convene a council of local AI models on a question. Every member answers independently, " +
			"peer-reviews the anonymized answers, and a chairman synthesizes a final verdict. " +
			"Use it before committing to a significant decision (architecture, tricky bug diagnosis, " +
			"risky refactor, technology choice) or whenever the user asks for a council, a second " +
			"opinion, or a review by multiple models. Slow (several local inference calls) — " +
			"do not use for trivial questions.",
		parameters: Type.Object({
			question: Type.String({
				description: "The question for the council. Frame it neutrally — do not include your own conclusion.",
			}),
			context: Type.Optional(
				Type.String({
					description: "Optional supporting context: relevant code, the plan under review, constraints, error output.",
				}),
			),
		}),

		async execute(_toolCallId, params, signal, onUpdate, ctx) {
			const cwd = process.cwd();
			const loaded = loadConfig(cwd);
			if ("error" in loaded) {
				return { content: [{ type: "text", text: loaded.error }], details: {}, isError: true };
			}
			const onProgress: ProgressFn = (text) => {
				onUpdate?.({ content: [{ type: "text", text }], details: {} });
			};
			try {
				const outcome = await runCouncil(loaded.config, params.question, params.context, cwd, signal, onProgress);
				return {
					content: [{ type: "text", text: outcome.report }],
					details: {
						question: outcome.question,
						tally: outcome.tally.map((t) => ({ member: t.member.name, points: t.points })),
						transcript: outcome.transcriptPath,
					},
				};
			} catch (e) {
				const message = e instanceof Error ? e.message : String(e);
				return { content: [{ type: "text", text: `Council failed: ${message}` }], details: {}, isError: true };
			}
		},
	});

	pi.registerCommand("council", {
		description: "Convene the local model council: /council <question>, or /council init to generate a config",
		handler: async (args, ctx) => {
			const trimmed = args.trim();
			if (!trimmed) {
				const loaded = loadConfig(process.cwd());
				if ("error" in loaded) {
					ctx.ui.notify(loaded.error, "warning");
				} else {
					const c = loaded.config;
					ctx.ui.notify(
						`Council ready (${loaded.source}): ${c.members.map((m) => m.name).join(", ")} — chairman: ${c.chairman.name}. Usage: /council <question>`,
						"info",
					);
				}
				return;
			}

			if (trimmed === "init") {
				const target = userConfigPath();
				if (fs.existsSync(target)) {
					ctx.ui.notify(`Config already exists: ${target} — edit it directly.`, "warning");
					return;
				}
				const { config, note } = generateConfigTemplate();
				fs.writeFileSync(target, `${JSON.stringify(config, null, 2)}\n`, "utf-8");
				ctx.ui.notify(`Wrote ${target}. ${note}`, "info");
				return;
			}

			// Route the question through the main agent so the tool call, progress, and
			// verdict all render natively in the session.
			pi.sendUserMessage(
				`Convene the council on the following question using the council tool, then briefly state whether you agree with the verdict:\n\n${trimmed}`,
			);
		},
	});
}
