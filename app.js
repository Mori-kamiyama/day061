import { env, pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

const MODEL_ID = "onnx-community/Phi-4-mini-instruct-ONNX";
const MODEL_REVISION = "main";
const PROMPT =
`Senior bragging/lecturing tone. 0=normal, 30=max patronizing.
"おはよう"→0 "頑張れ"→4 "入りたて"→6 "いつまでも"→11 "経験上"→15 "俺に言わせれば"→18 "昔は"→20 "俺ぐらいだと"→24 "若者は"→27 "俺の時代は"→30
Phrase:"{TEXT}" Score:`;

const speedValueEl = document.getElementById("speedValue");
const speedCopyEl = document.getElementById("speedCopy");
const statusLineEl = document.getElementById("statusLine");
const messageCardTextEl = document.getElementById("messageCardText");
const meterProgressEl = document.getElementById("meterProgress");
const DEFAULT_TRANSCRIPT_TEXT = "ここに文字起こしが表示されます";

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

const SPEED_BANDS = [
    {
        min: 0,
        max: 0,
        copy: "無風",
    },
    {
        min: 1,
        max: 5,
        copy: "木の葉が動くぐらい",
    },
    {
        min: 6,
        max: 11,
        copy: "砂埃が舞うぐらい",
    },
    {
        min: 12,
        max: 17,
        copy: "電線が鳴るぐらい",
    },
    {
        min: 18,
        max: 24,
        copy: "小枝が折れ、風に向かって歩けない",
    },
    {
        min: 25,
        max: 30,
        copy: "人家に損害発生！",
    }
];

let generator = null;
let recognition = null;
const isRecognitionAvailable = Boolean(SpeechRecognition);
let modelReady = false;
let lastTranscriptForScoring = "";
let latestTranscript = "";
let latestDisplayedTranscript = "";
let scoreTimer = 0;
let hasMicPermissionError = false;
let hasRecognitionNetworkError = false;
let progressLength = 0;
let loadingHintTimer = 0;
let displayedSpeed = 0;
let speedAnimationFrame = 0;
let speedAnimationToken = 0;

const MIN_SPEED_ANIMATION_MS = 420;
const MAX_SPEED_ANIMATION_MS = 900;
const SPEED_ANIMATION_MS_PER_UNIT = 24;
const SCORING_MAX_CHARS = 48;
const SCORING_RECENT_SENTENCE_COUNT = 1;
const SCORING_DEBOUNCE_MS = 450;
const SCORING_MAX_NEW_TOKENS = 4;

function clampSpeed(value) {
    return Math.max(0, Math.min(30, value));
}

function getBand(speed) {
    return SPEED_BANDS.find((band) => speed >= band.min && speed <= band.max) || SPEED_BANDS.at(-1);
}

function setStatus(message, isError = false) {
    statusLineEl.textContent = message;
    statusLineEl.classList.toggle("is-error", isError);
}

function getSpeechNetworkErrorMessage() {
    if (location.protocol === "file:") {
        return "file:// では音声認識が不安定です。localhost で開いてください。";
    }

    if (navigator.onLine === false) {
        return "ネット接続がありません。Web Speech API はオンライン接続が必要です。";
    }

    return "音声認識サービスに接続できません。Chrome で localhost/https から開いてください。";
}

function formatBytes(value) {
    if (!Number.isFinite(value) || value <= 0) {
        return "";
    }

    if (value < 1024 * 1024) {
        return `${Math.round(value / 1024)}KB`;
    }

    return `${(value / (1024 * 1024)).toFixed(1)}MB`;
}

function handleModelProgress(progress) {
    const status = progress?.status ?? "";
    const loaded = progress?.loaded ?? 0;
    const total = progress?.total ?? 0;

    if (status === "progress" && total > 0) {
        const percent = Math.min(100, Math.round((loaded / total) * 100));
        const loadedText = formatBytes(loaded);
        const totalText = formatBytes(total);
        setStatus(`モデル読込中 ${percent}% (${loadedText}/${totalText})`);
        return;
    }

    if (status === "download") {
        setStatus("Gemma 3 270M をダウンロードしています。");
        return;
    }

    if (status === "initiate") {
        setStatus("Gemma 3 270M を準備しています。");
        return;
    }

    if (status === "ready") {
        setStatus("Gemma 3 270M の準備が完了しました。");
    }
}

function setTranscriptCard(text) {
    const normalized = text.trim();
    latestDisplayedTranscript = normalized;
    messageCardTextEl.textContent = normalized || DEFAULT_TRANSCRIPT_TEXT;
}

function setUnknownState(statusText, cardText = DEFAULT_TRANSCRIPT_TEXT, isError = false) {
    renderUnknownMeter("録音できていません");
    setTranscriptCard(cardText);
    setStatus(statusText, isError);
}

function setScoringFailureState(statusText) {
    renderUnknownMeter("判定できませんでした");
    setStatus(statusText, true);
}

function renderUnknownMeter(copyText) {
    stopSpeedAnimation();
    speedValueEl.textContent = "?";
    speedValueEl.dataset.unknown = "true";
    speedCopyEl.textContent = copyText;
    setMeterProgress(null);
}

function setMeterProgress(speed) {
    if (!progressLength) {
        progressLength = meterProgressEl.getTotalLength();
        meterProgressEl.style.strokeDasharray = `${progressLength}`;
    }

    if (speed === null) {
        meterProgressEl.style.strokeDashoffset = `${progressLength}`;
        meterProgressEl.classList.add("is-unknown");
        return;
    }

    const ratio = clampSpeed(speed) / 30;
    const newOffset = progressLength * (1 - ratio);
    meterProgressEl.classList.remove("is-unknown");
    meterProgressEl.style.strokeDashoffset = `${newOffset}`;
}

function renderMeasuredState(speed) {
    const rounded = clampSpeed(Math.round(speed));
    const band = getBand(rounded);

    displayedSpeed = clampSpeed(speed);
    speedValueEl.textContent = String(rounded);
    speedValueEl.dataset.unknown = "false";
    speedCopyEl.textContent = band.copy;
    setMeterProgress(displayedSpeed);
    setStatus("録音中");
}

function stopSpeedAnimation() {
    if (!speedAnimationFrame) {
        return;
    }

    window.cancelAnimationFrame(speedAnimationFrame);
    speedAnimationFrame = 0;
}

function easeOutCubic(t) {
    return 1 - ((1 - t) ** 3);
}

function setMeasuredState(speed) {
    const target = clampSpeed(speed);
    const start = displayedSpeed;
    const delta = Math.abs(target - start);
    const duration = Math.min(
        MAX_SPEED_ANIMATION_MS,
        Math.max(MIN_SPEED_ANIMATION_MS, Math.round(delta * SPEED_ANIMATION_MS_PER_UNIT))
    );
    const token = speedAnimationToken + 1;
    speedAnimationToken = token;

    stopSpeedAnimation();

    if (delta < 0.1) {
        renderMeasuredState(target);
        return;
    }

    let startedAt = 0;
    const tick = (now) => {
        if (token !== speedAnimationToken) {
            return;
        }

        if (!startedAt) {
            startedAt = now;
        }

        const elapsed = now - startedAt;
        const progress = Math.min(1, elapsed / duration);
        const eased = easeOutCubic(progress);
        const current = start + ((target - start) * eased);
        renderMeasuredState(current);

        if (progress < 1) {
            speedAnimationFrame = window.requestAnimationFrame(tick);
            return;
        }

        renderMeasuredState(target);
        speedAnimationFrame = 0;
    };

    speedAnimationFrame = window.requestAnimationFrame(tick);
}

function normalizeGeneratedSpeed(resultText) {
    const matches = [...resultText.matchAll(/\b([0-9]|[12][0-9]|30)\b/g)];
    const normalized = matches.at(-1)?.[1] ?? "";
    if (!normalized) {
        return null;
    }

    const parsed = Number.parseInt(normalized, 10);
    if (!Number.isFinite(parsed)) {
        return null;
    }

    return clampSpeed(parsed);
}

function buildScoringPrompt(transcript) {
    return `${PROMPT}\nSpeech: ${transcript}\nScore:`;
}

function compactTranscriptForScoring(text) {
    const normalized = text
        .replace(/\s+/g, " ")
        .replace(/[。！？]+/g, "。")
        .trim();

    if (!normalized) {
        return "";
    }

    const sentenceParts = normalized
        .split("。")
        .map((part) => part.trim())
        .filter(Boolean);

    const recentSentences = sentenceParts.slice(-SCORING_RECENT_SENTENCE_COUNT).join("。");
    return recentSentences.slice(-SCORING_MAX_CHARS);
}

async function ensureGenerator() {
    if (generator) {
        return generator;
    }

    env.allowLocalModels = false;
    env.useBrowserCache = true;
    setStatus("Gemma 3 270M を準備しています。");
    window.clearTimeout(loadingHintTimer);
    loadingHintTimer = window.setTimeout(() => {
        setStatus("初回は数百MBのモデル読込が発生します。しばらくお待ちください。");
    }, 12000);

    let selectedDevice = "wasm";
    if (navigator.gpu) {
        selectedDevice = "webgpu";
    }

    try {
        generator = await pipeline("text-generation", MODEL_ID, {
            dtype: "q4",
            device: selectedDevice,
            progress_callback: handleModelProgress
        });
    } catch (error) {
        if (selectedDevice !== "webgpu") {
            throw error;
        }

        console.warn("WebGPU initialization failed. Falling back to WASM.", error);
        setStatus("WebGPU を利用できないため WASM で実行します。");
        generator = await pipeline("text-generation", MODEL_ID, {
            revision: MODEL_REVISION,
            dtype: "q4f16",
            device: "wasm",
            progress_callback: handleModelProgress
        });
    }

    window.clearTimeout(loadingHintTimer);

    return generator;
}

function extractGeneratedText(output) {
    const generated = output?.[0]?.generated_text;
    if (typeof generated === "string") {
        return generated;
    }

    if (Array.isArray(generated)) {
        const tail = generated.at(-1);
        if (typeof tail === "string") {
            return tail;
        }

        return tail?.content ?? "";
    }

    return generated?.content ?? "";
}

async function scoreTranscript(transcript) {
    const trimmed = transcript.trim();
    if (!trimmed || trimmed === lastTranscriptForScoring || !modelReady) {
        return;
    }

    lastTranscriptForScoring = trimmed;
    setStatus("先輩風を判定しています。");

    try {
        const llm = await ensureGenerator();
        const output = await llm(buildScoringPrompt(trimmed), {
            do_sample: false,
            max_new_tokens: SCORING_MAX_NEW_TOKENS,
            repetition_penalty: 1.05,
            return_full_text: false
        });

        const responseText = extractGeneratedText(output);
        const speed = normalizeGeneratedSpeed(responseText);

        if (speed === null) {
            console.warn("Unexpected model output:", responseText);
            setScoringFailureState("数値の判定に失敗しました。");
            return;
        }

        setMeasuredState(speed);
    } catch (error) {
        console.error("Scoring failed:", error);
        setScoringFailureState("モデルの判定に失敗しました。");
    }
}

function queueTranscriptScore(transcript) {
    window.clearTimeout(scoreTimer);
    scoreTimer = window.setTimeout(() => {
        scoreTranscript(transcript);
    }, SCORING_DEBOUNCE_MS);
}

function bootRecognition() {
    if (!isRecognitionAvailable) {
        setUnknownState("このブラウザは録音に対応していません。");
        return;
    }

    recognition = new SpeechRecognition();
    recognition.lang = "ja-JP";
    recognition.continuous = true;
    recognition.interimResults = true;

    recognition.onstart = () => {
        hasRecognitionNetworkError = false;
        if (modelReady) {
            setStatus("録音中");
        } else {
            setStatus("録音中 / モデルを読み込んでいます。");
        }
    };

    recognition.onresult = (event) => {
        let finalTranscript = latestTranscript;
        let interimTranscript = "";
        let newFinalTranscript = "";

        for (let i = event.resultIndex; i < event.results.length; i += 1) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                finalTranscript += transcript;
                newFinalTranscript += transcript;
            } else {
                interimTranscript += transcript;
            }
        }

        latestTranscript = finalTranscript;
        const visibleTranscript = `${finalTranscript} ${interimTranscript}`.trim();
        const scoringSource = (newFinalTranscript || interimTranscript || latestDisplayedTranscript).trim();
        const stableTranscript = compactTranscriptForScoring(scoringSource);

        setTranscriptCard(visibleTranscript);

        if (!stableTranscript) {
            return;
        }

        if (!modelReady) {
            setStatus("録音中 / モデル待機中");
            return;
        }

        queueTranscriptScore(stableTranscript);
    };

    recognition.onerror = (event) => {
        console.error("Speech recognition error:", event.error);

        if (event.error === "not-allowed" || event.error === "service-not-allowed") {
            hasMicPermissionError = true;
            setUnknownState("マイクを許可してください。", DEFAULT_TRANSCRIPT_TEXT, true);
            return;
        }

        if (event.error === "no-speech" || event.error === "audio-capture") {
            setUnknownState("録音できていません。", DEFAULT_TRANSCRIPT_TEXT, true);
            return;
        }

        if (event.error === "network") {
            hasRecognitionNetworkError = true;
            setUnknownState(getSpeechNetworkErrorMessage(), DEFAULT_TRANSCRIPT_TEXT, true);
            return;
        }

        setUnknownState("録音エラーが発生しました。", DEFAULT_TRANSCRIPT_TEXT, true);
    };

    recognition.onend = () => {
        if (hasMicPermissionError || hasRecognitionNetworkError) {
            return;
        }

        try {
            recognition.start();
        } catch (error) {
            console.error("Recognition restart failed:", error);
            setUnknownState("録音を再開できませんでした。", DEFAULT_TRANSCRIPT_TEXT, true);
        }
    };

    try {
        recognition.start();
    } catch (error) {
        console.error("Recognition start failed:", error);
        setUnknownState("録音を開始できませんでした。", DEFAULT_TRANSCRIPT_TEXT, true);
    }
}

async function init() {
    setUnknownState("モデルを読み込んでいます。", DEFAULT_TRANSCRIPT_TEXT);
    bootRecognition();

    try {
        await ensureGenerator();
        modelReady = true;
        if (latestTranscript.trim()) {
            queueTranscriptScore(latestTranscript);
        } else if (isRecognitionAvailable) {
            setStatus("録音中");
        } else {
            setStatus("モデルの読み込みが完了しました。");
        }
    } catch (error) {
        console.error("Model load failed:", error);
        setUnknownState("モデル読み込みに失敗しました。", DEFAULT_TRANSCRIPT_TEXT, true);
    }
}

init();
