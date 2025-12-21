// src/app/page.tsx
"use client";

import { useRef, useState } from "react";
import styles from "./page.module.css";

type Result = {
  observed: string[];
  likely_causes: string[];
  why: string;
  confidence: "high" | "medium" | "low" | string;
  question: string;
  image_caption: string;
  transcript?: string | null;
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export default function Page() {
  const [view, setView] = useState<"form" | "result">("form");

  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);

  const [text, setText] = useState("");
  const [audioFile, setAudioFile] = useState<File | null>(null);

  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState<string | null>(null);

  function resetAll() {
    setView("form");
    setResult(null);
    setError(null);
    setLoading(false);
    setText("");
    setAudioFile(null);
    setRecording(false);
    setImageFile(null);
    if (imagePreview) URL.revokeObjectURL(imagePreview);
    setImagePreview(null);
  }

  async function startRecording() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      chunksRef.current = [];

      mr.ondataavailable = (e) => chunksRef.current.push(e.data);
      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        const f = new File([blob], "recording.webm", { type: "audio/webm" });
        setAudioFile(f);
        stream.getTracks().forEach((t) => t.stop());
      };

      mr.start();
      mediaRecorderRef.current = mr;
      setRecording(true);
    } catch {
      setError("Microphone access denied. You can still upload an audio file.");
    }
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
    setRecording(false);
  }

  async function analyze() {
    if (!imageFile) {
      setError("Please upload an image (required).");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const form = new FormData();
      form.append("image", imageFile);
      if (audioFile) form.append("audio", audioFile);
      if (text.trim()) form.append("text", text.trim());

      const res = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const t = await res.text();
        throw new Error(t || "Request failed");
      }

      const json = (await res.json()) as Result;
      setResult(json);
      setView("result");
    } catch (e: any) {
      setError(e?.message ?? "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  const confPct = confidenceToPct(result?.confidence);

  return (
    <main className={styles.bg}>
      <div className={styles.shell}>
        {view === "form" ? (
          <div className={styles.card}>
            <div className={styles.header}>
              <div className={styles.logo}>✦</div>
              <div>
                <div className={styles.title}>ExplainMyWorld</div>
                <div className={styles.subtitle}>
                  Upload an image + optional voice/text. Get a grounded “what” + “why”.
                </div>
              </div>
            </div>

            {/* Image */}
            <div className={styles.section}>
              <div className={styles.labelRow}>
                <span className={styles.label}>Image</span>
                <span className={styles.required}>(required)</span>
              </div>

              <label className={styles.drop}>
                {imagePreview ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={imagePreview} alt="preview" className={styles.previewImg} />
                ) : (
                  <div className={styles.dropHint}>
                    <div>
                      <div className={styles.dropHintTitle}>Choose an image</div>
                      <div className={styles.dropHintSub}>PNG / JPG / JPEG</div>
                    </div>
                  </div>
                )}
                <input
                  type="file"
                  accept="image/*"
                  style={{ display: "none" }}
                  onChange={(e) => {
                    const f = e.target.files?.[0] ?? null;
                    setImageFile(f);
                    setError(null);
                    if (imagePreview) URL.revokeObjectURL(imagePreview);
                    setImagePreview(f ? URL.createObjectURL(f) : null);
                  }}
                />
              </label>
            </div>

            {/* Context */}
            <div className={styles.section}>
              <div className={styles.labelRow}>
                <span className={styles.label}>Context</span>
                <span className={styles.optional}>(optional)</span>
              </div>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="e.g., This is near campus at 5pm and it’s snowing."
                className={styles.textarea}
              />
            </div>

            {/* Voice */}
            <div className={styles.section}>
              <div className={styles.labelRow}>
                <span className={styles.label}>Voice</span>
                <span className={styles.optional}>(optional)</span>
              </div>

              <div className={styles.voiceRow}>
                <button
                  onClick={!recording ? startRecording : stopRecording}
                  className={[
                    styles.btnSoft,
                    recording ? styles.btnSoftDanger : styles.btnSoftGreen,
                  ].join(" ")}
                  disabled={loading}
                >
                  {recording ? "■ Stop Recording" : "🎙 Start Recording"}
                </button>

                <div className={styles.orText}>or</div>

                <label className={[styles.btnSoft, styles.btnSoftBlue].join(" ")}>
                  ⬆ Choose File
                  <input
                    type="file"
                    accept="audio/*"
                    style={{ display: "none" }}
                    onChange={(e) => setAudioFile(e.target.files?.[0] ?? null)}
                  />
                </label>
              </div>

              {(audioFile || recording) && (
                <div style={{ marginTop: 10, opacity: 0.75 }}>
                  {recording ? "Recording..." : `Audio: ${audioFile?.name}`}
                </div>
              )}
            </div>

            <button
              onClick={analyze}
              className={[styles.btnPrimary, loading ? styles.btnPrimaryDisabled : ""].join(" ")}
              disabled={loading}
            >
              {loading ? "Analyzing..." : "✦ Analyze"}
            </button>

            {error && <div className={styles.error}>{error}</div>}
          </div>
        ) : (
          <div className={styles.resultLayout}>
            <div className={styles.topBar}>
              <button onClick={resetAll} className={styles.backBtn}>
                ← Back
              </button>

              <div className={styles.brand}>
                <div className={styles.logoSmall}>✦</div>
                <div className={styles.brandName}>ExplainMyWorld</div>
              </div>
            </div>

            <div className={styles.resultCard}>
              <div className={styles.resultHeader}>Result</div>
              <div className={styles.hr} />

              <div className={styles.grid}>
                <Tile title="Image caption" tone="blue">
                  <div className={styles.text}>{result?.image_caption ?? "—"}</div>
                </Tile>

                <Tile title="Observed" tone="pink">
                  <ul className={styles.ul}>
                    {(result?.observed?.length ? result.observed : ["—"]).map((x, i) => (
                      <li key={i} className={styles.li}>
                        {x}
                      </li>
                    ))}
                  </ul>
                </Tile>

                <Tile title="Likely causes" tone="amber">
                  <ul className={styles.ul}>
                    {(result?.likely_causes?.length ? result.likely_causes : ["—"]).map((x, i) => (
                      <li key={i} className={styles.li}>
                        {x}
                      </li>
                    ))}
                  </ul>
                </Tile>

                <Tile title="Why" tone="green">
                  <div className={styles.text}>{result?.why ?? "—"}</div>
                </Tile>

                <Tile title="Confidence" tone="mint">
                  <div className={styles.progressRow}>
                    <div className={styles.progressLabel}>{result?.confidence ?? "—"}</div>
                    <div className={styles.progressTrack}>
                      <div className={styles.progressFill} style={{ width: `${confPct}%` }} />
                    </div>
                  </div>
                </Tile>

                <Tile title="Follow-up question" tone="violet">
                  <div className={styles.text} style={{ fontStyle: "italic" }}>
                    {result?.question ?? "—"}
                  </div>
                </Tile>
              </div>
            </div>

            <div className={styles.bottomBtnWrap}>
              <button onClick={resetAll} className={[styles.btnPrimary, styles.bottomBtn].join(" ")}>
                Analyze Another Image
              </button>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}

function confidenceToPct(conf?: string | null) {
  const c = (conf ?? "").toLowerCase();
  if (c.includes("high")) return 80;
  if (c.includes("medium")) return 55;
  if (c.includes("low")) return 30;
  return 50;
}

function Tile({
  title,
  tone,
  children,
}: {
  title: string;
  tone: "blue" | "pink" | "amber" | "green" | "mint" | "violet";
  children: React.ReactNode;
}) {
  const t = toneStyles[tone];
  return (
    <div className={styles.tile} style={{ borderColor: t.border, background: t.bg }}>
      <div className={styles.tileTitleRow}>
        <div className={styles.tileBar} style={{ background: t.bar }} />
        <div className={styles.tileTitle}>{title}</div>
      </div>
      {children}
    </div>
  );
}

const toneStyles: Record<string, { border: string; bg: string; bar: string }> = {
  blue: { border: "#bcd5ff", bg: "#f2f7ff", bar: "#2563eb" },
  pink: { border: "#f0c2ff", bg: "#fbf4ff", bar: "#d946ef" },
  amber: { border: "#ffd8a8", bg: "#fff7ea", bar: "#f59e0b" },
  green: { border: "#b9f6d3", bg: "#effff6", bar: "#10b981" },
  mint: { border: "#d6ffd6", bg: "#f4fff0", bar: "#22c55e" },
  violet: { border: "#d6d6ff", bg: "#f5f5ff", bar: "#7c3aed" },
};
