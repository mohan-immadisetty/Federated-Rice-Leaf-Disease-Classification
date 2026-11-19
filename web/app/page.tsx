"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Prediction = { class: string; label: string; prob: number };

type Result = {
  md5: string;
  e5: Prediction;
  fedavg: Prediction;
  top5: Prediction[];
  in_dataset: boolean;
  true_label: string | null;
  true_label_pretty: string | null;
  source: string | null;
  split: string | null;
  official_test: boolean;
  exclusion_reason: string | null;
  latency_ms: number;
};

type Sample = { file: string; true_class: string; source: string };

const MAX_UPLOAD = 3.5 * 1024 * 1024;

function pct(p: number) {
  return `${(100 * p).toFixed(1)}%`;
}

/** Keep the request under Vercel's ~4.5 MB serverless body limit. */
async function prepare(file: File): Promise<File> {
  if (file.size <= MAX_UPLOAD) return file;
  try {
    const bitmap = await createImageBitmap(file);
    const scale = Math.min(1, 2200 / Math.max(bitmap.width, bitmap.height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(bitmap.width * scale));
    canvas.height = Math.max(1, Math.round(bitmap.height * scale));
    const ctx = canvas.getContext("2d");
    if (!ctx) return file;
    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", 0.9)
    );
    if (!blob) return file;
    const name = file.name.replace(/\.\w+$/, "") + ".jpg";
    return new File([blob], name, { type: "image/jpeg" });
  } catch {
    return file;
  }
}

export default function Home() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [samples, setSamples] = useState<Sample[]>([]);
  const [over, setOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const previewRef = useRef<string | null>(null);

  useEffect(() => {
    fetch("/samples/samples.json")
      .then((r) => (r.ok ? r.json() : []))
      .then((rows: Sample[]) => setSamples(rows))
      .catch(() => setSamples([]));
  }, []);

  useEffect(() => {
    previewRef.current = preview;
  }, [preview]);

  useEffect(() => {
    return () => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    };
  }, []);

  const run = useCallback(async (file: File) => {
    setBusy(true);
    setError(null);
    setResult(null);
    setFileName(file.name);
    const url = URL.createObjectURL(file);
    setPreview((old) => {
      if (old) URL.revokeObjectURL(old);
      return url;
    });

    try {
      const upload = await prepare(file);
      const body = new FormData();
      body.append("file", upload, upload.name);
      const res = await fetch("/api/predict", { method: "POST", body });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || `Request failed (${res.status})`);
      }
      setResult(data as Result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }, []);

  const runSample = useCallback(
    async (sample: Sample) => {
      try {
        const res = await fetch(`/samples/${sample.file}`);
        const blob = await res.blob();
        await run(new File([blob], sample.file, { type: blob.type || "image/jpeg" }));
      } catch {
        setError("Could not load the sample image.");
      }
    },
    [run]
  );

  const correct = result?.true_label ? result.e5.class === result.true_label : null;

  return (
    <main>
      <header className="hero">
        <div className="badges">
          <span className="badge green">Final E5 model</span>
          <span className="badge">19 rice-leaf classes</span>
          <span className="badge">FedAvg baseline included</span>
        </div>
        <h1>🌾 Rice Leaf Disease Checker</h1>
        <p>
          Upload a rice-leaf photo. The model predicts the disease and shows the top-5
          probabilities. <strong>E5</strong> fuses frozen FedAvg EfficientNetV2-S features
          with frozen DINOv2-base features and classifies them with a federated linear head.
        </p>
        <p className="muted">
          Held-out test (n = 2,793): <strong>92.34%</strong> accuracy and{" "}
          <strong>89.58</strong> macro-F1. The FedAvg baseline scores 85.82% / 67.39.
        </p>
      </header>

      <section className="card">
        <label
          className={`drop${over ? " over" : ""}`}
          htmlFor="file"
          onDragEnter={(e) => {
            e.preventDefault();
            setOver(true);
          }}
          onDragOver={(e) => {
            e.preventDefault();
            setOver(true);
          }}
          onDragLeave={(e) => {
            e.preventDefault();
            setOver(false);
          }}
          onDrop={(e) => {
            e.preventDefault();
            setOver(false);
            const f = e.dataTransfer.files?.[0];
            if (f) void run(f);
          }}
        >
          <div className="icon">📷</div>
          <div>
            Drop a rice-leaf image here or <strong>browse</strong>
          </div>
          <small>JPG, PNG, WEBP · large images are downscaled automatically</small>
        </label>
        <input
          id="file"
          ref={inputRef}
          type="file"
          accept="image/*"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void run(f);
            e.currentTarget.value = "";
          }}
        />

        {busy && (
          <div className="spinner">
            <span className="dot" />
            Running E5 inference… the free Space can take a few seconds to wake up.
          </div>
        )}
        {error && <div className="error">{error}</div>}

        {samples.length > 0 && (
          <>
            <div className="section-label">Or try a real held-out test image:</div>
            <div className="samples">
              {samples.map((s) => (
                <button
                  key={s.file}
                  className="sample"
                  title={s.true_class.replaceAll("_", " ")}
                  onClick={() => void runSample(s)}
                  type="button"
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={`/samples/${s.file}`} alt={s.true_class.replaceAll("_", " ")} />
                </button>
              ))}
            </div>
          </>
        )}
      </section>

      {result && (
        <section className="card">
          <div className="result-grid">
            <div>
              {preview && (
                // eslint-disable-next-line @next/next/no-img-element
                <img className="preview" src={preview} alt={fileName || "uploaded leaf"} />
              )}
              <div className="muted" style={{ marginTop: 10 }}>
                {fileName}
              </div>
            </div>

            <div>
              <div className="kicker">E5 prediction</div>
              <div className="prediction">{result.e5.label}</div>
              <div className="confidence">
                confidence {pct(result.e5.prob)} · {result.latency_ms} ms
              </div>

              <div className="verdict">
                {result.true_label ? (
                  correct ? (
                    <span className="ok">✅ Correct — true label: {result.true_label_pretty}</span>
                  ) : (
                    <span className="bad">❌ Wrong — true label: {result.true_label_pretty}</span>
                  )
                ) : (
                  <span className="muted">No ground-truth label for this image.</span>
                )}
              </div>

              {result.in_dataset ? (
                <div className="meta">
                  <span className="badge">{result.source}</span>
                  <span className="badge">
                    {result.official_test ? "official held-out test" : `split: ${result.split}`}
                  </span>
                  {result.split === "train" && (
                    <span className="badge">⚠ training image — not a fair test</span>
                  )}
                </div>
              ) : (
                <div className="meta">
                  <span className="badge">new image — not in the project dataset</span>
                </div>
              )}

              <div className="bars">
                {result.top5.map((p) => (
                  <div className="bar" key={p.class}>
                    <span>{p.label}</span>
                    <span className="track">
                      <span className="fill" style={{ width: pct(p.prob) }} />
                    </span>
                    <span className="muted">{pct(p.prob)}</span>
                  </div>
                ))}
              </div>

              <div className="baseline">
                <span className="muted">FedAvg baseline (same backbone): </span>
                <strong>{result.fedavg.label}</strong>{" "}
                <span className="muted">({pct(result.fedavg.prob)})</span>
                {result.true_label && (
                  <span className={result.fedavg.class === result.true_label ? "ok" : "bad"}>
                    {result.fedavg.class === result.true_label ? " ✅" : " ❌"}
                  </span>
                )}
              </div>
            </div>
          </div>
        </section>
      )}

      <footer>
        Inference runs on a Hugging Face Space (FastAPI + PyTorch); this page is a
        Next.js app on Vercel and never loads the model itself.{" "}
        <a href="/api/health">API health</a> ·{" "}
        <a href="https://huggingface.co/facebook/dinov2-base">DINOv2-base</a>
        <br />
        Images from the project dataset are matched by exact MD5; resized or edited
        images are treated as new.
      </footer>
    </main>
  );
}
