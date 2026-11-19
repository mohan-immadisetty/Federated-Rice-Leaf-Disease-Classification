export const runtime = "nodejs";
export const maxDuration = 60;

export async function POST(req: Request) {
  const space = process.env.SPACE_URL?.replace(/\/+$/, "");
  if (!space) {
    return Response.json(
      { detail: "SPACE_URL is not configured on the server. Set it in Vercel project settings." },
      { status: 500 }
    );
  }

  let form: FormData;
  try {
    form = await req.formData();
  } catch {
    return Response.json({ detail: "Expected multipart/form-data." }, { status: 400 });
  }

  const file = form.get("file");
  if (!(file instanceof File) || file.size === 0) {
    return Response.json({ detail: "Missing image file field `file`." }, { status: 400 });
  }

  const body = new FormData();
  body.append("file", file, file.name || "leaf.jpg");

  try {
    const upstream = await fetch(`${space}/predict`, {
      method: "POST",
      body,
      cache: "no-store",
      signal: AbortSignal.timeout(55_000),
    });
    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: { "content-type": upstream.headers.get("content-type") ?? "application/json" },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown upstream error";
    return Response.json({ detail: `Inference service unreachable: ${message}` }, { status: 502 });
  }
}
