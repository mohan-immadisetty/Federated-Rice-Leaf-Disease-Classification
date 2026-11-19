export const runtime = "nodejs";
export const maxDuration = 15;

export async function GET() {
  const space = process.env.SPACE_URL?.replace(/\/+$/, "");
  if (!space) {
    return Response.json({ status: "unconfigured", detail: "SPACE_URL is not set" }, { status: 500 });
  }
  try {
    const upstream = await fetch(`${space}/health`, {
      cache: "no-store",
      signal: AbortSignal.timeout(10_000),
    });
    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: { "content-type": "application/json" },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return Response.json({ status: "unreachable", detail: message }, { status: 502 });
  }
}
