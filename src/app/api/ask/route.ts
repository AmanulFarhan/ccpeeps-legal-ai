const BACKEND_TIMEOUT_MS = 180_000;

export async function POST(request: Request) {
  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return Response.json({ error: "Request body must be valid JSON." }, { status: 400 });
  }

  if (
    !payload ||
    typeof payload !== "object" ||
    !("question" in payload) ||
    typeof payload.question !== "string" ||
    !payload.question.trim()
  ) {
    return Response.json({ error: "Enter a question to continue." }, { status: 400 });
  }

  const responseLanguage =
    "response_language" in payload &&
    (payload.response_language === "English" || payload.response_language === "Malayalam")
      ? payload.response_language
      : "English";
  const backendUrl = (process.env.RAG_API_URL ?? "http://127.0.0.1:8000").replace(/\/+$/, "");

  try {
    const backendResponse = await fetch(`${backendUrl}/api/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: payload.question.trim(),
        response_language: responseLanguage,
      }),
      cache: "no-store",
      signal: AbortSignal.timeout(BACKEND_TIMEOUT_MS),
    });

    const result: unknown = await backendResponse.json().catch(() => null);
    if (!backendResponse.ok) {
      const detail =
        result && typeof result === "object" && "detail" in result && typeof result.detail === "string"
          ? result.detail
          : "The legal assistant could not process this question.";
      return Response.json({ error: detail }, { status: backendResponse.status });
    }

    return Response.json(result);
  } catch (error) {
    const timedOut = error instanceof Error && error.name === "TimeoutError";
    return Response.json(
      {
        error: timedOut
          ? "The request took too long. Please try again."
          : "The RAG backend is unavailable. Start the Python backend and try again.",
      },
      { status: timedOut ? 504 : 503 },
    );
  }
}
