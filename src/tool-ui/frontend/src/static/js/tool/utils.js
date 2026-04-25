export async function readErrorMessage(response, fallback) {
  const text = await response.text();
  if (!text) {
    return fallback;
  }

  try {
    const data = JSON.parse(text);
    return data.detail || data.message || fallback;
  } catch {
    return text;
  }
}
