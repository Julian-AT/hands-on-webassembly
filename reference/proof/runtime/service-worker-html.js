// Each response owns its transform. Work on bytes so split UTF-8 characters
// and untouched body content survive exactly as the ASGI application sent them.
function createInjectSocketFilter() {
  const tag = new TextEncoder().encode("</head>");
  const insertion = new TextEncoder().encode(
    `<script src="${dirname(self.location.pathname)}/shinylive-inject-socket.js" type="module"></script>\n`
  );
  let tail = new Uint8Array(0), injected = false;
  const html = response => /^text\/html(?:;|$)/i.test(response.headers.get("content-type") || "");
  function filter(chunk, response, final = false) {
    if (!html(response) || injected) return chunk;
    const bytes = new Uint8Array(tail.length + chunk.length);
    bytes.set(tail); bytes.set(chunk, tail.length);
    let match = -1;
    outer: for (let i = 0; i <= bytes.length - tag.length; i++) {
      for (let j = 0; j < tag.length; j++) if (bytes[i + j] !== tag[j]) continue outer;
      match = i; break;
    }
    if (match >= 0) {
      const result = new Uint8Array(bytes.length + insertion.length);
      result.set(bytes.subarray(0, match));
      result.set(insertion, match);
      result.set(bytes.subarray(match), match + insertion.length);
      tail = new Uint8Array(0); injected = true;
      return result;
    }
    // A tag may cross any chunk boundary. Hold at most tag.length - 1 bytes.
    const end = final ? bytes.length : Math.max(0, bytes.length - tag.length + 1);
    tail = bytes.slice(end);
    return bytes.subarray(0, end);
  }
  filter.transformResponseStart = message => {
    const headers = new Headers(message.headers);
    if (html({headers})) {
      if (headers.has("content-encoding") && headers.get("content-encoding") !== "identity") {
        throw new Error("Cannot transform encoded application HTML. Retry with an uncompressed response.");
      }
      // Length/range metadata and representation checksums describe the
      // original body, not the injected stream. Preserve other response headers.
      for (const name of ["content-length", "content-range", "accept-ranges", "etag", "content-md5", "digest", "content-digest", "repr-digest"]) headers.delete(name);
    }
    return {...message, headers: [...headers]};
  };
  return filter;
}
