import type { SSEPatch, SSESnapshot } from "./types.js";

export class DeepSeekSSEParser {
  private buffer = "";
  private activePath = "";
  private activeFragmentType: "RESPONSE" | "THINK" | null = null;
  private textBuffer = "";
  private thinkingBuffer = "";
  private lastEmittedTextLen = 0;
  private lastEmittedThinkLen = 0;
  private done = false;
  private messageId: number | null = null;
  private onChunk: (
    content: string,
    reasoning: string,
    done: boolean,
    messageId: number | null,
  ) => void;

  constructor(
    onChunk: (content: string, reasoning: string, done: boolean, messageId: number | null) => void,
  ) {
    this.onChunk = onChunk;
  }

  feed(data: string): void {
    this.buffer += data;
    const lines = this.buffer.split("\n");
    this.buffer = lines.pop() || "";

    for (const line of lines) {
      this.processLine(line);
    }
  }

  flush(): void {
    if (this.buffer) {
      this.processLine(this.buffer);
      this.buffer = "";
    }
    this.finalize();
  }

  private processLine(line: string): void {
    const trimmed = line.trim();
    if (!trimmed.startsWith("data: ")) return;

    const payload = trimmed.slice(6).trim();
    if (payload === "[DONE]") {
      this.finalize();
      return;
    }

    try {
      const parsed = JSON.parse(payload) as Record<string, unknown>;
      const data = parsed.data as { biz_code?: number; biz_msg?: string } | undefined;

      if (data?.biz_code != null && data.biz_code !== 0) {
        throw new Error(`DeepSeek API error: ${data.biz_msg || `biz_code ${data.biz_code}`}`);
      }

      if (typeof parsed.response_message_id === "number") {
        this.messageId = parsed.response_message_id;
        return;
      }

      if (
        "v" in parsed &&
        typeof parsed.v === "object" &&
        parsed.v !== null &&
        "response" in (parsed.v as Record<string, unknown>)
      ) {
        const snapshot = parsed.v as SSESnapshot;
        this.handleSnapshot(snapshot);
        this.emitDelta();
      } else {
        this.handlePatch(parsed as SSEPatch);
      }
    } catch (e) {
      if (e instanceof SyntaxError) return;
      throw e;
    }
  }

  private fragmentContent(val: unknown): string {
    if (val == null) return "";
    if (typeof val === "string") return val;
    return JSON.stringify(val);
  }

  private handleSnapshot(snapshot: SSESnapshot): void {
    for (const fragment of snapshot.response.fragments) {
      if (fragment.type === "RESPONSE") {
        this.textBuffer += this.fragmentContent(fragment.content);
        this.activeFragmentType = "RESPONSE";
      } else if (fragment.type === "THINK" || fragment.type === "THINKING") {
        this.thinkingBuffer += this.fragmentContent(fragment.content);
        this.activeFragmentType = "THINK";
      } else if (fragment.content != null) {
        this.textBuffer += this.fragmentContent(fragment.content);
      }
    }
    this.activePath = "response/fragments/-1/content";
  }

  private handlePatch(patch: SSEPatch): void {
    if (patch.p) {
      this.activePath = patch.p;

      if (patch.p === "response/status" && patch.v === "FINISHED") {
        this.finalize();
        return;
      }

      if (patch.p === "response/message_id" && typeof patch.v === "number") {
        this.messageId = patch.v;
        return;
      }

      if (patch.p === "response/fragments" && patch.o === "APPEND" && Array.isArray(patch.v)) {
        for (const f of patch.v) {
          if (f.type === "THINK" || f.type === "THINKING") {
            this.thinkingBuffer += this.fragmentContent(f.content);
            this.activeFragmentType = "THINK";
          } else if (f.type === "RESPONSE") {
            this.textBuffer += this.fragmentContent(f.content);
            this.activeFragmentType = "RESPONSE";
          } else if (f.content != null) {
            this.textBuffer += this.fragmentContent(f.content);
          }
        }

        this.emitDelta();
        return;
      }

      if (patch.p.endsWith("/content") && typeof patch.v === "string") {
        this.appendContentFromPath(patch.p, patch.v);
        this.emitDelta();
      }
    } else if (typeof patch.v === "string" && this.activePath.endsWith("/content")) {
      this.appendContentFromPath(this.activePath, patch.v);
      this.emitDelta();
    }
  }

  private appendContentFromPath(path: string, value: string): void {
    if (path.includes("THINK") || path.includes("think") || this.activeFragmentType === "THINK") {
      this.thinkingBuffer += value;
    } else {
      this.textBuffer += value;
    }
  }

  private emitDelta(): void {
    const newText = this.textBuffer.slice(this.lastEmittedTextLen);
    const newThink = this.thinkingBuffer.slice(this.lastEmittedThinkLen);
    if (newText || newThink || this.done) {
      this.lastEmittedTextLen = this.textBuffer.length;
      this.lastEmittedThinkLen = this.thinkingBuffer.length;
      this.onChunk(newText, newThink, this.done, this.messageId);
    }
  }

  private finalize(): void {
    this.done = true;
    this.emitDelta();
  }
}
