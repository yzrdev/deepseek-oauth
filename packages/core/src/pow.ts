import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { PoWChallenge, PoWResponse } from "./types.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

let wasmSolve:
  | ((salt: string, expireAt: number, difficulty: number, target: string) => number)
  | null = null;

function loadWasm(): (salt: string, expireAt: number, difficulty: number, target: string) => number {
  if (wasmSolve) return wasmSolve;

  const wasmPath = join(__dirname, "..", "solver.wasm");
  if (!existsSync(wasmPath)) throw new Error("WASM solver not found — PoW requires WASM support");

  const buf = readFileSync(wasmPath);
  const mod = new WebAssembly.Module(buf);
  const instance = new WebAssembly.Instance(mod, {});
  const mem = instance.exports.memory as WebAssembly.Memory;
  const memBuf = new Uint8Array(mem.buffer);
  const enc = new TextEncoder();

  wasmSolve = (salt: string, expireAt: number, difficulty: number, target: string): number => {
    const needed = salt.length + target.length + 512;
    if (needed > mem.buffer.byteLength) {
      mem.grow(Math.ceil((needed - mem.buffer.byteLength) / 65536) + 1);
    }

    const buf2 = new Uint8Array(mem.buffer);
    const saltBytes = enc.encode(salt);
    const targetBytes = enc.encode(target);

    buf2.set(saltBytes, 0);
    buf2.set(targetBytes, 256);

    const answer = (instance.exports.solve_pow_opt as CallableFunction)(
      0,
      saltBytes.length,
      BigInt(expireAt),
      difficulty,
      256,
      targetBytes.length,
    ) as number;

    return answer;
  };

  return wasmSolve;
}



export function solvePoW(challenge: PoWChallenge): PoWResponse {
  const solver = loadWasm();
  const answer = solver(challenge.salt, challenge.expire_at, challenge.difficulty, challenge.challenge);
  if (answer < 0) {
    throw new Error(`PoW solve failed: no answer found within ${challenge.difficulty} attempts`);
  }
  return {
    algorithm: challenge.algorithm,
    challenge: challenge.challenge,
    salt: challenge.salt,
    answer,
    signature: challenge.signature,
    target_path: challenge.target_path,
  };
}


export function encodePowResponse(response: PoWResponse): string {
  return btoa(JSON.stringify(response));
}
