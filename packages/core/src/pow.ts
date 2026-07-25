import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { PoWChallenge, PoWResponse } from "./types.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

let wasmSolve:
  | ((salt: string, expireAt: number, difficulty: number, target: string) => number)
  | null = null;

function loadWasm(): typeof wasmSolve {
  if (wasmSolve) return wasmSolve;

  const wasmPath = join(__dirname, "..", "solver.wasm");
  if (!existsSync(wasmPath)) return null;

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

const RATE = 136;
const STATE_SIZE = 200;

const RC: bigint[] = [
  0x0000000000000001n,
  0x0000000000008082n,
  0x800000000000808an,
  0x8000000080008000n,
  0x000000000000808bn,
  0x0000000080000001n,
  0x8000000080008081n,
  0x8000000000008009n,
  0x000000000000008an,
  0x0000000000000088n,
  0x0000000080008009n,
  0x000000008000000an,
  0x000000008000808bn,
  0x800000000000008bn,
  0x8000000000008089n,
  0x8000000000008003n,
  0x8000000000008002n,
  0x8000000000000080n,
  0x000000000000800an,
  0x800000008000000an,
  0x8000000080008081n,
  0x8000000000008080n,
  0x0000000080000001n,
  0x8000000080008008n,
];

const RHO_OFFSETS = [
  0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14,
];

function rotl64(x: bigint, n: number): bigint {
  return ((x << BigInt(n & 63)) | (x >> BigInt((64 - (n & 63)) & 63))) & 0xffffffffffffffffn;
}

const _state = new Uint8Array(STATE_SIZE);
const _lanes = new Array<bigint>(25);
const _C = new Array<bigint>(5);
const _padded = new Uint8Array(RATE * 3);
const _result = new Uint8Array(32);
const _inputBuf = new Uint8Array(512);

function bytesToLanes(bytes: Uint8Array, lanes: bigint[]): void {
  for (let y = 0; y < 5; y++) {
    for (let x = 0; x < 5; x++) {
      let v = 0n;
      const off = 8 * (5 * y + x);
      for (let z = 0; z < 8; z++) {
        v |= BigInt(bytes[off + z]) << BigInt(8 * z);
      }
      lanes[x + 5 * y] = v;
    }
  }
}

function lanesToBytes(lanes: bigint[], bytes: Uint8Array): void {
  for (let y = 0; y < 5; y++) {
    for (let x = 0; x < 5; x++) {
      let v = lanes[x + 5 * y];
      const off = 8 * (5 * y + x);
      for (let z = 0; z < 8; z++) {
        bytes[off + z] = Number(v & 0xffn);
        v >>= 8n;
      }
    }
  }
}

function keccakF1600(state: bigint[], startRound: number, endRound: number): void {
  for (let r = startRound; r < endRound; r++) {
    for (let x = 0; x < 5; x++) {
      _C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
    }
    for (let x = 0; x < 5; x++) {
      const d = _C[(x + 4) % 5] ^ rotl64(_C[(x + 1) % 5], 1);
      for (let y = 0; y < 5; y++) state[x + 5 * y] ^= d;
    }

    let x = 1;
    let y = 0;
    let current = state[x + 5 * y];
    for (let t = 0; t < 24; t++) {
      const nx = y;
      const ny = (2 * x + 3 * y) % 5;
      const tmp = state[nx + 5 * ny];
      state[nx + 5 * ny] = rotl64(current, RHO_OFFSETS[x + 5 * y]);
      current = tmp;
      x = nx;
      y = ny;
    }

    for (let y = 0; y < 5; y++) {
      const iy = 5 * y;
      const l0 = state[iy];
      const l1 = state[1 + iy];
      const l2 = state[2 + iy];
      const l3 = state[3 + iy];
      const l4 = state[4 + iy];
      state[iy] = l0 ^ (~l1 & l2);
      state[1 + iy] = l1 ^ (~l2 & l3);
      state[2 + iy] = l2 ^ (~l3 & l4);
      state[3 + iy] = l3 ^ (~l4 & l0);
      state[4 + iy] = l4 ^ (~l0 & l1);
    }

    state[0] ^= RC[r];
  }
}

function hashInto(input: Uint8Array, out: Uint8Array): void {
  for (let i = 0; i < STATE_SIZE; i++) _state[i] = 0;

  const inputLen = input.length;
  const k = (RATE - ((inputLen + 2) % RATE)) % RATE;
  const paddedLen = inputLen + 2 + k;

  _padded.set(input, 0);
  _padded[inputLen] = 0x06;
  for (let i = 0; i < k; i++) _padded[inputLen + 1 + i] = 0x00;
  _padded[paddedLen - 1] = 0x80;

  for (let off = 0; off < paddedLen; off += RATE) {
    for (let j = 0; j < RATE; j++) _state[j] ^= _padded[off + j];
    bytesToLanes(_state, _lanes);
    keccakF1600(_lanes, 1, 24);
    lanesToBytes(_lanes, _state);
  }

  for (let i = 0; i < 32; i++) out[i] = _state[i];
}

export function deepSeekHashV1(input: Uint8Array): Uint8Array {
  const out = new Uint8Array(32);
  hashInto(input, out);
  return out;
}

export function solvePoW(challenge: PoWChallenge): PoWResponse {
  const solver = loadWasm();

  let answer: number;
  if (solver) {
    answer = solver(challenge.salt, challenge.expire_at, challenge.difficulty, challenge.challenge);
    if (answer < 0) {
      throw new Error(`PoW solve failed: no answer found within ${challenge.difficulty} attempts`);
    }
  } else {
    answer = solvePoWJS(challenge);
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

function solvePoWJS(challenge: PoWChallenge): number {
  const prefix = `${challenge.salt}_${challenge.expire_at}_`;
  const prefixLen = prefix.length;
  for (let i = 0; i < prefixLen; i++) _inputBuf[i] = prefix.charCodeAt(i);

  const challengeHex = challenge.challenge;

  for (let n = 0; n <= challenge.difficulty; n++) {
    const nStr = String(n);
    const nLen = nStr.length;
    for (let d = 0; d < nLen; d++) _inputBuf[prefixLen + d] = nStr.charCodeAt(d);

    hashInto(_inputBuf.subarray(0, prefixLen + nLen), _result);

    let match = true;
    for (let i = 0; i < 32 && match; i++) {
      const expected =
        (hexVal(challengeHex.charCodeAt(i * 2)) << 4) | hexVal(challengeHex.charCodeAt(i * 2 + 1));
      if (_result[i] !== expected) match = false;
    }

    if (match) return n;
  }

  throw new Error(`PoW solve failed: no answer found within ${challenge.difficulty} attempts`);
}

function hexVal(c: number): number {
  if (c >= 48 && c <= 57) return c - 48;
  if (c >= 65 && c <= 70) return c - 55;
  if (c >= 97 && c <= 102) return c - 87;
  return 0;
}

export function encodePowResponse(response: PoWResponse): string {
  return btoa(JSON.stringify(response));
}
