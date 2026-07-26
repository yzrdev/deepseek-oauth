const { execSync } = require("node:child_process");
const { existsSync, statSync, renameSync, unlinkSync } = require("node:fs");
const { join } = require("node:path");

const wasmOut = join(__dirname, "..", "packages", "core", "solver.wasm");
const tmpWasm = wasmOut + ".tmp";
const cSrc = join(__dirname, "..", "packages", "core", "solver.c");

const isWindows = process.platform === "win32";
const isMac = process.platform === "darwin";
const isLinux = process.platform === "linux";

const llvmPaths = [
  ...(isWindows ? ["C:\\Program Files\\LLVM\\bin\\clang.exe"] : []),
  ...(isMac ? ["/opt/homebrew/opt/llvm/bin/clang", "/usr/local/opt/llvm/bin/clang"] : []),
  ...(isLinux ? ["/usr/bin/clang"] : []),
  "clang",
];

function findClang() {
  for (const p of llvmPaths) {
    try {
      execSync(`"${p}" --version`, { stdio: "pipe" });
      return p;
    } catch {}
  }
  return null;
}

function tryCompile(clangPath) {
  try {
    
    execSync(
      `"${clangPath}" --target=wasm32 -nostdlib -O3 -msimd128 -mbulk-memory -Wl,--no-entry -Wl,--export-all -o "${tmpWasm}" "${cSrc}"`,
      { stdio: "pipe" },
    );
    
    if (existsSync(tmpWasm)) {
      if (existsSync(wasmOut)) unlinkSync(wasmOut);
      renameSync(tmpWasm, wasmOut);
      return true;
    }
    return false;
  } catch (e) {
    
    try { if (existsSync(tmpWasm)) unlinkSync(tmpWasm); } catch {}
    return false;
  }
}

function installLlvm() {
  console.log("[build:wasm] Installing LLVM...");
  if (isMac) {
    try {
      execSync("brew install llvm", { stdio: "inherit" });
      return true;
    } catch {
      return false;
    }
  }
  if (isLinux) {
    try {
      execSync("sudo apt-get update -qq && sudo apt-get install -y -qq clang", {
        stdio: "inherit",
      });
      return true;
    } catch {
      return false;
    }
  }
  if (isWindows) {
    try {
      execSync(
        "winget install LLVM.LLVM --accept-source-agreements --accept-package-agreements --silent",
        { stdio: "inherit" },
      );
      return true;
    } catch {
      return false;
    }
  }
  return false;
}

let clang = findClang();

if (clang) {
  console.log("[build:wasm] Compiling solver.c → solver.wasm");
  if (tryCompile(clang) && existsSync(wasmOut)) {
    console.log(`[build:wasm] Done (${statSync(wasmOut).size} bytes)`);
    process.exit(0);
  }
}

if (existsSync(wasmOut)) {
  const reason = clang ? "compilation failed —" : "no clang found —";
  console.log(
    `[build:wasm] ${reason} using pre-built solver.wasm (${statSync(wasmOut).size} bytes)`,
  );
  process.exit(0);
}

if (installLlvm()) {
  clang = findClang();
  if (clang) {
    console.log("[build:wasm] Compiling solver.c → solver.wasm");
    if (tryCompile(clang) && existsSync(wasmOut)) {
      console.log(`[build:wasm] Done (${statSync(wasmOut).size} bytes)`);
      process.exit(0);
    }
  }
}

if (existsSync(wasmOut)) {
  console.log(`[build:wasm] Using pre-built solver.wasm (${statSync(wasmOut).size} bytes)`);
} else {
  console.warn("[build:wasm] solver.wasm not found — PoW will fail. Install clang or check in a pre-built solver.wasm");
}