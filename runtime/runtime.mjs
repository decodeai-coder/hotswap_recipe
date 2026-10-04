#!/usr/bin/env node
import http from "node:http";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { createReadStream } from "node:fs";
import { mkdir, readFile, rename, stat, unlink, writeFile } from "node:fs/promises";
import { dirname, extname, resolve } from "node:path";
import os from "node:os";

const execFileAsync = promisify(execFile);
const HOST = process.env.MODEL_FLEET_HOST || "127.0.0.1";
const PORT = Number(process.env.MODEL_FLEET_PORT || 8199);
const DATA_ROOT = process.env.MODEL_FLEET_DATA_ROOT || resolve(os.homedir(), ".local/share/model-fleet-data");
const STATE_PATH = process.env.MODEL_FLEET_STATE_PATH || resolve(DATA_ROOT, "state.json");
const TEXT_URL = process.env.MODEL_FLEET_TEXT_HEALTH_URL || "http://127.0.0.1:8091/health";
const COMFY_URL = process.env.MODEL_FLEET_COMFY_HEALTH_URL || "http://127.0.0.1:8190/system_stats";
const CONTINUITY_URL = process.env.MODEL_FLEET_CONTINUITY_HEALTH_URL || "http://127.0.0.1:8092/health";
const OUTPUT_ROOT = process.env.MODEL_FLEET_OUTPUT_ROOT || resolve(DATA_ROOT, "output");
const PUBLIC_HOST = process.env.MODEL_FLEET_PUBLIC_HOST || HOST;
const GPU_SELECTOR = process.env.MODEL_FLEET_MEDIA_GPU_UUID || process.env.MODEL_FLEET_GPU_UUID || process.env.MODEL_FLEET_GPU_INDEX || "0";
const ENCODER_GPU_SELECTOR = process.env.MODEL_FLEET_ENCODER_GPU_UUID || "";
const ENCODER_DEVICE = process.env.MODEL_FLEET_ENCODER_DEVICE || "cuda:1";
const VAE_DEVICE = process.env.MODEL_FLEET_VAE_DEVICE || ENCODER_DEVICE;
const SYSTEMCTL_SCOPE = process.env.MODEL_FLEET_SYSTEMCTL_SCOPE || "user";
const EXTERNAL_LEASE_BASE_URL = (process.env.MODEL_FLEET_EXTERNAL_LEASE_BASE_URL || "").replace(/\/$/, "");
const EXTERNAL_LEASE_ACQUIRE_URL = process.env.MODEL_FLEET_EXTERNAL_LEASE_ACQUIRE_URL || (EXTERNAL_LEASE_BASE_URL ? `${EXTERNAL_LEASE_BASE_URL}/acquire` : "");
const EXTERNAL_LEASE_RENEW_URL = process.env.MODEL_FLEET_EXTERNAL_LEASE_RENEW_URL || (EXTERNAL_LEASE_BASE_URL ? `${EXTERNAL_LEASE_BASE_URL}/renew` : "");
const EXTERNAL_LEASE_RELEASE_URL = process.env.MODEL_FLEET_EXTERNAL_LEASE_RELEASE_URL || (EXTERNAL_LEASE_BASE_URL ? `${EXTERNAL_LEASE_BASE_URL}/release` : "");
const EXTERNAL_LEASE_TTL_MS = Number(process.env.MODEL_FLEET_EXTERNAL_LEASE_TTL_MS || 21600000);
const EXTERNAL_LEASE_RENEW_INTERVAL_MS = Number(process.env.MODEL_FLEET_EXTERNAL_LEASE_RENEW_INTERVAL_MS || 60000);
const EXTERNAL_LEASE_REQUEST_TIMEOUT_MS = Number(process.env.MODEL_FLEET_EXTERNAL_LEASE_REQUEST_TIMEOUT_MS || 90000);
const EXTERNAL_LEASE_STATE_PATH = process.env.MODEL_FLEET_EXTERNAL_LEASE_STATE_PATH || resolve(DATA_ROOT, "external-lease.json");
function envFlag(name, defaultValue = false) {
  const raw = process.env[name];
  if (raw == null || raw === "") return defaultValue;
  return ["1", "true", "yes", "on"].includes(String(raw).toLowerCase());
}

const STICKY_MEDIA_ENABLED = envFlag("MODEL_FLEET_STICKY_MEDIA");
const STICKY_MEDIA_RETRY_MS = Number(process.env.MODEL_FLEET_STICKY_MEDIA_RETRY_MS || 2000);
const MEDIA_PRELOAD_ENABLED = envFlag("MODEL_FLEET_MEDIA_PRELOAD");
const MEDIA_PRELOAD_MODEL = process.env.MODEL_FLEET_MEDIA_PRELOAD_MODEL || "minimax_h3_fl2va_pruned-Q4_K_M.gguf";
const MEDIA_PRELOAD_ENCODER = process.env.MODEL_FLEET_MEDIA_PRELOAD_ENCODER || "qwen3vl_32b_minimax_h3-Q4_K_M.gguf";
const MEDIA_PRELOAD_VIDEO_VAE = process.env.MODEL_FLEET_MEDIA_PRELOAD_VIDEO_VAE || "minimax_h3_video_vae_fp16_model_fleet.safetensors";
const MEDIA_PRELOAD_AUDIO_VAE = process.env.MODEL_FLEET_MEDIA_PRELOAD_AUDIO_VAE || "minimax_h3_audio_vae_fp32_model_fleet.safetensors";
const MEDIA_PRELOAD_TIMEOUT_MS = Number(process.env.MODEL_FLEET_MEDIA_PRELOAD_TIMEOUT_MS || 900000);
const MEDIA_CONTINUITY_ENABLED = envFlag("MODEL_FLEET_MEDIA_CONTINUITY", true);
const CONTINUITY_UNIT = "model-fleet-g9v3-media.service";

const profileCatalog = {
  "qwen27-long": { unit: "model-fleet-qwen27-long.service", role: "main", context: 262144, concurrency: 1, vision: true, mtp: true },
  "qwen27-multi": { unit: "model-fleet-qwen27-multi.service", role: "main", context: 393216, concurrency: 3, per_slot_context: 131072, vision: true, mtp: true },
  "qwen35-long": { unit: "model-fleet-qwen35-long.service", role: "main", context: 262144, concurrency: 1, vision: true, mtp: true },
  "qwen35-multi": { unit: "model-fleet-qwen35-multi.service", role: "main", context: 393216, concurrency: 3, per_slot_context: 131072, vision: true, mtp: true },
  g9v3: { unit: "model-fleet-g9v3.service", role: "continuity", context: 131072, concurrency: 1, weights: "bf16", kv_cache: "fp8", note: "online FP8 weights are disabled on RTX 3090 because Ampere Marlin produced corrupt output" },
  media: { role: "media", continuity: MEDIA_CONTINUITY_ENABLED ? CONTINUITY_UNIT : null, continuity_context: MEDIA_CONTINUITY_ENABLED ? 131072 : 0, comfy: "model-fleet-comfy.service" },
  audio: { role: "audio", continuity: MEDIA_CONTINUITY_ENABLED ? CONTINUITY_UNIT : null, continuity_context: MEDIA_CONTINUITY_ENABLED ? 131072 : 0, comfy: "model-fleet-comfy.service" },
  idle: { role: "idle" },
};
const legacyProfileNames = ["qwen27-long", "qwen27-multi", "qwen35-long", "qwen35-multi", "g9v3", "media", "audio", "idle"];
const configuredProfileNames = (process.env.MODEL_FLEET_AVAILABLE_PROFILES || legacyProfileNames.join(",")).split(",").map((name) => name.trim()).filter(Boolean);
const unknownProfileNames = configuredProfileNames.filter((name) => !profileCatalog[name]);
if (unknownProfileNames.length) throw new Error(`Unknown MODEL_FLEET_AVAILABLE_PROFILES: ${unknownProfileNames.join(", ")}`);
const profiles = Object.fromEntries(configuredProfileNames.map((name) => [name, profileCatalog[name]]));
const externallyLeasedTargets = new Set(["media", "audio", "g9v3"]);
const mainUnits = Object.values(profiles).filter((p) => p.unit).map((p) => p.unit);
const allUnits = [...new Set([...mainUnits, CONTINUITY_UNIT, "model-fleet-proxy.service", "model-fleet-comfy.service"])];
let transitionQueue = Promise.resolve();
let activeLease = null;
let externalLease = null;
let externalLeaseLoaded = false;
let externalLeaseRenewal = null;
let externalLeaseRenewing = false;
let stickyPreemptRequested = false;
let stickyReconcileTimer = null;
let stickyReconcileRunning = false;

function currentLease() {
  if (activeLease && Date.now() >= activeLease.expires_at_ms) activeLease = null;
  return activeLease;
}

function assertLease(owner) {
  const lease = currentLease();
  if (lease && lease.owner !== owner) {
    const error = new Error(`GPU fleet is leased for ${lease.purpose} until ${lease.expires_at}`);
    error.code = "LEASED";
    throw error;
  }
}

function acquireLease(owner, purpose = "media generation", ttlMs = 10800000) {
  if (!owner) throw new Error("lease owner is required");
  const lease = currentLease();
  if (lease && lease.owner !== owner) {
    const error = new Error(`GPU fleet is leased for ${lease.purpose} until ${lease.expires_at}`);
    error.code = "LEASED";
    throw error;
  }
  const boundedTtl = Math.max(60000, Math.min(Number(ttlMs) || 10800000, 21600000));
  activeLease = { owner, purpose, acquired_at: new Date().toISOString(), expires_at: new Date(Date.now() + boundedTtl).toISOString(), expires_at_ms: Date.now() + boundedTtl };
  return activeLease;
}

function releaseLease(owner) {
  const lease = currentLease();
  if (!lease) return null;
  if (lease.owner !== owner) {
    const error = new Error("lease is owned by another client");
    error.code = "LEASED";
    throw error;
  }
  activeLease = null;
  return null;
}

function externalLeaseConfigured() {
  return Boolean(EXTERNAL_LEASE_ACQUIRE_URL && EXTERNAL_LEASE_RENEW_URL && EXTERNAL_LEASE_RELEASE_URL);
}

async function persistExternalLease() {
  await mkdir(dirname(EXTERNAL_LEASE_STATE_PATH), { recursive: true });
  if (!externalLease) {
    try { await unlink(EXTERNAL_LEASE_STATE_PATH); }
    catch (error) { if (error.code !== "ENOENT") throw error; }
    return;
  }
  const temporary = `${EXTERNAL_LEASE_STATE_PATH}.tmp-${process.pid}`;
  await writeFile(temporary, `${JSON.stringify(externalLease, null, 2)}\n`, { mode: 0o600 });
  await rename(temporary, EXTERNAL_LEASE_STATE_PATH);
}

async function loadExternalLease() {
  if (externalLeaseLoaded) return externalLease;
  externalLeaseLoaded = true;
  try { externalLease = JSON.parse(await readFile(EXTERNAL_LEASE_STATE_PATH, "utf8")); }
  catch (error) { if (error.code !== "ENOENT") throw error; }
  if (externalLease?.token) scheduleExternalLeaseRenewal();
  return externalLease;
}

async function externalLeaseRequest(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(EXTERNAL_LEASE_REQUEST_TIMEOUT_MS),
  });
  const value = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(value.error || value.message || `external residency lease HTTP ${response.status}`);
  return value;
}

function scheduleExternalLeaseRenewal() {
  if (externalLeaseRenewal || !externalLeaseConfigured()) return;
  externalLeaseRenewal = setInterval(() => {
    void renewExternalLease().catch(async (error) => {
      if (!externalLease) return;
      externalLease.last_renewal_error = error.message;
      externalLease.last_renewal_error_at = new Date().toISOString();
      try { await persistExternalLease(); } catch {}
      console.error(`external residency lease renewal failed: ${error.message}`);
    });
  }, EXTERNAL_LEASE_RENEW_INTERVAL_MS);
  externalLeaseRenewal.unref?.();
}

function stopExternalLeaseRenewal() {
  if (externalLeaseRenewal) clearInterval(externalLeaseRenewal);
  externalLeaseRenewal = null;
}

async function acquireExternalLease(target, reason) {
  if (!externalLeaseConfigured()) return null;
  await loadExternalLease();
  if (externalLease?.token) return externalLease;
  const preemptible = STICKY_MEDIA_ENABLED && target === "media";
  const value = await externalLeaseRequest(EXTERNAL_LEASE_ACQUIRE_URL, { purpose: `model-fleet:${target}:${reason}`, ttl_ms: EXTERNAL_LEASE_TTL_MS, preemptible });
  if (!value.token) throw new Error("external residency lease acquire response did not include token");
  externalLease = {
    token: value.token,
    purpose: `model-fleet:${target}:${reason}`,
    acquired_at: new Date().toISOString(),
    expires_at: value.expires_at || null,
    last_renewed_at: null,
    last_renewal_error: null,
    preemptible,
  };
  await persistExternalLease();
  scheduleExternalLeaseRenewal();
  return externalLease;
}

async function renewExternalLease() {
  if (!externalLeaseConfigured() || externalLeaseRenewing) return externalLease;
  await loadExternalLease();
  if (!externalLease?.token) return null;
  externalLeaseRenewing = true;
  try {
    const value = await externalLeaseRequest(EXTERNAL_LEASE_RENEW_URL, { token: externalLease.token, ttl_ms: EXTERNAL_LEASE_TTL_MS });
    externalLease.expires_at = value.expires_at || externalLease.expires_at;
    externalLease.last_renewed_at = new Date().toISOString();
    externalLease.last_renewal_error = null;
    delete externalLease.last_renewal_error_at;
    await persistExternalLease();
    if (STICKY_MEDIA_ENABLED && value.preempt_requested === true) {
      stickyPreemptRequested = true;
      scheduleStickyReconcile(0);
    }
    return externalLease;
  } finally {
    externalLeaseRenewing = false;
  }
}

async function releaseExternalLease() {
  if (!externalLeaseConfigured()) return null;
  await loadExternalLease();
  if (!externalLease?.token) return null;
  stopExternalLeaseRenewal();
  while (externalLeaseRenewing) await new Promise((resolvePromise) => setTimeout(resolvePromise, 10));
  try {
    const value = await externalLeaseRequest(EXTERNAL_LEASE_RELEASE_URL, { token: externalLease.token });
    if (value.released !== true) throw new Error("external residency lease release response did not confirm released=true");
    externalLease = null;
    await persistExternalLease();
    return null;
  } catch (error) {
    scheduleExternalLeaseRenewal();
    throw error;
  }
}

function externalLeaseSummary() {
  return {
    configured: externalLeaseConfigured(),
    active: Boolean(externalLease?.token),
    purpose: externalLease?.purpose || null,
    acquired_at: externalLease?.acquired_at || null,
    expires_at: externalLease?.expires_at || null,
    last_renewed_at: externalLease?.last_renewed_at || null,
    last_renewal_error: externalLease?.last_renewal_error || null,
  };
}

async function systemctl(...args) {
  const scope = SYSTEMCTL_SCOPE === "user" ? ["--user"] : [];
  const { stdout } = await execFileAsync("systemctl", [...scope, ...args], { timeout: 180000 });
  return stdout.trim();
}

async function active(unit) {
  try { return (await systemctl("is-active", unit)) === "active"; }
  catch { return false; }
}

async function stop(unit) {
  if (await active(unit)) await systemctl("stop", unit);
}

async function start(unit) {
  if (!(await active(unit))) await systemctl("start", unit);
}

async function waitHttp(url, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let lastError = "not ready";
  let consecutive = 0;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(3000) });
      if (response.ok) {
        consecutive += 1;
        if (consecutive >= 3) return;
        await new Promise((resolve) => setTimeout(resolve, 1000));
        continue;
      }
      consecutive = 0;
      lastError = `HTTP ${response.status}`;
    } catch (error) { consecutive = 0; lastError = error.message; }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error(`${url} did not become ready: ${lastError}`);
}

async function saveState(state) {
  await mkdir(DATA_ROOT, { recursive: true });
  await writeFile(STATE_PATH, `${JSON.stringify(state, null, 2)}\n`);
}

async function loadState() {
  try { return JSON.parse(await readFile(STATE_PATH, "utf8")); }
  catch { return { target: "idle", phase: "unknown", updated_at: null }; }
}

async function gpuStatus(selector = GPU_SELECTOR) {
  try {
    const { stdout } = await execFileAsync("nvidia-smi", ["--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits", "-i", selector]);
    const [index, uuid, name, total, used, free, utilization] = stdout.trim().split(",").map((v) => v.trim());
    return { index: Number(index), uuid, name, memory_total_mib: Number(total), memory_used_mib: Number(used), memory_free_mib: Number(free), utilization_percent: Number(utilization) };
  } catch (error) { return { error: error.message }; }
}

async function status() {
  const state = await loadState();
  await loadExternalLease();
  const units = {};
  for (const unit of allUnits) units[unit] = await active(unit);
  const live = {
    main: await httpReady(TEXT_URL),
    continuity: MEDIA_CONTINUITY_ENABLED ? await httpReady(CONTINUITY_URL) : false,
    comfy: await httpReady(COMFY_URL),
  };
  const workersLive = mediaWorkersLive(live);
  const targetRole = profiles[state.target]?.role;
  const stale = state.phase === "ready" && state.target !== "idle" && (
    ((state.target === "media" || state.target === "audio") && !workersLive)
    || (targetRole === "continuity" && !live.continuity)
    || (targetRole === "main" && !live.main)
  );
  const phase = stale ? "stale" : state.phase;
  return {
    ...state,
    phase,
    lease: currentLease(),
    external_residency_lease: externalLeaseSummary(),
    sticky_media: {
      enabled: STICKY_MEDIA_ENABLED,
      desired_target: STICKY_MEDIA_ENABLED ? "media" : null,
      preempt_requested: stickyPreemptRequested,
    },
    gpu: await gpuStatus(),
    encoder_gpu: ENCODER_GPU_SELECTOR ? await gpuStatus(ENCODER_GPU_SELECTOR) : null,
    units,
    live,
    endpoints: {
      main: `http://${PUBLIC_HOST}:8091`,
      continuity: MEDIA_CONTINUITY_ENABLED ? `http://${PUBLIC_HOST}:8092` : null,
      comfy: `http://${PUBLIC_HOST}:8190`,
      api: `http://${PUBLIC_HOST}:8200/v1`,
    },
    profiles,
  };
}

function mediaWorkersLive(live) {
  if (!live.comfy) return false;
  if (MEDIA_CONTINUITY_ENABLED && !live.continuity) return false;
  return true;
}

async function httpReady(url) {
  try { return (await fetch(url, { signal: AbortSignal.timeout(1500) })).ok; }
  catch { return false; }
}

async function waitComfyPrompt(baseUrl, workflow, label, outputNode = null) {
  const submitted = await fetch(`${baseUrl}/prompt`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ prompt: workflow, client_id: `model-fleet-preload-${process.pid}` }),
    signal: AbortSignal.timeout(10000),
  });
  if (!submitted.ok) throw new Error(`${label} prompt rejected: ${await submitted.text()}`);
  const value = await submitted.json();
  if (!value.prompt_id) throw new Error(`${label} response did not include prompt_id`);
  const deadline = Date.now() + MEDIA_PRELOAD_TIMEOUT_MS;
  while (Date.now() < deadline) {
    const response = await fetch(`${baseUrl}/history/${encodeURIComponent(value.prompt_id)}`, { signal: AbortSignal.timeout(5000) });
    if (response.ok) {
      const history = await response.json();
      const record = history[value.prompt_id];
      if (record?.status?.status_str === "error") {
        const detail = record?.status?.messages?.find((row) => row[0] === "execution_error")?.[1]?.exception_message;
        throw new Error(`${label} failed: ${detail || "ComfyUI execution error"}`);
      }
      if (record?.status?.completed) return outputNode ? record.outputs?.[outputNode] || { status: "resident" } : { status: "resident" };
    }
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 500));
  }
  throw new Error(`${label} ${value.prompt_id} timed out`);
}

async function preloadMediaResidency() {
  if (!MEDIA_PRELOAD_ENABLED) return null;
  const workflow = {
    "1": { class_type: "UnetLoaderGGUF", inputs: { unet_name: MEDIA_PRELOAD_MODEL } },
    "2": { class_type: "ModelFleetCLIPLoaderGGUF", inputs: { clip_name: MEDIA_PRELOAD_ENCODER, type: "minimax", encoder_device: ENCODER_DEVICE } },
    "3": { class_type: "ModelFleetPreloadMedia", inputs: { model: ["1", 0], clip: ["2", 0], video_vae: ["4", 0], audio_vae: ["5", 0] } },
    "4": { class_type: "ModelFleetVAELoader", inputs: { vae_name: MEDIA_PRELOAD_VIDEO_VAE, vae_device: VAE_DEVICE } },
    "5": { class_type: "ModelFleetVAELoader", inputs: { vae_name: MEDIA_PRELOAD_AUDIO_VAE, vae_device: VAE_DEVICE } },
  };
  return waitComfyPrompt("http://127.0.0.1:8190", workflow, "media preload", "3");
}

async function startMediaWorkers() {
  await stop("model-fleet-proxy.service");
  if (MEDIA_CONTINUITY_ENABLED) {
    await start(CONTINUITY_UNIT);
    await waitHttp(CONTINUITY_URL, 900000);
  } else {
    await stop(CONTINUITY_UNIT);
  }
  await start("model-fleet-comfy.service");
  await waitHttp(COMFY_URL, 300000);
  await preloadMediaResidency();
}

async function transition(target, reason = "unspecified", owner = null) {
  if (!profiles[target]) throw new Error(`unknown or unavailable target ${target}`);
  assertLease(owner);
  if (STICKY_MEDIA_ENABLED && target === "idle" && owner && !stickyPreemptRequested) {
    const current = await loadState();
    if (current.target === "media" && current.phase === "ready") {
      if (MEDIA_PRELOAD_ENABLED) await preloadMediaResidency();
      return status();
    }
  }
  const startedAt = new Date().toISOString();
  await saveState({ target, phase: "transitioning", reason, started_at: startedAt, updated_at: startedAt });
  let externalLeaseHeld = false;
  try {
    if (externallyLeasedTargets.has(target)) {
      externalLeaseHeld = externalLeaseConfigured();
      await acquireExternalLease(target, reason);
    }
    if (target === "media" || target === "audio") {
      for (const unit of mainUnits) await stop(unit);
      await startMediaWorkers();
    } else if (profiles[target].role === "main") {
      await stop("model-fleet-comfy.service");
      await stop(CONTINUITY_UNIT);
      await stop("model-fleet-proxy.service");
      for (const unit of mainUnits) if (unit !== profiles[target].unit) await stop(unit);
      await start(profiles[target].unit);
      await waitHttp(TEXT_URL, 600000);
    } else if (profiles[target].role === "continuity") {
      await stop("model-fleet-comfy.service");
      await stop(CONTINUITY_UNIT);
      await stop("model-fleet-proxy.service");
      for (const unit of mainUnits) if (unit !== profiles[target].unit) await stop(unit);
      await start(profiles[target].unit);
      await waitHttp(CONTINUITY_URL, 900000);
    } else {
      for (const unit of allUnits) await stop(unit);
      if (target === "idle") await releaseExternalLease();
    }
    const completed = { target, phase: "ready", reason, started_at: startedAt, updated_at: new Date().toISOString() };
    await saveState(completed);
    return status();
  } catch (error) {
    let recoveryError = null;
    if (externalLeaseHeld) {
      try {
        for (const unit of allUnits) await stop(unit);
        await releaseExternalLease();
      } catch (recovery) { recoveryError = recovery.message; }
    }
    await saveState({ target, phase: "error", reason, error: error.message, recovery_error: recoveryError, started_at: startedAt, updated_at: new Date().toISOString() });
    throw error;
  }
}

function enqueueTransition(target, reason, owner = null) {
  transitionQueue = transitionQueue.then(
    () => transition(target, reason, owner),
    () => transition(target, reason, owner),
  );
  return transitionQueue;
}

function scheduleStickyReconcile(delayMs = STICKY_MEDIA_RETRY_MS) {
  if (!STICKY_MEDIA_ENABLED || stickyReconcileTimer) return;
  stickyReconcileTimer = setTimeout(() => {
    stickyReconcileTimer = null;
    void reconcileStickyMedia();
  }, Math.max(0, delayMs));
  stickyReconcileTimer.unref?.();
}

async function reconcileStickyMedia() {
  if (!STICKY_MEDIA_ENABLED || stickyReconcileRunning) return;
  stickyReconcileRunning = true;
  try {
    if (currentLease()) return;
    await loadExternalLease();
    if (stickyPreemptRequested) {
      const state = await loadState();
      if (state.target !== "idle" || externalLease?.token) {
        await enqueueTransition("idle", "sticky media yielded to DeepSeek");
      }
      stickyPreemptRequested = false;
    }
    const state = await loadState();
    await loadExternalLease();
    const leaseOk = !externalLeaseConfigured() || Boolean(externalLease?.token);
    const continuityOk = !MEDIA_CONTINUITY_ENABLED || (await active(CONTINUITY_UNIT) && await httpReady(CONTINUITY_URL));
    const ready = state.target === "media"
      && state.phase === "ready"
      && leaseOk
      && continuityOk
      && await active("model-fleet-comfy.service")
      && await httpReady(COMFY_URL);
    if (!ready) await enqueueTransition("media", "sticky media residency");
  } catch (error) {
    console.error(`sticky media reconciliation failed: ${error.message}`);
  } finally {
    stickyReconcileRunning = false;
    scheduleStickyReconcile();
  }
}

function json(response, code, value) {
  const body = JSON.stringify(value, null, 2);
  response.writeHead(code, { "content-type": "application/json", "content-length": Buffer.byteLength(body) });
  response.end(body);
}

async function body(request) {
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  return chunks.length ? JSON.parse(Buffer.concat(chunks).toString("utf8")) : {};
}

http.createServer(async (request, response) => {
  try {
    const url = new URL(request.url, `http://${request.headers.host || "localhost"}`);
    if (request.method === "GET" && request.url === "/health") return json(response, 200, { status: "ok" });
    if (request.method === "GET" && request.url === "/status") return json(response, 200, await status());
    if (request.method === "GET" && request.url === "/profiles") return json(response, 200, profiles);
    if (request.method === "POST" && url.pathname === "/lease/acquire") {
      const input = await body(request);
      return json(response, 200, acquireLease(input.owner, input.purpose, input.ttl_ms));
    }
    if (request.method === "POST" && url.pathname === "/lease/release") {
      const input = await body(request);
      return json(response, 200, { lease: releaseLease(input.owner) });
    }
    if (request.method === "GET" && url.pathname === "/view") {
      const filename = url.searchParams.get("filename") || "";
      const subfolder = url.searchParams.get("subfolder") || "";
      const filePath = resolve(OUTPUT_ROOT, subfolder, filename);
      const rootPrefix = `${resolve(OUTPUT_ROOT)}/`;
      if (!filename || (!filePath.startsWith(rootPrefix) && filePath !== resolve(OUTPUT_ROOT))) return json(response, 400, { error: "invalid output path" });
      const info = await stat(filePath);
      const mime = { ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".mp4": "video/mp4", ".webm": "video/webm", ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac" }[extname(filePath).toLowerCase()] || "application/octet-stream";
      response.writeHead(200, { "content-type": mime, "content-length": info.size, "cache-control": "private, max-age=3600" });
      return createReadStream(filePath).pipe(response);
    }
    if (request.method === "POST" && url.pathname === "/transition") {
      const input = await body(request);
      return json(response, 200, await enqueueTransition(input.target, input.reason, input.owner));
    }
    return json(response, 404, { error: "not found" });
  } catch (error) {
    return json(response, error.code === "LEASED" ? 409 : 500, { error: error.message, code: error.code, stack: error.stack });
  }
}).listen(PORT, HOST, () => {
  console.log(`model-fleet manager listening on ${HOST}:${PORT}`);
  scheduleStickyReconcile(0);
});
