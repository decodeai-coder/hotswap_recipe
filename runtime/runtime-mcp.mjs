#!/usr/bin/env node
import { createInterface } from "node:readline";

const API = process.env.MODEL_FLEET_URL || "http://127.0.0.1:8199";
const legacyProfiles = ["qwen27-long", "qwen27-multi", "qwen35-long", "qwen35-multi", "g9v3", "media", "audio", "idle"];
const availableProfiles = (process.env.MODEL_FLEET_AVAILABLE_PROFILES || legacyProfiles.join(",")).split(",").map((name) => name.trim()).filter(Boolean);
const tools = [
  { name: "get_runtime_status", description: "Show the selected GPU's model-residency state, active services, endpoints, and VRAM usage.", inputSchema: { type: "object", additionalProperties: false, properties: {} } },
  { name: "list_runtime_profiles", description: "List coding, long-context, multi-user, media, and idle residency profiles.", inputSchema: { type: "object", additionalProperties: false, properties: {} } },
  { name: "activate_runtime_profile", description: "Exclusively activate one selected-GPU profile. Media and audio start ComfyUI for image, video, speech, and transcription loads. A G9v3 continuity sidecar is started only when MODEL_FLEET_MEDIA_CONTINUITY is enabled.", inputSchema: { type: "object", additionalProperties: false, required: ["target"], properties: { target: { type: "string", enum: availableProfiles }, reason: { type: "string" } } } },
];

async function request(path, options) {
  const response = await fetch(`${API}${path}`, options);
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
  return value;
}
const result = (value) => ({ content: [{ type: "text", text: JSON.stringify(value, null, 2) }], structuredContent: value });

async function handle(message) {
  if (message.method === "initialize") return { protocolVersion: message.params?.protocolVersion || "2025-06-18", capabilities: { tools: {} }, serverInfo: { name: "model-fleet-runtime", version: "0.1.0" } };
  if (message.method === "tools/list") return { tools };
  if (message.method === "tools/call") {
    const { name, arguments: args = {} } = message.params || {};
    if (name === "get_runtime_status") return result(await request("/status"));
    if (name === "list_runtime_profiles") return result(await request("/profiles"));
    if (name === "activate_runtime_profile") return result(await request("/transition", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(args) }));
    throw new Error(`unknown tool ${name}`);
  }
  if (message.method?.startsWith("notifications/")) return undefined;
  if (message.method === "resources/list") return { resources: [] };
  if (message.method === "resources/templates/list") return { resourceTemplates: [] };
  if (message.method === "prompts/list") return { prompts: [] };
  throw new Error(`unsupported method ${message.method}`);
}

function send(value) { process.stdout.write(`${JSON.stringify(value)}\n`); }
const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
input.on("line", async (line) => {
  if (!line.trim()) return;
  let message;
  try {
    message = JSON.parse(line);
    const value = await handle(message);
    if (message.id !== undefined && value !== undefined) send({ jsonrpc: "2.0", id: message.id, result: value });
  } catch (error) {
    if (message?.id !== undefined) send({ jsonrpc: "2.0", id: message.id, error: { code: -32000, message: error.message } });
  }
});
