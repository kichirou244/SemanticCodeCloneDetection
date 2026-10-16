#!/usr/bin/env node
/**
 * Đẩy task trong tasks/tasks.json lên Notion.
 *
 * Trang đích: "Project Manager" (mặc định NOTION_PAGE_ID bên dưới).
 * Script sẽ tự dò database con (cái mẫu sẵn) trong trang rồi thêm task vào đó.
 *
 * Cần cung cấp (xem tasks/README.md và tasks/.env.example):
 *   NOTION_TOKEN       - Internal Integration Secret (secret_... hoặc ntn_...)
 *   NOTION_PAGE_ID     - id trang "Project Manager" (đã share cho integration)
 *   NOTION_DATABASE_ID - (tuỳ chọn) id database; nếu có thì bỏ qua bước dò trang
 *   NOTION_TITLE_PROP  - (tuỳ chọn) tên property title, mặc định tự dò theo kiểu
 *   NOTION_DATE_PROP   - (tuỳ chọn) mặc định tự dò
 *   NOTION_STATUS_PROP - (tuỳ chọn) mặc định tự dò
 *   NOTION_NOTES_PROP  - (tuỳ chọn) mặc định tự dò
 *   NOTION_VERSION     - (tuỳ chọn) mặc định "2022-06-28"
 *
 * Dùng:
 *   node tasks/notion.js --dry-run     xem trước, không gọi API
 *   node tasks/notion.js --schema      chỉ in ra schema + mapping rồi thoát
 *   node tasks/notion.js               tạo task thật
 *   node tasks/notion.js --file other.json
 */
"use strict";

const fs = require("node:fs");
const path = require("node:path");

const NOTION_API = "https://api.notion.com/v1";
// Trang "Project Manager" của bạn (lấy từ URL .../Project-Manager-<id>).
const DEFAULT_PAGE_ID = "";
const RATE_LIMIT_MS = 350; // Notion cho ~3 request/giây

function loadDotEnv(file) {
  try {
    const txt = fs.readFileSync(file, "utf8");
    for (const raw of txt.split(/\r?\n/)) {
      const line = raw.trim();
      if (!line || line.startsWith("#")) continue;
      const eq = line.indexOf("=");
      if (eq === -1) continue;
      const key = line.slice(0, eq).trim();
      let val = line.slice(eq + 1).trim();
      if (
        (val.startsWith('"') && val.endsWith('"')) ||
        (val.startsWith("'") && val.endsWith("'"))
      ) {
        val = val.slice(1, -1);
      }
      if (!(key in process.env)) process.env[key] = val;
    }
  } catch {
    /* không có .env thì thôi */
  }
}

function parseArgs(argv) {
  const args = {
    dryRun: false,
    schemaOnly: false,
    projectOnly: false,
    list: false,
    tasks: false,
    file: null,
    help: false,
    statusName: null,
    statusValue: null,
    archiveName: null,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--dry-run" || a === "-n") args.dryRun = true;
    else if (a === "--schema" || a === "-s") args.schemaOnly = true;
    else if (a === "--project" || a === "-p") args.projectOnly = true;
    else if (a === "--list" || a === "-l") args.list = true;
    else if (a === "--tasks" || a === "-t") args.tasks = true;
    else if (a === "--help" || a === "-h") args.help = true;
    else if (a === "--file" || a === "-f") args.file = argv[++i];
    else if (a === "--status") {
      args.statusName = argv[++i];
      args.statusValue = argv[++i];
    } else if (a === "--archive") {
      args.archiveName = argv[++i];
    }
  }
  return args;
}

// Nhận cả id trần (32 hex) lẫn URL dán vào, trả về dạng có gạch nối.
function normalizeId(id) {
  if (!id) return id;
  const m = String(id).match(/[0-9a-fA-F]{32}/);
  if (!m) return id;
  const h = m[0];
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
}

function truncate(s, n) {
  s = String(s ?? "");
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function notionFetch(pathname, { token, method = "GET", body } = {}) {
  const res = await fetch(`${NOTION_API}${pathname}`, {
    method,
    headers: {
      Authorization: `Bearer ${token}`,
      "Notion-Version": process.env.NOTION_VERSION || "2022-06-28",
      "Content-Type": "application/json",
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  let json;
  try {
    json = text ? JSON.parse(text) : {};
  } catch {
    json = { raw: text };
  }
  if (!res.ok) {
    const detail = json.message || json.raw || res.statusText;
    throw new Error(`Notion API ${res.status}: ${detail}`);
  }
  return json;
}

// Duyệt đệ quy các block trong trang để tìm database con đầu tiên.
async function findChildDatabase(blockId, token, seen = new Set()) {
  if (seen.has(blockId)) return null;
  seen.add(blockId);
  let cursor;
  do {
    const q = cursor ? `?start_cursor=${cursor}` : "";
    const res = await notionFetch(`/blocks/${blockId}/children${q}`, { token });
    for (const b of res.results || []) {
      if (b.type === "child_database") return b.id;
      if (b.has_children) {
        const found = await findChildDatabase(b.id, token, seen);
        if (found) return found;
      }
    }
    cursor = res.has_more ? res.next_cursor : null;
  } while (cursor);
  return null;
}

// Liệt kê mọi database / trang con dưới 1 block (để tìm đúng db tasks).
async function listDatabases(blockId, token, seen = new Set(), depth = 0) {
  if (seen.has(blockId)) return;
  seen.add(blockId);
  const pad = "  ".repeat(depth);
  let cursor;
  do {
    const q = cursor ? `?start_cursor=${cursor}` : "";
    const res = await notionFetch(`/blocks/${blockId}/children${q}`, { token });
    for (const b of res.results || []) {
      if (b.type === "child_database") {
        const title = (b.child_database && b.child_database.title) || "(không tên)";
        console.log(`${pad}- [db]   ${title}   ${normalizeId(b.id)}`);
      } else if (b.type === "child_page") {
        console.log(`${pad}- [page] ${b.child_page.title}   ${normalizeId(b.id)}`);
      }
      if (b.has_children) await listDatabases(b.id, token, seen, depth + 1);
    }
    cursor = res.has_more ? res.next_cursor : null;
  } while (cursor);
}

async function resolveDatabaseId(token) {
  if (process.env.NOTION_DATABASE_ID) return normalizeId(process.env.NOTION_DATABASE_ID);
  const pageId = normalizeId(process.env.NOTION_PAGE_ID || DEFAULT_PAGE_ID);
  console.log(`Chưa có NOTION_DATABASE_ID, dò database con trong trang ${pageId} ...`);
  const dbId = await findChildDatabase(pageId, token);
  if (!dbId) {
    throw new Error(
      "Không tìm thấy database con trong trang. Kiểm tra: (1) trang đã share cho integration chưa, " +
      "(2) đúng trang 'Project Manager' chưa, hoặc (3) set thẳng NOTION_DATABASE_ID."
    );
  }
  return normalizeId(dbId);
}

// Dò tên property theo kiểu dữ liệu; ưu tiên tên do env chỉ định.
function resolveProps(dbProps) {
  const names = Object.keys(dbProps || {});
  const fromEnv = (envName, types) => {
    const wanted = process.env[envName];
    if (wanted && dbProps[wanted] && types.includes(dbProps[wanted].type)) return wanted;
    return names.find((n) => types.includes(dbProps[n].type)) || null;
  };
  return {
    title: fromEnv("NOTION_TITLE_PROP", ["title"]),
    date: fromEnv("NOTION_DATE_PROP", ["date"]),
    status: fromEnv("NOTION_STATUS_PROP", ["status", "select"]),
    notes: fromEnv("NOTION_NOTES_PROP", ["rich_text"]),
  };
}

// Tên (title) của 1 page bất kỳ.
function pageTitle(page) {
  for (const v of Object.values(page.properties || {})) {
    if (v.type === "title") return (v.title || []).map((t) => t.plain_text).join("");
  }
  return "";
}

// Tìm project theo tên trong db projects, trả về page id (để gắn vào cột relation).
async function resolveProjectPageId(token) {
  const name = process.env.NOTION_PROJECT_NAME;
  if (!name) return null;
  const dbId = normalizeId(process.env.NOTION_PROJECTS_DATABASE_ID);
  if (!dbId) throw new Error("Có NOTION_PROJECT_NAME nhưng thiếu NOTION_PROJECTS_DATABASE_ID.");

  const db = await notionFetch(`/databases/${dbId}`, { token });
  const titleProp =
    (process.env.NOTION_PROJECT_TITLE_PROP && db.properties[process.env.NOTION_PROJECT_TITLE_PROP]
      ? process.env.NOTION_PROJECT_TITLE_PROP
      : Object.keys(db.properties || {}).find((n) => db.properties[n].type === "title"));
  if (!titleProp) throw new Error("Không tìm thấy property title trong db projects.");

  const res = await notionFetch(`/databases/${dbId}/query`, {
    token,
    method: "POST",
    body: {
      filter: { property: titleProp, title: { contains: name } },
      page_size: 10,
    },
  });
  const results = res.results || [];
  if (!results.length) {
    throw new Error(
      `Không tìm thấy project nào có tên chứa "${name}" trong db projects. Kiểm tra db/quyền share.`
    );
  }
  const exact = results.find((p) => pageTitle(p).trim().toLowerCase() === name.trim().toLowerCase());
  const pick = exact || results[0];
  if (!exact && results.length > 1) {
    console.log("Có nhiều project khớp, dùng cái đầu:");
    for (const r of results) console.log(`  - ${pageTitle(r)} (${r.id})`);
  }
  console.log(`Project: "${pageTitle(pick)}" -> ${pick.id}`);
  return pick.id;
}

// Tìm cột relation trong db tasks để gắn project.
function resolveRelationProp(dbProps, wanted) {
  if (wanted && dbProps[wanted] && dbProps[wanted].type === "relation") return wanted;
  return Object.keys(dbProps || {}).find((n) => dbProps[n].type === "relation") || null;
}

function buildProperties(task, props, dbProps) {
  const out = {};
  if (props.title) {
    out[props.title] = { title: [{ text: { content: truncate(task.title, 2000) } }] };
  }
  if (props.date && task.date) {
    out[props.date] = { date: { start: task.date } };
  }
  if (props.status && task.status) {
    const type = dbProps[props.status].type;
    out[props.status] =
      type === "status" ? { status: { name: task.status } } : { select: { name: task.status } };
  }
  if (props.notes && task.notes) {
    out[props.notes] = { rich_text: [{ text: { content: truncate(task.notes, 2000) } }] };
  }
  return out;
}

// Tìm task theo tên (ưu tiên khớp chính xác, không thì cái chứa tên).
async function findTaskByName(token, dbId, titleProp, name, { exact = false } = {}) {
  const filter = {
    property: titleProp,
    title: exact ? { equals: name } : { contains: name },
  };
  const res = await notionFetch(`/databases/${dbId}/query`, {
    token,
    method: "POST",
    body: { filter, page_size: 10 },
  });
  const results = res.results || [];
  if (!results.length) return null;
  return results.find((p) => pageTitle(p).trim() === String(name).trim()) || results[0];
}

function printSchema(db, props, relationProp) {
  console.log(`\nDatabase: ${(db.title || []).map((t) => t.plain_text).join("") || "(không tên)"}`);
  console.log(`ID: ${db.id}`);
  console.log("Property trong database:");
  for (const [name, def] of Object.entries(db.properties || {})) {
    let extra = "";
    if (def.type === "status") extra = ` [${(def.status.options || []).map((o) => o.name).join(", ")}]`;
    else if (def.type === "select") extra = ` [${(def.select.options || []).map((o) => o.name).join(", ")}]`;
    console.log(`  - ${name} (${def.type})${extra}`);
  }
  console.log("\nMapping sẽ dùng:");
  console.log(`  title    -> ${props.title || "(không thấy!)"}`);
  console.log(`  date     -> ${props.date || "(bỏ qua)"}`);
  console.log(`  status   -> ${props.status || "(bỏ qua)"}`);
  console.log(`  notes    -> ${props.notes || "(bỏ qua)"}`);
  console.log(`  relation -> ${relationProp || "(không có cột relation để gắn project)"}`);
  console.log("");
}

function printHelp() {
  console.log(`Đẩy task lên Notion.

  node tasks/notion.js --dry-run    xem trước (không gọi API)
  node tasks/notion.js --list       liệt kê database con trong trang Project Manager
  node tasks/notion.js --tasks      liệt kê task hiện có (ngày, status)
  node tasks/notion.js --schema     in schema + mapping rồi thoát
  node tasks/notion.js --project    chỉ tìm project theo tên rồi thoát
  node tasks/notion.js              sync task (tạo mới / cập nhật theo tên)
  node tasks/notion.js --status "<tên task>" "<trạng thái>"   đổi status 1 task
  node tasks/notion.js -f file.json dùng file task khác

Env cần: NOTION_TOKEN, và NOTION_PAGE_ID (mặc định ${DEFAULT_PAGE_ID})
hoặc NOTION_DATABASE_ID. Xem tasks/README.md.`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    printHelp();
    return;
  }

  loadDotEnv(path.join(__dirname, ".env"));

  const file = path.resolve(args.file || path.join(__dirname, "tasks.json"));
  const tasks = JSON.parse(fs.readFileSync(file, "utf8"));
  if (!Array.isArray(tasks) || tasks.length === 0) {
    throw new Error(`${file} rỗng hoặc không phải mảng task.`);
  }
  console.log(`Đọc ${tasks.length} task từ ${file}`);

  if (args.dryRun) {
    for (const t of tasks) console.log(`  [dry-run] ${t.date || "(no date)"} | ${t.title}`);
    return;
  }

  const token = process.env.NOTION_TOKEN;
  if (!token) throw new Error("Thiếu NOTION_TOKEN (xem tasks/.env.example).");

  if (args.list) {
    const pageId = normalizeId(process.env.NOTION_PAGE_ID || DEFAULT_PAGE_ID);
    console.log(`Liệt kê database/trang con trong ${pageId}:\n`);
    await listDatabases(pageId, token);
    return;
  }

  // Liệt kê task hiện có (để check/đổi status): --tasks
  if (args.tasks) {
    const dbId0 = await resolveDatabaseId(token);
    const db0 = await notionFetch(`/databases/${dbId0}`, { token });
    const props0 = resolveProps(db0.properties);
    const rows = [];
    let cursor;
    do {
      const body = { page_size: 100, ...(cursor ? { start_cursor: cursor } : {}) };
      const res = await notionFetch(`/databases/${dbId0}/query`, { token, method: "POST", body });
      rows.push(...(res.results || []));
      cursor = res.has_more ? res.next_cursor : null;
    } while (cursor);
    const val = (p, name) => {
      const prop = name && p.properties[name];
      if (!prop) return null;
      if (prop.type === "date") return prop.date && prop.date.start;
      if (prop.type === "status") return prop.status && prop.status.name;
      if (prop.type === "select") return prop.select && prop.select.name;
      return null;
    };
    rows.sort((a, b) => String(val(a, props0.date)).localeCompare(String(val(b, props0.date))));
    console.log(`Có ${rows.length} task:\n`);
    for (const p of rows) {
      const d = val(p, props0.date) || "—";
      const s = val(p, props0.status) || "—";
      console.log(`  ${d}  [${s}]  ${pageTitle(p)}`);
    }
    return;
  }

  // Đổi status 1 task theo tên: --status "<tên>" "<trạng thái>"
  if (args.statusName) {
    if (!args.statusValue) throw new Error('Dùng: --status "<tên task>" "<trạng thái>"');
    const dbId0 = await resolveDatabaseId(token);
    const db0 = await notionFetch(`/databases/${dbId0}`, { token });
    const props0 = resolveProps(db0.properties);
    if (!props0.status) throw new Error("Db tasks không có cột status.");
    const page = await findTaskByName(token, dbId0, props0.title, args.statusName);
    if (!page) throw new Error(`Không tìm thấy task tên chứa "${args.statusName}".`);
    const type = db0.properties[props0.status].type;
    const properties = {
      [props0.status]:
        type === "status" ? { status: { name: args.statusValue } } : { select: { name: args.statusValue } },
    };
    await notionFetch(`/pages/${page.id}`, { token, method: "PATCH", body: { properties } });
    console.log(`Đổi status "${pageTitle(page)}" -> ${args.statusValue}`);
    return;
  }

  // Archive 1 task theo tên: --archive "<tên>"
  if (args.archiveName) {
    const dbId0 = await resolveDatabaseId(token);
    const db0 = await notionFetch(`/databases/${dbId0}`, { token });
    const props0 = resolveProps(db0.properties);
    const page = await findTaskByName(token, dbId0, props0.title, args.archiveName);
    if (!page) throw new Error(`Không tìm thấy task tên chứa "${args.archiveName}".`);
    await notionFetch(`/pages/${page.id}`, { token, method: "PATCH", body: { archived: true } });
    console.log(`Đã archive "${pageTitle(page)}"`);
    return;
  }

  // Tìm project theo tên (nếu có) trước khi làm gì khác.
  const projectPageId = await resolveProjectPageId(token);
  if (args.projectOnly) return;

  const dbId = await resolveDatabaseId(token);
  const db = await notionFetch(`/databases/${dbId}`, { token });
  const props = resolveProps(db.properties);
  const relationProp = projectPageId
    ? resolveRelationProp(db.properties, process.env.NOTION_PROJECT_PROP)
    : null;
  printSchema(db, props, relationProp);

  if (args.schemaOnly) return;
  if (!props.title) throw new Error("Database không có property kiểu title, không tạo được.");
  if (projectPageId && !relationProp) {
    console.warn("Cảnh báo: db tasks không có cột relation -> task không được gắn project.");
  }

  // Sync: task đã tồn tại (cùng tên) thì cập nhật, chưa có thì tạo mới.
  let created = 0;
  let updated = 0;
  for (const task of tasks) {
    const properties = buildProperties(task, props, db.properties);
    if (relationProp && projectPageId) {
      properties[relationProp] = { relation: [{ id: projectPageId }] };
    }
    try {
      const existing = await findTaskByName(token, dbId, props.title, task.title, { exact: true });
      if (existing) {
        await notionFetch(`/pages/${existing.id}`, {
          token,
          method: "PATCH",
          body: { properties },
        });
        updated++;
        console.log(`  ~ ${task.title} (cập nhật)`);
      } else {
        const body = { parent: { database_id: dbId }, properties };
        const page = await notionFetch("/pages", { token, method: "POST", body });
        created++;
        console.log(`  + ${task.title} -> ${page.url || page.id}`);
      }
    } catch (e) {
      console.error(`  ! ${task.title}: ${e.message}`);
    }
    await sleep(RATE_LIMIT_MS);
  }
  console.log(`\nXong: tạo mới ${created}, cập nhật ${updated} (tổng ${tasks.length}).`);
}

main().catch((e) => {
  console.error(`\nLỗi: ${e.message}`);
  // Dùng exitCode thay vì process.exit() để tránh lỗi libuv assertion trên Windows.
  process.exitCode = 1;
});
