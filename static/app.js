"use strict";
/* 阅卷助手前端：无构建步骤的原生 JS。 */

// ------------------------------------------------------------------ 工具
const $ = (s, el = document) => el.querySelector(s);
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "value") el.value = v;
    else if (k === "checked") el.checked = !!v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}
function fill(el, ...kids) {   // 和 replaceChildren 一样，但会跳过 null/false（否则会显示成文字 "null"）
  el.replaceChildren(...kids.flat(Infinity).filter(k => k !== null && k !== undefined && k !== false));
  return el;
}
const svgNS = "http://www.w3.org/2000/svg";
function s(tag, attrs, ...kids) {
  const el = document.createElementNS(svgNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "class") el.setAttribute("class", v);
    else el.setAttribute(k, v);
  }
  kids.flat().forEach(k => k !== null && k !== undefined && el.append(k instanceof Node ? k : document.createTextNode(String(k))));
  return el;
}
async function api(method, url, body, form) {
  const opt = { method, headers: {} };
  if (form) opt.body = body;
  else if (body !== undefined) { opt.body = JSON.stringify(body); opt.headers["Content-Type"] = "application/json"; }
  const r = await fetch(url, opt);
  const text = await r.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!r.ok) {
    let msg = (data && data.detail) || text || r.statusText;
    if (Array.isArray(msg)) msg = msg.map(d => d.msg).join("；");
    throw new Error(msg);
  }
  return data;
}
function upload(url, fd, onprog) {   // 和 api() 一样，但能显示上传进度（几百 MB 的扫描件要传一会儿）
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.open("POST", url);
    x.upload.onprogress = ev => { if (ev.lengthComputable && onprog) onprog(ev.loaded, ev.total); };
    x.onload = () => {
      let data = null;
      try { data = x.responseText ? JSON.parse(x.responseText) : null; } catch { data = x.responseText; }
      if (x.status >= 200 && x.status < 300) return resolve(data);
      let msg = (data && data.detail) || x.responseText || x.statusText;
      if (Array.isArray(msg)) msg = msg.map(d => d.msg).join("；");
      reject(new Error(msg));
    };
    x.onerror = () => reject(new Error("上传失败：网络断了，或者阅卷助手没在运行"));
    x.send(fd);
  });
}
function mb(bytes) { return bytes >= 1048576 ? `${(bytes / 1048576).toFixed(bytes >= 1048576 * 100 ? 0 : 1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`; }
let toastTimer;
function toast(msg, err) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "show" + (err ? " err" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.className = ""), err ? 6000 : 2600);
}
function fileUrl(p) { return `/api/exams/${state.exam.id}/file?p=${encodeURIComponent(p)}&t=${state.bust}`; }
function fmt(x) { return x === null || x === undefined || x === "" ? "–" : (+x).toString(); }
function renderMath(el) {
  if (window.renderMathInElement) {
    try {
      window.renderMathInElement(el, { delimiters: [{ left: "$$", right: "$$", display: true }, { left: "$", right: "$", display: false }], throwOnError: false });
    } catch (e) { /* 公式有错就显示原文 */ }
  }
  return el;
}
function mathBlock(text, cls) {
  const el = h("div", { class: cls || "render" });
  const src = String(text || "");
  const re = /~~([\s\S]+?)~~/g;
  let last = 0, m;
  while ((m = re.exec(src))) {
    el.append(src.slice(last, m.index));
    el.append(h("del", { title: "学生划掉的内容" }, m[1]));
    last = re.lastIndex;
  }
  el.append(src.slice(last));
  return renderMath(el);
}
function lightbox(src) {
  const lb = h("div", { class: "lightbox", onclick: () => lb.remove() }, h("img", { src, alt: "放大查看" }));
  document.body.append(lb);
}
function textBox(title, text) {   // 没有原题截图时（扫描件、照片录入的），看录入的题目文字
  const lb = h("div", { class: "lightbox", onclick: () => lb.remove() },
    h("div", { class: "sheet textbox", onclick: ev => ev.stopPropagation() }, h("h3", { style: "margin-bottom:8px" }, title), mathBlock(text),
      h("div", { class: "row", style: "margin-top:12px" }, h("button", { onclick: () => lb.remove() }, "关闭"))));
  document.body.append(lb);
}
const KINDS = ["解答题", "填空题", "选择题", "学生信息", "其他"];
const STRUCT_EXAMPLE = "1-8 单选 5分\n9-11 多选 6分\n12-14 填空 5分\n15 解答 13分 2问\n16-17 解答 15分";
const STATE_TEXT = { ok: "AI", review: "待复核", teacher: "已改", error: "失败", pending: "排队", grading: "评分中", transcribing: "识别中", align_failed: "对齐失败", missing: "缺页" };
// 考试类型：平时作业不打分，每题只判 对 / 半对 / 错（按 1 / 0.5 / 0 存）
const KIND_TEXT = { exam: "考试", quiz: "周测", homework: "平时作业" };
const KIND_HINT = { exam: "月考、期中、期末：按评分细则打分", quiz: "按分值打分", homework: "不打分：只判对错、指出错在哪" };
const isHW = () => !!state.exam && state.exam.kind === "homework";
const verdictOf = v => v == null ? null : +v >= 1 - 1e-9 ? "对" : +v <= 0 ? "错" : "半对";
const MARK = { 对: "✓", 半对: "½", 错: "✗" };
const VCLS = { 对: "ok", 半对: "half", 错: "bad" };

// ------------------------------------------------------------------ 状态与路由
const state = { exam: null, qs: null, tmpl: null, layoutSel: null, drawing: false, results: null, filter: "all", sid: null, num: null, bust: Date.now(), poll: null, settings: null };

window.addEventListener("hashchange", route);
window.addEventListener("DOMContentLoaded", route);
document.addEventListener("keydown", onKey);

function setNav(which) {
  document.querySelectorAll("[data-nav]").forEach(a => a.classList.toggle("on", a.dataset.nav === which));
}
async function route() {
  clearInterval(state.poll);
  const hash = location.hash.slice(1) || "/";
  const parts = hash.split("/").filter(Boolean);
  const app = $("#app");
  app.className = "";
  try {
    if (parts[0] === "settings") { setNav("settings"); await viewSettings(app); }
    else if (parts[0] === "roster") { setNav("roster"); await viewRoster(app); }
    else if (parts[0] === "new") { setNav("exams"); await viewNew(app); }
    else if (parts[0] === "exam" && parts[1]) { setNav("exams"); await viewExam(app, parts[1], parts[2] || "questions", parts[3]); }
    else { setNav("exams"); await viewExams(app); }
  } catch (e) {
    fill(app, h("div", { class: "notice err" }, "加载失败：" + e.message));
  }
}

// ------------------------------------------------------------------ 考试列表
async function viewExams(app) {
  const list = await api("GET", "/api/exams");
  const head = h("div", { class: "row spread", style: "margin-bottom:16px" }, h("h1", {}, "考试"),
    h("a", { class: "btn primary", href: "#/new" }, "新建考试"));
  if (!list.length) {
    fill(app, head, h("div", { class: "sheet empty" }, h("h2", {}, "还没有考试"),
      h("p", {}, "新建一场考试：上传试卷和答题纸，程序会录入题目、识别答题纸上的题框；之后上传学生答题纸的扫描件就能批改解答题。"),
      h("a", { class: "btn primary", href: "#/new" }, "新建考试")));
    return;
  }
  const rows = list.map(e => h("tr", { class: "click", onclick: () => (location.hash = `#/exam/${e.id}`) },
    h("td", {}, h("b", {}, e.title)), h("td", {}, h("span", { class: "badge kind-" + (e.kind || "exam") }, KIND_TEXT[e.kind] || "考试")),
    h("td", { class: "muted num" }, e.created_at),
    h("td", { class: "num" }, e.n_questions), h("td", { class: "num" }, e.n_sheets)));
  fill(app, head, h("div", { class: "sheet" }, h("table", { class: "list" },
    h("thead", {}, h("tr", {}, h("th", {}, "名称"), h("th", {}, "类型"), h("th", {}, "创建时间"), h("th", {}, "解答题"), h("th", {}, "已批答卷"))),
    h("tbody", {}, rows))));
}

// ------------------------------------------------------------------ 班级名单
async function viewRoster(app) {
  const list = await api("GET", "/api/rosters");
  const form = h("form", { class: "stack", onsubmit: async ev => {
      ev.preventDefault();
      const fd = new FormData(form);
      if (!form.text.value.trim() && !form.files.files.length) { toast("粘贴名单，或者选择名单文件", true); return; }
      try {
        const r = await upload("/api/rosters/import", fd);
        toast(`已导入：${r.classes.map(c => `${c.name} ${c.count} 人`).join("，")}` + (r.bad.length ? `；有 ${r.bad.length} 行没看懂` : ""), !!r.bad.length);
        viewRoster(app);
      } catch (er) { toast(er.message, true); }
    } },
    h("div", { class: "grid2" },
      h("label", { class: "field" }, h("span", {}, "班级名称"), h("input", { type: "text", name: "name", placeholder: "例如：高一5班（名单里有“班级”一列时可以不填）" })),
      h("label", { class: "field" }, h("span", {}, "导入方式"), h("select", { name: "mode" },
        h("option", { value: "replace" }, "替换同名班级的名单"), h("option", { value: "append" }, "追加到同名班级")))),
    h("label", { class: "field" }, h("span", {}, "粘贴名单（从 Excel 里选中“学号、姓名”两列复制过来就行）"),
      h("textarea", { name: "text", rows: 6, placeholder: "学号\t姓名\n2501\t张三\n2502\t李四" })),
    h("label", { class: "field" }, h("span", {}, "或者上传名单文件"), h("input", { type: "file", name: "files", accept: ".xlsx,.csv,.txt" }),
      h("div", { class: "hint" }, "Excel（.xlsx）、CSV、TXT 都行。第一行是表头时按“学号/考号/序号”“姓名”“班级”找列；有“班级”一列的会按班级分开。")),
    h("div", {}, h("button", { class: "primary", type: "submit" }, "导入名单")));
  const cards = list.map(c => {
    const box = h("div", { class: "sheet stack" });
    const draw = () => fill(box,
      h("div", { class: "row spread" }, h("div", {}, h("b", {}, c.name), h("span", { class: "muted" }, `　${c.count} 人`,
        c.sid_lengths.length ? `，学号 ${c.sid_lengths.join("/")} 位` : "", c.updated_at ? `，${c.updated_at} 更新` : "")),
        h("div", { class: "row" },
          h("button", { onclick: async () => {
            const full = await api("GET", `/api/rosters/${c.id}`);
            const nm = h("input", { type: "text", value: full.name });
            const ta = h("textarea", { rows: Math.min(18, full.students.length + 2), value: full.students.map(s => `${s.sid}\t${s.name}`).join("\n") });
            fill(box, h("label", { class: "field" }, h("span", {}, "班级名称"), nm),
              h("label", { class: "field" }, h("span", {}, "名单（每行“学号 姓名”）"), ta),
              h("div", { class: "row" },
                h("button", { class: "primary", onclick: async () => { try { const r = await api("PUT", `/api/rosters/${c.id}`, { name: nm.value, text: ta.value }); toast(`已保存 ${r.class.students.length} 人` + (r.bad.length ? `；有 ${r.bad.length} 行没看懂：${r.bad[0]}` : ""), !!r.bad.length); viewRoster(app); } catch (er) { toast(er.message, true); } } }, "保存"),
                h("button", { onclick: draw }, "取消")));
          } }, "查看 / 修改"),
          h("button", { class: "ghost", onclick: async () => { if (!confirm(`删除“${c.name}”的名单？已经批改的成绩不受影响，只是不再按这个名单对姓名。`)) return; try { await api("DELETE", `/api/rosters/${c.id}`); viewRoster(app); } catch (er) { toast(er.message, true); } } }, "删除"))));
    draw();
    return box;
  });
  fill(app, h("h1", {}, "班级名单"),
    h("p", { class: "muted" }, "名单用来把答题纸上涂的学号对上姓名、列出谁没交，也可以按名单打印“每人一份”的答题纸（姓名印好、学号涂好）。名单只存在这台电脑上。"),
    cards.length ? h("div", { class: "stack", style: "margin-bottom:18px" }, cards) : h("div", { class: "sheet empty" }, h("h2", {}, "还没有名单"), h("p", {}, "在下面粘贴或上传一个班的名单。")),
    h("details", { class: "sheet", open: !cards.length || null }, h("summary", {}, h("b", {}, cards.length ? "导入或更新名单" : "导入名单")), form));
}

// ------------------------------------------------------------------ 新建考试
async function viewNew(app) {
  const st = await api("GET", "/api/settings");
  const TEXT = {   // 随类型变的提示
    exam: { title: "例如：高二第二学期期中考试", answers: "有官方评分标准最好；没有的话可以让 AI 先写，你审核后再用。",
      sheet: "答题纸的 PDF 或扫描图。没有的话，建好后可以按题目生成答题纸，或者用一份学生答题纸来识别版面。",
      struct: "每行写“题号 题型 分值”，例如“1-8 单选 5分”“13 解答 12分 2问”。", ex: STRUCT_EXAMPLE },
    quiz: { title: "例如：第 5 周周测", answers: "有答案就传；没有的话 AI 会先做选择、填空题（每题独立做两遍核对），解答题写参考答案和评分细则，你审核后再用。",
      sheet: "周测一般不用传：建好后在“答题纸版面”里按题目生成答题纸，打印出来就能用。",
      struct: "每行写“题号 题型 分值”，例如“1-8 单选 5分”“13 解答 12分 2问”。", ex: STRUCT_EXAMPLE },
    homework: { title: "例如：5.2 导数的运算 课后作业", answers: "可以不传：AI 会先做选择、填空题（每题独立做两遍核对），解答题写参考答案。作业不打分，只判对错。",
      sheet: "作业一般不用传：建好后在“答题纸版面”里按题目生成答题纸（不印分值），打印出来就能用。",
      struct: "每行写“题号 题型”，作业不用写分值，例如“1-6 单选”“7-9 填空”“10 解答 2问”。", ex: "1-6 单选\n7-8 多选\n9-11 填空\n12 解答 2问\n13-14 解答" },
  };
  const hints = { answers: h("div", { class: "hint" }), sheet: h("div", { class: "hint" }), struct: h("div", { class: "hint" }) };
  const kindBox = h("div", { class: "kinds", role: "radiogroup", "aria-label": "类型" }, Object.entries(KIND_TEXT).map(([v, t]) =>
    h("label", { class: "kind" }, h("input", { type: "radio", name: "kind", value: v, checked: v === "exam" || null, onchange: upd }),
      h("span", {}, h("b", {}, t), h("small", {}, KIND_HINT[v])))));
  const form = h("form", { class: "sheet stack", onsubmit: submit },
    h("h1", {}, "新建考试"),
    !st.api_key ? h("div", { class: "notice" }, "还没有填写 DeepSeek API Key，录题和识别题号都需要它。", h("a", { href: "#/settings" }, "去设置")) : null,
    h("div", { class: "field" }, h("span", {}, "类型"), kindBox),
    h("label", { class: "field" }, h("span", {}, "名称"), h("input", { type: "text", name: "title", required: true })),
    h("div", { class: "grid2" },
      h("label", { class: "field" }, h("span", {}, "试卷"), h("input", { type: "file", name: "paper", multiple: true, accept: ".pdf,.png,.jpg,.jpeg,.tif,.tiff,.txt,.md" }),
        h("div", { class: "hint" }, "PDF、图片或文字（.txt）。Word 试卷请先另存为 PDF，公式才不会丢。作业拍照上传也可以。")),
      h("label", { class: "field" }, h("span", {}, "空白答题纸（可选）"), h("input", { type: "file", name: "sheet", multiple: true, accept: ".pdf,.png,.jpg,.jpeg,.tif,.tiff" }), hints.sheet)),
    h("div", { class: "grid2" },
      h("label", { class: "field" }, h("span", {}, "参考答案 / 评分标准（可选）"), h("input", { type: "file", name: "answers", multiple: true, accept: ".pdf,.png,.jpg,.jpeg,.tif,.tiff,.txt,.md" }), hints.answers),
      h("label", { class: "field" }, h("span", {}, "每份答题纸扫描后是几张图"),
        h("select", { name: "pages_per_sheet" }, h("option", { value: "1" }, "1 张（单面）"), h("option", { value: "2" }, "2 张（双面或两页）"),
          h("option", { value: "3" }, "3 张"), h("option", { value: "4" }, "4 张")),
        h("div", { class: "hint" }, "按题目生成答题纸的话不用管这个，会按生成的页数来。"))),
    h("label", { class: "field" }, h("span", {}, "也可以直接粘贴题目文字（可选）"), h("textarea", { name: "question_text", placeholder: "15.（12分）如图，在三棱锥 P-ABC 中……" })),
    h("label", { class: "field" }, h("span", {}, "没有试卷文件？写一下题目结构（可选）"),
      h("textarea", { name: "structure", rows: 4 }), hints.struct),
    h("label", { class: "check" }, h("input", { type: "checkbox", name: "generate_answers", checked: true }), "没有参考答案的题让 AI 先做：选择、填空题做出答案，解答题写参考答案和评分细则"),
    h("div", { class: "row" }, h("button", { class: "primary", type: "submit" }, "创建并开始录题"), h("a", { class: "btn ghost", href: "#/" }, "取消")));
  function upd() {
    const k = (form.querySelector("input[name=kind]:checked") || {}).value || "exam";
    const t = TEXT[k];
    form.title.placeholder = t.title;
    form.structure.placeholder = t.ex;
    hints.answers.textContent = t.answers;
    hints.sheet.textContent = t.sheet;
    hints.struct.textContent = t.struct + "建好后可以在“答题纸版面”里按题目生成答题纸，打印出来就能用。";
    kindBox.querySelectorAll(".kind").forEach(l => l.classList.toggle("on", l.querySelector("input").checked));
  }
  upd();
  fill(app, form);

  async function submit(ev) {
    ev.preventDefault();
    const fd = new FormData(form);
    fd.set("generate_answers", form.generate_answers.checked ? "true" : "false");
    const btn = form.querySelector("button[type=submit]");
    btn.disabled = true; btn.textContent = "上传中…";
    try {
      const r = await api("POST", "/api/exams", fd, true);
      location.hash = `#/exam/${r.exam.id}/questions`;
    } catch (e) { toast(e.message, true); btn.disabled = false; btn.textContent = "创建并开始录题"; }
  }
}

// ------------------------------------------------------------------ 考试页
const TABS = [["questions", "题目与评分细则"], ["layout", "答题纸版面"], ["scans", "上传扫描件"], ["review", "复核与成绩"]];

async function loadExam(id) {
  state.exam = await api("GET", `/api/exams/${id}`);
  state.qs = JSON.parse(JSON.stringify(state.exam.questions || []));
  state.tmpl = state.exam.template ? JSON.parse(JSON.stringify(state.exam.template)) : null;
  state.bust = Date.now();
  return state.exam;
}

async function viewExam(app, id, tab, sub) {
  if (!state.exam || state.exam.id !== id || !sub) await loadExam(id);
  const e = state.exam;
  app.className = tab === "review" ? "wide" : "";
  const sol = (e.questions || []).filter(q => q.type === "解答题");
  const allQ = (e.questions || []).filter(q => ["选择题", "填空题", "解答题"].includes(q.type));   // 作业的解答题可以没有参考答案（AI 自己做了再判）
  const done = { questions: allQ.length > 0 && allQ.every(q => (q.answer || "").trim() || (q.type === "解答题" && e.kind === "homework")),
    layout: !!e.template, scans: (e.sheets || []).length > 0, review: false };
  const title = h("input", { type: "text", value: e.title, "aria-label": "考试名称",
    onchange: async ev => { try { await api("PUT", `/api/exams/${id}/title`, { title: ev.target.value }); toast("名称已保存"); } catch (er) { toast(er.message, true); } } });
  const kind = e.kind || "exam";
  const kindSel = h("select", { class: "kindsel", "aria-label": "类型", title: "考试 / 周测按分值打分；平时作业不打分，只判对错",
    onchange: async () => {
      const nk = kindSel.value;
      const flip = (kind === "homework") !== (nk === "homework");
      const n = (e.sheets || []).length;
      if (flip && n && !confirm(`已经批了 ${n} 份答卷。改成${KIND_TEXT[nk]}后要按${nk === "homework" ? "“对 / 半对 / 错”" : "分数"}重新批一遍（会调用模型），老师改过的${nk === "homework" ? "分数" : "对错"}会清掉。继续吗？`)) {
        kindSel.value = kind; return;
      }
      try { await api("PUT", `/api/exams/${id}/kind`, { kind: nk }); toast(`已改成${KIND_TEXT[nk]}`); await loadExam(id); route(); }
      catch (er) { toast(er.message, true); kindSel.value = kind; }
    } }, Object.entries(KIND_TEXT).map(([v, t]) => h("option", { value: v, selected: v === kind || null }, t)));
  const head = h("div", { class: "exam-head" },
    h("div", { class: "row spread" }, h("h1", {}, title),
      h("div", { class: "row" }, kindSel, h("button", { class: "ghost", onclick: () => delExam(e) }, "删除考试"))),
    h("div", { class: "muted" }, `创建于 ${e.created_at}，`, kind === "homework" ? "平时作业（不打分，只判对错），" : "",
      `解答题 ${sol.length} 道，已批答卷 ${(e.sheets || []).length} 份`,
      e.usage && e.usage.calls ? `，累计调用模型 ${e.usage.calls} 次（输入约 ${Math.round((e.usage.prompt_tokens || 0) / 1000)}k tokens，输出约 ${Math.round((e.usage.completion_tokens || 0) / 1000)}k tokens）` : ""));
  const steps = h("nav", { class: "steps" }, TABS.map(([k, name], i) =>
    h("a", { href: `#/exam/${id}/${k}`, class: (k === tab ? "on " : "") + (done[k] ? "done" : "") }, h("b", {}, i + 1), name)));
  const job = h("div", { id: "jobbox" });
  const warn = (e.warnings || []).length && tab !== "review" ? h("details", { class: "notice", style: "margin-bottom:16px", open: e.warnings.length <= 6 || null },
    h("summary", {}, h("b", {}, `需要你看一下（${e.warnings.length} 条）`)),
    h("ul", {}, e.warnings.map(w => { const m = w.match(/^第 ?(\d+) ?题/); return h("li", {}, m && tab === "questions" ? h("a", { href: `#q-${m[1]}`, onclick: ev => { ev.preventDefault(); const el = document.getElementById(`q-${m[1]}`); if (el) el.scrollIntoView({ behavior: "smooth", block: "start" }); } }, w) : w); }))) : null;
  const lastErr = e.last_job && e.last_job.status === "error" && !(e.job && e.job.status === "running")
    ? h("div", { class: "notice err", style: "margin-bottom:16px" }, `上次“${e.last_job.label}”出错：${e.last_job.error}`) : null;
  const body = h("div", {});
  fill(app, head, steps, job, lastErr, warn, body);
  if (!e.has_key) job.append(h("div", { class: "notice", style: "margin-bottom:16px" }, "还没有填写 DeepSeek API Key。", h("a", { href: "#/settings" }, "去设置")));
  if (e.job && e.job.status === "running") watchJob(e.job);
  if (tab === "questions") renderQuestions(body);
  else if (tab === "layout") renderLayout(body);
  else if (tab === "scans") renderScans(body);
  else if (tab === "review") await renderReview(body, sub);
}

async function delExam(e) {
  if (!confirm(`删除“${e.title}”？所有上传的文件和批改结果都会删掉，不能恢复。`)) return;
  try { await api("DELETE", `/api/exams/${e.id}`); state.exam = null; location.hash = "#/"; } catch (er) { toast(er.message, true); }
}

function watchJob(job, after) {
  const box = $("#jobbox");
  if (!box) return;
  const bar = h("i", { style: `width:${Math.round(job.progress * 100)}%` });
  const msg = h("div", { class: "msg" }, `${job.label}：${job.message}`);
  fill(box, h("div", { class: "progress" }, h("div", { class: "bar" }, bar), msg));
  clearInterval(state.poll);
  state.poll = setInterval(async () => {
    let j;
    try { j = await api("GET", `/api/jobs/${job.id}`); } catch { return; }
    bar.style.width = `${Math.round(j.progress * 100)}%`;
    msg.textContent = `${j.label}：${j.message}`;
    if (j.status !== "running") {
      clearInterval(state.poll);
      if (j.status === "error") toast(`${j.label}出错：${j.error}`, true);
      else toast(`${j.label}完成`);
      await loadExam(state.exam.id);
      if (after) after(j); else route();
    }
  }, 1500);
}

// ------------------------------------------------------------------ 第 1 步：题目与评分细则
function renderQuestions(body) {
  const e = state.exam;
  const sol = state.qs.filter(q => q.type === "解答题");
  const others = state.qs.filter(q => q.type !== "解答题");
  const upload = h("details", { class: "sheet", open: !state.qs.length || null },
    h("summary", {}, h("b", {}, state.qs.length ? "补充上传试卷或参考答案" : "上传试卷，让 AI 录入题目")),
    uploadSetupForm());
  const hw = isHW();
  const quickTa = h("textarea", { rows: 5, placeholder: hw ? "1-6 单选\n7-8 多选\n9-11 填空\n12 解答 2问\n13-14 解答" : STRUCT_EXAMPLE });
  const quick = h("details", { class: "sheet", open: !state.qs.length || null, style: "margin-top:12px" },
    h("summary", {}, h("b", {}, "按题号快速添加"), h("span", { class: "muted" }, hw ? "　没有试卷文件时，写题号和题型就行（作业不用写分值）" : "　没有试卷文件时，写题号、题型和分值就行")),
    h("div", { class: "stack", style: "margin-top:12px" }, quickTa,
      h("div", { class: "row" }, h("button", { onclick: async () => {
        if (!quickTa.value.trim()) { toast("先写题目结构", true); return; }
        try {
          const r = await api("POST", `/api/exams/${e.id}/questions/quick`, { text: quickTa.value });
          toast(`已添加/更新 ${r.added} 道题` + (r.bad.length ? `；这几行没看懂：${r.bad.join("；")}` : ""), !!r.bad.length);
          await loadExam(e.id); route();
        } catch (er) { toast(er.message, true); } } }, "添加"),
        h("span", { class: "hint" }, hw ? "每行一段：“1-6 单选”“7-8 多选”“9-11 填空”“12 解答 2问”。已有的题号只更新题型，答案保留。"
          : "每行一段：“1-8 单选 5分”“9-11 多选 6分”“12-14 填空 5分”“15 解答 13分 2问”。已有的题号只更新题型和分值，答案保留。"))));
  const cards = sol.map(q => questionCard(q));
  const empty = !state.qs.length ? h("div", { class: "sheet empty" }, h("h2", {}, "还没有题目"),
    h("p", {}, "上传试卷后会自动录入；也可以按题号快速添加，或者手动添加解答题。")) : null;
  const addBtn = h("button", { onclick: () => { const n = prompt("题号（例如 18）"); if (!n) return; state.qs.push({ number: n.trim(), type: "解答题", score: null, stem: "", figure: "", parts: [], answer: "", rubric: "", notes: "", confirmed: false, answer_source: "none" }); renderQuestions(body); } }, "添加解答题");
  const save = h("div", { class: "sticky-save" }, addBtn, h("button", { class: "primary", onclick: () => saveQuestions() }, "保存题目"));
  fill(body, upload, quick, h("div", { style: "height:16px" }), objectiveTable(others), empty, ...cards, save);
}

function objectiveTable(others) {
  if (!others.length) return null;
  const e = state.exam;
  const hw = isHW();
  const kinds = [["选择题", others.filter(q => q.type === "选择题")], ["填空题", others.filter(q => q.type === "填空题")],
    ["其他", others.filter(q => !["选择题", "填空题"].includes(q.type))]].filter(([, qs]) => qs.length);
  // 没有答案的选择/填空题：让 AI 做（每题独立做两遍，不一致的标出来请老师核对）
  const objs = others.filter(q => ["选择题", "填空题"].includes(q.type));
  const blank = objs.filter(q => !(q.answer || "").trim());
  const canSolve = blank.filter(q => (q.paper_crops || []).length || (q.stem || "").trim());
  const noText = blank.filter(q => !canSolve.includes(q)).map(q => q.number);
  const byAI = objs.filter(q => q.answer_source === "ai" && (q.answer || "").trim());
  const ansChecks = q => (q.checks || []).filter(c => c.where === "答案");
  const agreeing = byAI.filter(q => !q.confirmed && !ansChecks(q).length);
  const disputed = byAI.filter(q => !q.confirmed && ansChecks(q).length);
  const solve = async redo => {
    const again = byAI.filter(q => !q.confirmed).length;
    if (redo && !confirm(`让 AI 把它做的、还没确认的 ${again} 道选择、填空题重新做一遍？老师填写、改过、确认过的答案不动。`)) return;
    if (!(await saveQuestions(true))) return;
    try { const r = await api("POST", `/api/exams/${e.id}/objective/solve`, { redo: !!redo }); watchJob(r.job); } catch (er) { toast(er.message, true); }
  };
  const aiBar = canSolve.length || byAI.length || noText.length ? h("div", { class: "aibar" },
    h("div", { class: "row" },
      canSolve.length ? h("button", { class: "primary", type: "button", onclick: () => solve(false) }, `让 AI 做（${canSolve.length} 道没有答案）`) : null,
      agreeing.length ? h("button", { type: "button", onclick: async () => {
        if (!confirm(`AI 两遍做得一样的 ${agreeing.length} 道题（第 ${agreeing.map(q => q.number).join("、")} 题），都确认用 AI 的答案？`)) return;
        agreeing.forEach(q => (q.confirmed = true)); await saveQuestions(); } }, `确认两遍一致的 ${agreeing.length} 道`) : null,
      byAI.some(q => !q.confirmed) ? h("button", { class: "ghost", type: "button", title: "还没确认的 AI 答案重新做一遍", onclick: () => solve(true) }, "AI 重做") : null),
    h("div", { class: "hint" }, canSolve.length || byAI.length ? "AI 每道题独立做两遍：两遍一样的标“AI”；不一样的标红，请你核对后点“确认”。改过的答案算老师填的。" : "",
      disputed.length ? h("b", { class: "red-text" }, `第 ${disputed.map(q => q.number).join("、")} 题 AI 两遍做得不一样，请核对。`) : null,
      noText.length ? `第 ${noText.join("、")} 题没有题目内容，AI 没法做：上传试卷，或者直接填答案。` : "")) : null;
  return h("section", { class: "qcard", id: "objective" },
    h("header", {}, h("h3", {}, "选择题和填空题"),
      h("span", { class: "muted" }, hw ? "参考答案可以直接改。" : "分值和参考答案都可以直接改。", "选择题写字母（多选写在一起，如 BD）；填空题写 LaTeX，不加 $。")),
    h("div", { class: "body stack" }, aiBar, kinds.map(([t, qs]) => {
      const total = qs.reduce((a, q) => a + (+q.score || 0), 0);
      const key = (qs.find(q => (q.answer_crops || []).length) || {}).answer_crops;
      const missing = qs.filter(q => !(q.answer || "").trim()).length;
      return h("div", {},
        h("div", { class: "row spread", style: "margin-bottom:6px" },
          h("h3", {}, `${t} ${qs.length} 道`, hw ? "" : `，共 ${total} 分`, missing && t !== "其他" ? h("span", { class: "badge red", style: "margin-left:8px" }, `${missing} 道缺答案`) : null),
          key && key.length ? h("button", { class: "ghost", type: "button", onclick: () => lightbox(fileUrl(key[0])) }, "看答案原图") : null),
        h("div", { class: "tablewrap" }, h("table", { class: "objt" },
          h("thead", {}, h("tr", {}, h("th", {}, "题号"), hw ? null : h("th", {}, "分值"), h("th", {}, "参考答案"), t === "填空题" ? h("th", {}, "显示效果") : null,
            t === "选择题" ? h("th", { title: "生成答题纸时每题印几个选项格" }, "选项数") : null,
            h("th", {}, "原题"), t !== "选择题" ? h("th", {}, "判分说明（选填）") : null)),
          h("tbody", {}, qs.map(q => {
            const pv = h("span", { class: "pv" });
            const upd = () => fill(pv, q.answer ? mathBlock("$" + q.answer + "$", "pvmath") : "");
            if (t === "填空题") upd();
            const ck = ansChecks(q);
            const sol = q.ai_solve || {};
            const ai = q.answer_source === "ai" && (q.answer || "").trim();
            const aiInfo = ai ? h("div", { class: "aiinfo" },
              h("span", { class: "badge " + (q.confirmed ? "ok" : ck.length ? "red" : "ai"),
                title: (sol.answers || []).length ? `AI 两遍的答案：${sol.answers.map(a => a || "空").join(" / ")}` : "" }, q.confirmed ? "AI，已确认" : ck.length ? "AI，待核对" : "AI"),
              !q.confirmed ? h("button", { class: "ghost tiny", type: "button", title: "核对过了，用这个答案", onclick: async () => { q.confirmed = true; await saveQuestions(); } }, "确认") : null,
              ck.length && !q.confirmed ? h("div", { class: "red-text small" }, ck[0].msg) : null,
              sol.brief ? h("div", { class: "muted small brief", title: sol.brief }, "思路：" + sol.brief) : null) : null;
            return h("tr", { class: ck.length && !q.confirmed ? "rv" : "" },
              h("td", { class: "num" }, q.number),
              hw ? null : h("td", {}, h("input", { type: "number", min: "0", step: "0.5", value: q.score ?? "", "aria-label": `第${q.number}题分值`, style: "width:68px",
                oninput: ev => (q.score = ev.target.value === "" ? null : +ev.target.value) })),
              h("td", {}, h("input", { type: "text", value: q.answer || "", "aria-label": `第${q.number}题参考答案`,
                style: t === "选择题" ? "width:80px" : "", placeholder: t === "选择题" ? "B" : "\\frac{\\sqrt{5}}{5}",
                oninput: ev => { q.answer = ev.target.value; if (t === "填空题") upd(); } }), aiInfo),
              t === "填空题" ? h("td", {}, pv) : null,
              t === "选择题" ? h("td", {}, h("input", { type: "number", min: "2", max: "8", value: q.options ?? "", placeholder: "4", style: "width:56px",
                "aria-label": `第${q.number}题选项数`, oninput: ev => (q.options = ev.target.value === "" ? null : +ev.target.value) })) : null,
              h("td", {}, (q.paper_crops || []).length ? h("button", { class: "ghost", type: "button", onclick: () => lightbox(fileUrl(q.paper_crops[0])) }, "看原题")
                : (q.stem || "").trim() ? h("button", { class: "ghost", type: "button", title: q.stem.slice(0, 200), onclick: () => textBox(`第${q.number}题`, q.stem) }, "看题目")
                : h("span", { class: "muted" }, "—")),
              t !== "选择题" ? h("td", {}, h("input", { type: "text", value: q.notes || "", placeholder: "如：写成小数也算对",
                oninput: ev => (q.notes = ev.target.value) })) : null);
          })))));
    })));
}

function uploadSetupForm() {
  const f = h("form", { class: "stack", style: "margin-top:12px", onsubmit: async ev => {
      ev.preventDefault();
      const fd = new FormData(f);
      fd.set("generate_answers", f.generate_answers.checked ? "true" : "false");
      try { const r = await api("POST", `/api/exams/${state.exam.id}/setup`, fd, true); if (r.job) { toast("已开始"); watchJob(r.job); } else toast("没有选择文件", true); }
      catch (er) { toast(er.message, true); }
    } },
    h("div", { class: "grid2" },
      h("label", { class: "field" }, h("span", {}, "试卷"), h("input", { type: "file", name: "paper", multiple: true, accept: ".pdf,.png,.jpg,.jpeg,.tif,.tiff,.txt,.md" })),
      h("label", { class: "field" }, h("span", {}, "参考答案 / 评分标准"), h("input", { type: "file", name: "answers", multiple: true, accept: ".pdf,.png,.jpg,.jpeg,.tif,.tiff,.txt,.md" }))),
    h("label", { class: "check" }, h("input", { type: "checkbox", name: "generate_answers", checked: true }), "没有参考答案的解答题，让 AI 先生成"),
    h("div", { class: "row" }, h("button", { type: "submit" }, "上传并录入"),
      (state.exam.files.paper || []).length || (state.exam.files.answers || []).length
        ? h("button", { type: "button", onclick: async () => {
            if (!confirm("用已经上传的试卷和答案重新录入？题干和参考答案会按原文件重新录一遍，你手动改过的题干和答案会被覆盖（手动改过的答题纸版面会保留）。")) return;
            const fd = new FormData(); fd.set("generate_answers", f.generate_answers.checked ? "true" : "false");
            try { const r = await api("POST", `/api/exams/${state.exam.id}/setup`, fd, true); if (r.job) watchJob(r.job); } catch (er) { toast(er.message, true); }
          } }, "用已上传的文件重新录入") : null),
    h("div", { class: "hint" }, "数字版 PDF（Word/WPS 导出的）会按题号位置逐题录入，并和原文核对；扫描件和照片会整份交给模型录入，请对照原图检查。只上传试卷时，现有的参考答案会保留。"));
  return f;
}

function questionCard(q) {
  const id = state.exam.id;
  const hw = isHW();      // 平时作业：不打分，不显示满分和小问分值
  let preview = false;
  const checks = q.checks || [];
  const nCheck = checks.length;
  const badge = q.answer_source === "ai" ? (q.confirmed ? h("span", { class: "badge ok" }, "AI 生成，已确认") : h("span", { class: "badge ai" }, "AI 生成，待确认"))
    : q.answer_source === "key" ? (nCheck && !q.confirmed ? h("span", { class: "badge ai" }, `录自答案文件，${nCheck} 处待核对`) : h("span", { class: "badge ok" }, q.confirmed ? "录自答案文件，已确认" : "录自答案文件"))
    : q.answer_source === "teacher" ? h("span", { class: "badge ok" }, "老师填写") : h("span", { class: "badge red" }, "缺参考答案");
  const ta = (key, label, rows, hint) => {
    const t = h("textarea", { rows: rows || 5, value: q[key] || "", oninput: ev => { q[key] = ev.target.value; } });
    return h("label", { class: "field" }, h("span", {}, label), t, hint ? h("div", { class: "hint" }, hint) : null);
  };
  const partsBox = h("div", { class: "parts" });
  const drawParts = () => {
    fill(partsBox,
      ...(q.parts || []).map((p, k) => h("span", { class: "part" },
        h("input", { class: "lbl", type: "text", value: p.label, "aria-label": "小问编号", oninput: ev => (p.label = ev.target.value) }),
        hw ? null : h("input", { type: "number", step: "0.5", min: "0", value: p.score ?? "", "aria-label": "小问分值", oninput: ev => (p.score = ev.target.value === "" ? null : +ev.target.value) }), hw ? null : "分",
        h("button", { class: "ghost", type: "button", title: "删除这一小问", onclick: () => { q.parts.splice(k, 1); drawParts(); } }, "×"))),
      h("button", { type: "button", onclick: () => { q.parts = q.parts || []; q.parts.push({ label: `(${q.parts.length + 1})`, score: null, text: "" }); drawParts(); } }, "加一小问"),
      (() => {
        const sum = (q.parts || []).reduce((a, p) => a + (+p.score || 0), 0);
        return !hw && q.parts && q.parts.length && q.score && sum !== +q.score ? h("span", { class: "badge red" }, `小问合计 ${sum} ≠ 满分 ${q.score}`) : null;
      })());
  };
  drawParts();
  const conv = { cumulative: "答案里的“……X分”是累计分", cumulative_all: "答案里的“……X分”是全题累计分", step: "答案里的“……X分”是每步分值" }[q.score_convention];
  const src = (rels, caption) => (rels || []).length
    ? h("figure", { class: "srcimg" }, h("figcaption", {}, caption),
        rels.map(r => h("img", { src: fileUrl(r), alt: caption, loading: "lazy", onclick: () => lightbox(fileUrl(r)) })))
    : null;
  const pair = (left, right) => left ? h("div", { class: "pair" }, left, right) : right;
  const paperSide = h("div", { class: "stack" },
    ta("stem", "题目全文（含所有小问；数学式写成 $...$）", Math.min(14, 4 + (q.stem || "").split("\n").length * 1.4 | 0)),
    ta("figure", "图形说明（给模型看，选填）", 2),
    h("div", { class: "field" }, h("span", { class: "muted", style: "font-size:13px" }, hw ? "小问（逐个判对错）" : "小问分值", !hw && q.score_from ? `（满分由${q.score_from}）` : ""), partsBox));
  const answerSide = h("div", { class: "stack" },
    hw ? ta("answer", "参考答案（选填：没有的话，AI 批改时自己先做一遍再判）", Math.min(18, 5 + (q.answer || "").split("\n").length * 1.3 | 0))
      : ta("answer", "参考答案（步骤后的“……X分”就是评分标准，原样保留）", Math.min(18, 5 + (q.answer || "").split("\n").length * 1.3 | 0), conv),
    hw ? null : ta("rubric", "评分细则补充（选填：答案里没写到的给分、扣分规则）", 3),
    ta("notes", hw ? "批改说明（选填，例如：第(2)问用向量法也算对；只要求写出结论）" : "阅卷补充说明（选填，例如：第(2)问用向量法同样给满分）", 2));
  const editView = h("div", { class: "stack" },
    pair(src(q.paper_crops, "原题（试卷截图）"), paperSide),
    h("hr", { class: "sep" }),
    pair(src(q.answer_crops, "原答案（答案截图）"), answerSide));
  const pv = h("div", { class: "grid2", hidden: true });
  const checkBox = nCheck && !q.confirmed ? h("div", { class: "notice", style: "margin-bottom:14px" },
    h("b", {}, "录入后自动核对发现："), h("ul", {}, checks.map(c => h("li", {}, `${c.where}：${c.msg}`))),
    h("div", { class: "hint" }, "对照左边的原图检查，改好后点“确认无误”。")) : null;
  const bodyEl = h("div", { class: "body" }, checkBox, editView, pv,
    q.issues ? h("div", { class: "notice", style: "margin-top:12px" }, "AI 提示：" + q.issues) : null);
  const needConfirm = !q.confirmed && (q.answer_source === "ai" || (q.answer_source === "key" && nCheck) || nCheck);
  const card = h("section", { class: "qcard", id: `q-${q.number}` },
    h("header", {},
      h("h3", {}, `第${q.number}题`),
      hw ? null : h("label", { class: "row", style: "gap:6px" }, "满分", h("input", { type: "number", min: "0", step: "0.5", value: q.score ?? "", oninput: ev => { q.score = ev.target.value === "" ? null : +ev.target.value; drawParts(); } }), "分"),
      hw && q.answer_source === "none" ? h("span", { class: "badge" }, "没有参考答案，AI 会自己做") : badge,
      h("span", { style: "margin-left:auto" }),
      h("button", { type: "button", onclick: () => { preview = !preview; editView.hidden = preview; pv.hidden = !preview;
        if (preview) fill(pv, h("div", {}, h("h3", {}, "题目"), mathBlock(q.stem)), h("div", {}, h("h3", {}, "参考答案"), mathBlock(q.answer), q.rubric ? h("h3", { style: "margin-top:10px" }, "评分细则补充") : null, q.rubric ? mathBlock(q.rubric) : null)); } }, "预览公式"),
      h("button", { type: "button", onclick: async () => { if (!(await saveQuestions(true))) return; try { const r = await api("POST", `/api/exams/${id}/questions/${encodeURIComponent(q.number)}/generate`); watchJob(r.job); } catch (er) { toast(er.message, true); } } },
        q.answer ? "让 AI 重写答案" : hw ? "让 AI 写参考答案" : "让 AI 生成答案和细则"),
      needConfirm ? h("button", { class: "primary", type: "button", onclick: async () => { q.confirmed = true; await saveQuestions(); } }, "确认无误") : null,
      h("button", { class: "ghost", type: "button", title: "删除这道题", onclick: () => { if (!confirm(`删除第${q.number}题？`)) return; state.qs.splice(state.qs.indexOf(q), 1); renderQuestions(card.parentElement); } }, "删除")),
    bodyEl);
  return card;
}

async function saveQuestions(quiet) {
  try {
    state.exam = await api("PUT", `/api/exams/${state.exam.id}/questions`, { questions: state.qs });
    state.qs = JSON.parse(JSON.stringify(state.exam.questions));
    if (!quiet) { toast("题目已保存"); route(); }
    return true;
  } catch (e) { toast(e.message, true); return false; }
}

// ------------------------------------------------------------------ 第 2 步：答题纸版面
function renderLayout(body) {
  const e = state.exam;
  const reform = h("form", { class: "stack", onsubmit: async ev => {
      ev.preventDefault();
      const fd = new FormData(reform);
      if (!reform.files.files.length) { toast("请选择答题纸文件", true); return; }
      try { const r = await api("POST", `/api/exams/${e.id}/layout`, fd, true); watchJob(r.job); } catch (er) { toast(er.message, true); }
    } },
    h("div", { class: "row" },
      h("label", { class: "check" }, h("input", { type: "radio", name: "source", value: "blank", checked: true }), "空白答题纸"),
      h("label", { class: "check" }, h("input", { type: "radio", name: "source", value: "scan" }), `一份学生答题纸（取前 ${e.pages_per_sheet || 1} 张）`)),
    h("input", { type: "file", name: "files", multiple: true, accept: ".pdf,.png,.jpg,.jpeg,.tif,.tiff" }),
    h("div", {}, h("button", { class: e.template ? "" : "primary", type: "submit" }, e.template ? "重新识别版面" : "识别版面")),
    h("div", { class: "hint" }, "程序会找出定位点和所有封闭的题框，再让 DeepSeek 看图标出每个框是第几题、几分。识别后可以在图上修改。"));
  if (!state.tmpl) {
    // 考试一般有学校统一的答题纸（推荐上传识别）；周测、作业推荐按题目生成
    const own = (e.kind || "exam") === "exam";
    const rec = h("span", { class: "badge ok", style: "margin-left:8px" }, "推荐");
    const genCard = h("div", { class: "sheet stack" }, h("h2", {}, "按题目生成答题纸", own ? null : rec),
      h("p", { class: "muted" }, "按第 1 步的题目自动排版：选择题涂卡、填空题横线、解答题方框，带定位点、二维码和涂学号的格子。打印或复印出来就能用，扫描后不用再识别版面。"),
      state.qs.length ? sheetGenForm(e) : h("div", { class: "notice" }, "还没有题目。先在第 1 步上传试卷，或者“按题号快速添加”。"));
    const ownCard = h("div", { class: "sheet stack" }, h("h2", {}, "用现成的答题纸", own ? rec : null),
      h("p", { class: "muted" }, own ? "考试用学校统一印的答题纸：上传一张空白答题纸（PDF 或扫描图），程序识别题框和题号。手边没有空白的，就用一份学生的答题纸。"
        : "每次的答题纸不一样也没关系：上传一张空白答题纸（PDF 或扫描图），程序识别题框和题号。手边没有空白的，就用一份学生的答题纸。"), reform);
    fill(body, h("div", { class: "grid2 choose" }, ...(own ? [ownCard, genCard] : [genCard, ownCard])));
    return;
  }
  const t = state.tmpl;
  const by = { ai: "DeepSeek 识别", pdf_text: "直接从答题纸 PDF 的文字读出题号", "pdf_text+ai": "部分从 PDF 文字读出，其余由 DeepSeek 识别",
    heuristic: "按规则粗略标注（没有调用 AI），请逐个检查", teacher: "老师修改过", generated: "按题目生成", none: "未标注" }[t.labeled_by] || t.labeled_by;
  const gen = t.generated;
  const side = h("div", { class: "side" });
  const pagesEl = h("div", { class: "stack" });
  const drawBtn = h("button", { onclick: () => { state.drawing = !state.drawing; drawBtn.textContent = state.drawing ? "在图上拖动画框…（点此取消）" : "画一个新区域"; pagesEl.querySelectorAll(".page-view").forEach(v => v.classList.toggle("drawing", state.drawing)); } }, "画一个新区域");
  const redraw = () => {
    fill(pagesEl, ...t.pages.map(pg => pageView(pg, redraw, side)));
    renderSide(side, redraw);
  };
  fill(body, gen ? generatedBar(e, gen) : null,
    h("div", { class: "row spread", style: "margin-bottom:10px" },
      h("div", {}, h("b", {}, "版面"), h("span", { class: "muted" }, `：${t.pages.length} 页，${by}。红框是要批改的解答题。点一个框可以修改题号、题型和分值。`)),
      h("div", { class: "row" }, drawBtn, h("button", { class: "primary", onclick: saveTemplate }, "保存版面"))),
    h("div", { class: "layout-wrap" }, pagesEl, side),
    gen ? null : h("details", { class: "sheet", style: "margin-top:16px" }, h("summary", {}, h("b", {}, "改用按题目生成的答题纸")),
      h("p", { class: "muted" }, "按第 1 步的题目重新排一份答题纸（带定位点、二维码、涂学号），会替换现在的版面。"), sheetGenForm(e)),
    h("details", { class: "sheet", style: "margin-top:16px" }, h("summary", {}, h("b", {}, gen ? "改用现成的答题纸（上传识别）" : "重新识别版面")), reform));
  redraw();
}

const PAPER_TEXT = { A4: "A4 竖版", A3: "A3 横版两栏" };
const SPACE_TEXT = { compact: "紧凑", normal: "标准", loose: "宽松" };

function sheetGenForm(e) {
  const gen = (e.template || {}).generated || {};
  const digits = h("input", { type: "number", name: "id_digits", min: "0", max: "12", value: gen.id_digits ?? 4, style: "width:84px" });
  const digitHint = h("div", { class: "hint" }, "学生在格子里涂自己的学号，程序按名单对上姓名。填 0 就不印涂学号的格子（只写姓名，由 AI 认）。");
  api("GET", "/api/rosters").then(list => {   // 有名单时，默认位数按名单里的学号长度
    const lens = [...new Set(list.flatMap(c => c.sid_lengths))];
    if (!lens.length) return;
    if (gen.id_digits == null) digits.value = Math.max(...lens);
    digitHint.textContent = `名单里的学号是 ${lens.join("/")} 位。` + digitHint.textContent;
  }).catch(() => {});
  const btn = h("button", { class: "primary", type: "submit" }, e.template && gen.paper ? "重新生成" : "生成答题纸");
  const form = h("form", { class: "stack", onsubmit: async ev => {
      ev.preventDefault();
      const graded = (e.sheets || []).length;
      if (e.template && !confirm(graded ? `已经批过 ${graded} 份答卷。重新生成后版面会变，之前打印的答题纸就和新版面对不上了。继续吗？`
        : "用生成的答题纸替换现在的版面？")) return;
      btn.disabled = true; btn.textContent = "正在排版…";
      try {
        await api("POST", `/api/exams/${e.id}/answer-sheet`, { paper: form.paper.value, space: form.space.value, id_digits: +digits.value, title: form.title.value });
        toast("答题纸已生成，可以下载打印了"); await loadExam(e.id); route();
      } catch (er) { toast(er.message, true); btn.disabled = false; btn.textContent = "生成答题纸"; }
    } },
    h("div", { class: "grid2" },
      h("label", { class: "field" }, h("span", {}, "纸张"), h("select", { name: "paper" }, Object.entries(PAPER_TEXT).map(([v, t]) => h("option", { value: v, selected: (gen.paper || "A4") === v || null }, t))),
        h("div", { class: "hint" }, "题目多（比如一整套试卷）用 A3 两栏，平时作业用 A4。")),
      h("label", { class: "field" }, h("span", {}, "解答题作答空间"), h("select", { name: "space" }, Object.entries(SPACE_TEXT).map(([v, t]) => h("option", { value: v, selected: (gen.space || "normal") === v || null }, t))),
        h("div", { class: "hint" }, isHW() ? "作业不印分值，按小问数给方框高度；“宽松”多留 25% 的地方。生成后可以拖方框下边改高度。"
          : "按每题分值给方框高度；“宽松”多留 25% 的地方。生成后可以拖方框下边改高度。"))),
    h("div", { class: "grid2" },
      h("label", { class: "field" }, h("span", {}, "学号位数"), digits, digitHint),
      h("label", { class: "field" }, h("span", {}, "标题"), h("input", { type: "text", name: "title", value: gen.title || e.title }))),
    h("div", {}, btn));
  return form;
}

function generatedBar(e, gen) {
  const n = (gen.plan && gen.plan.pages || []).length;
  const pick = h("select", { "aria-label": "按哪个班的名单打印", style: "width:auto" });
  const personal = h("div", { class: "row tight-row" });
  api("GET", "/api/rosters").then(list => {
    if (!list.length) { fill(personal, h("a", { href: "#/roster" }, "导入班级名单"), h("span", { class: "muted" }, "后可以打印“每人一份”")); return; }
    fill(pick, list.map(c => h("option", { value: c.id }, `${c.name}（${c.count} 人）`)));
    fill(personal, pick, h("button", { onclick: () => window.open(`/api/exams/${e.id}/answer-sheet.pdf?classes=${encodeURIComponent(pick.value)}`, "_blank") }, "每人一份 PDF"));
  }).catch(() => {});
  const stale = (e.warnings || []).some(w => w.startsWith("答题纸是按改动之前的题目生成的"));
  return h("div", { class: "sheet stack", style: "margin-bottom:16px" },
    h("div", { class: "row spread" },
      h("div", {}, h("b", {}, "答题纸是按题目生成的"), h("span", { class: "muted" }, `：${PAPER_TEXT[gen.paper] || gen.paper}，${n} 页${n > 1 ? "（双面打印）" : ""}，`,
        gen.id_digits ? `涂 ${gen.id_digits} 位学号，` : "不涂学号，", `${gen.at} 生成`)),
      h("div", { class: "row" }, h("a", { class: "btn primary", href: `/api/exams/${e.id}/answer-sheet.pdf`, target: "_blank" }, "下载答题纸 PDF"), personal)),
    stale ? h("div", { class: "notice" }, "题目改过了，答题纸是按改之前的题目排的。请在下面“重新生成”，再重新打印。") : null,
    gen.paper === "A4" && n > 2 ? h("div", { class: "notice" }, `A4 要印 ${n} 页（${Math.ceil(n / 2)} 张纸），学生容易把几张纸弄散。可以改用 A3 两栏重新生成。`) : null,
    h("div", { class: "hint" }, "打印时选“实际大小”，不要“缩放到页面”；多页的双面打印。通用版每份都一样，可以直接复印；“每人一份”按名单把姓名印好、学号涂好，适合考试时按座位发。"),
    h("details", {}, h("summary", {}, "重新生成（改纸张、学号位数、作答空间）"), sheetGenForm(e)));
}

function pageView(pg, redraw, side) {
  const [W, H] = pg.size;
  const img = h("img", { src: fileUrl(pg.image), alt: `答题纸第${pg.page + 1}页`, draggable: "false" });
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none" });
  for (const r of pg.regions) {
    const [x0, y0, x1, y1] = r.rect;
    const sel = state.layoutSel && state.layoutSel.page === pg.page && state.layoutSel.id === r.id;
    const label = (r.questions && r.questions.length ? `第${r.questions.length > 3 ? r.questions[0] + "-" + r.questions[r.questions.length - 1] : r.questions.join(",")}题 ` : "") + (r.kind || "未标注") + (r.full_score && !isHW() ? ` ${r.full_score}分` : "");
    const fs = Math.round(Math.max(W, H) / 75);          // 字号按模板尺寸缩放，显示出来约 13px
    const tw = Math.min(x1 - x0, Math.max(4 * fs, label.length * fs * 0.95 + fs));
    svg.append(s("g", { class: `region k-${r.kind || ""}${sel ? " sel" : ""}`, onclick: ev => { if (state.drawing) return; ev.stopPropagation(); state.layoutSel = { page: pg.page, id: r.id }; redraw(); } },
      s("rect", { x: x0, y: y0, width: x1 - x0, height: y1 - y0 }),
      s("rect", { class: "tag", x: x0, y: y0, width: tw, height: Math.round(fs * 1.6), style: "stroke:none" }),
      s("text", { x: x0 + fs * 0.5, y: y0 + fs * 1.15, "font-size": fs }, label)));
    const sf = Math.round(fs * 0.85);
    for (const sl of r.slots || []) {
      const [a, b, c, d] = sl.rect;
      svg.append(s("g", { class: "slot" + (sl.q ? "" : " noq") }, s("rect", { x: a, y: b, width: c - a, height: d - b }),
        s("text", { x: c - sf * 0.3, y: b + sf, "font-size": sf, "text-anchor": "end" }, sl.q ? `第${sl.q}空` : "?")));
    }
    for (const g of r.groups || []) {
      for (const o of g.options) {
        const [a, b, c, d] = o.rect;
        svg.append(s("rect", { class: "opt", x: a, y: b, width: c - a, height: d - b }));
      }
      const [a, b, c, d] = g.rect;
      // 一行里并排好几道题时，题号标在这一组选项的右边（只写数字，免得挤到下一组）；一行一道就标在区域右边
      const shared = (r.groups || []).some(o => o !== g && o.rect[1] < d && o.rect[3] > b);
      svg.append(shared
        ? s("text", { class: "optq" + (g.q ? "" : " noq"), x: c + sf * 0.25, y: d - (d - b) * 0.2, "font-size": Math.round(sf * 0.8) }, g.q || "?")
        : s("text", { class: "optq" + (g.q ? "" : " noq"), x: x1 - sf * 0.6, y: d - sf * 0.15, "font-size": sf, "text-anchor": "end" }, g.q ? `第${g.q}题` : "?"));
    }
  }
  const view = h("div", { class: "page-view" + (state.drawing ? " drawing" : "") }, img, svg);
  // 拖动画新区域
  let start = null, ghost = null;
  const pt = ev => { const b = view.getBoundingClientRect(); return [(ev.clientX - b.left) * W / b.width, (ev.clientY - b.top) * H / b.height]; };
  view.addEventListener("pointerdown", ev => { if (!state.drawing) return; start = pt(ev); ghost = s("rect", { x: start[0], y: start[1], width: 1, height: 1, style: "fill:rgba(36,88,166,.15);stroke:#2458a6;stroke-width:3;vector-effect:non-scaling-stroke" }); svg.append(ghost); view.setPointerCapture(ev.pointerId); });
  view.addEventListener("pointermove", ev => { if (!start) return; const p = pt(ev); ghost.setAttribute("x", Math.min(p[0], start[0])); ghost.setAttribute("y", Math.min(p[1], start[1])); ghost.setAttribute("width", Math.abs(p[0] - start[0])); ghost.setAttribute("height", Math.abs(p[1] - start[1])); });
  view.addEventListener("pointerup", ev => {
    if (!start) return;
    const p = pt(ev);
    const rect = [Math.min(p[0], start[0]), Math.min(p[1], start[1]), Math.max(p[0], start[0]), Math.max(p[1], start[1])].map(Math.round);
    start = null;
    if (rect[2] - rect[0] < 20 || rect[3] - rect[1] < 20) { ghost.remove(); return; }
    let k = 1; const ids = new Set(pg.regions.map(r => r.id)); while (ids.has(`R${k}`)) k++;
    const r = { id: `R${k}`, rect, kind: "解答题", questions: [], full_score: null, header: "" };
    pg.regions.push(r);
    state.layoutSel = { page: pg.page, id: r.id };
    state.drawing = false;
    redraw();
    toast("已添加区域，在右边填写题号");
  });
  addGrips(pg, svg, view, W, H);
  return h("div", {}, h("div", { class: "muted", style: "margin-bottom:4px" }, `第 ${pg.page + 1} 页`,
    pg.markers && pg.markers.length ? `（找到 ${pg.markers.length} 个定位点）` : "（没找到定位点，对齐会改用图像特征）",
    (pg.notes || []).length ? "：" + pg.notes.join("；") : "", pg.unboxed ? `；AI 提示：${pg.unboxed}` : ""), view);
}

// 生成的答题纸：拖解答题方框的下边框改高度，松手后按新高度重新排版、重新生成 PDF
function boxTotalMM(q) {
  const plan = ((state.tmpl || {}).generated || {}).plan;
  return plan ? plan.pages.flatMap(p => p.boxes).filter(b => b.q === q).reduce((a, b) => a + (b.rect[3] - b.rect[1]), 0) : 0;
}
async function setBoxHeight(q, mm) {
  const e = state.exam;
  const graded = (e.sheets || []).length;
  if (graded && !confirm(`已经批过 ${graded} 份答卷。改了方框高度，之前打印的答题纸就和新版面对不上了。继续吗？`)) return false;
  const busy = h("div", { class: "busy" }, "正在重新排版…");
  document.body.append(busy);
  try {
    await api("POST", `/api/exams/${e.id}/answer-sheet`, { heights: { [q]: mm } });
    await loadExam(e.id);
    route();
    toast(mm ? `第${q}题方框改成 ${Math.round(mm)} 毫米，已重新排版` : `第${q}题方框恢复自动高度`);
    return true;
  } catch (er) { toast(er.message, true); return false; }
  finally { busy.remove(); }
}
function addGrips(pg, svg, view, W, H) {
  const gen = state.tmpl && state.tmpl.generated;
  if (!gen || !gen.plan || state.drawing) return;
  const f = W / gen.plan.size[0];                        // 模板像素 / 毫米
  const bottom = H - 14 * f;                             // 正文最低到这里（下面是页码和定位点）
  const fs = Math.round(Math.max(W, H) / 70);
  const pt = ev => { const b = view.getBoundingClientRect(); return (ev.clientY - b.top) * H / b.height; };
  for (const r of pg.regions) {
    if (r.kind !== "解答题" || (r.questions || []).length !== 1) continue;
    const q = r.questions[0];
    const [x0, y0, x1, y1] = r.rect;
    const box = s("rect", { class: "grip-box", x: x0, y: y0, width: x1 - x0, height: y1 - y0 });
    const label = s("text", { class: "grip-label", x: (x0 + x1) / 2, y: y1 - fs * 0.9, "font-size": fs, "text-anchor": "middle" });
    const grip = s("g", { class: "grip", "data-q": q },
      s("rect", { class: "hit", x: x0, y: y1 - 10, width: x1 - x0, height: 20 }),
      s("rect", { class: "bar", x: (x0 + x1) / 2 - 45, y: y1 - 6, width: 90, height: 12, rx: 5 }));
    grip.append(s("title", {}, `拖动改第${q}题方框的高度`));
    let startY = null, curY = y1;
    grip.addEventListener("pointerdown", ev => {
      ev.stopPropagation(); ev.preventDefault();
      startY = pt(ev); curY = y1;
      grip.setPointerCapture(ev.pointerId);
      svg.append(box, label);
    });
    grip.addEventListener("pointermove", ev => {
      if (startY === null) return;
      curY = Math.max(y0 + 25 * f, Math.min(bottom, y1 + pt(ev) - startY));
      box.setAttribute("height", curY - y0);
      grip.setAttribute("transform", `translate(0 ${curY - y1})`);      // 把手跟着鼠标走
      label.setAttribute("y", curY - fs * 0.9);
      label.textContent = `第${q}题 ${Math.round(boxTotalMM(q) + (curY - y1) / f)} 毫米`;
    });
    grip.addEventListener("pointerup", async ev => {
      if (startY === null) return;
      startY = null;
      const delta = (curY - y1) / f;
      if (Math.abs(delta) < 2) { box.remove(); label.remove(); return; }
      if (!(await setBoxHeight(q, Math.round(boxTotalMM(q) + delta)))) { box.remove(); label.remove(); route(); }
    });
    svg.append(grip);
  }
}

function renderSide(side, redraw) {
  const sel = state.layoutSel;
  const pg = sel && state.tmpl.pages.find(p => p.page === sel.page);
  const r = pg && pg.regions.find(x => x.id === sel.id);
  if (!r) {
    const sol = state.qs.filter(q => q.type === "解答题").map(q => q.number);
    const mapped = new Set(state.tmpl.pages.flatMap(p => p.regions.filter(x => x.kind === "解答题").flatMap(x => x.questions || [])));
    const missing = sol.filter(n => !mapped.has(n));
    const placed = new Set(state.tmpl.pages.flatMap(p => p.regions.flatMap(x => [...(x.slots || []).map(t => t.q), ...(x.groups || []).map(t => t.q)])));
    const objMissing = state.qs.filter(q => ["选择题", "填空题"].includes(q.type) && !placed.has(q.number)).map(q => q.number);
    fill(side, h("div", { class: "sheet stack" }, h("h3", {}, "区域"), h("p", { class: "muted" }, "点图上的一个框来修改它。填空题的每条横线、选择题的每组选项框会用细框标出来。"),
      missing.length ? h("div", { class: "notice" }, `这些解答题还没有对应的作答区域：第 ${missing.join("、")} 题。选中一个框填上题号，或者“画一个新区域”。`) : h("p", {}, "所有解答题都已经有对应的作答区域。"),
      objMissing.length ? h("div", { class: "notice" }, `这些选择/填空题在答题纸上还没有找到作答位置：第 ${objMissing.join("、")} 题。点开对应的区域检查题型和题号。`) : null));
    return;
  }
  const qs = h("input", { type: "text", value: (r.questions || []).join(","), oninput: ev => { r.questions = ev.target.value.split(/[,，、\s]+/).map(x => x.trim()).filter(Boolean); } });
  fill(side, h("div", { class: "sheet stack" },
    h("div", { class: "row spread" }, h("h3", {}, `区域 ${r.id}（第 ${pg.page + 1} 页）`), h("button", { class: "ghost", onclick: () => { state.layoutSel = null; redraw(); } }, "完成")),
    r.header ? h("div", { class: "muted" }, `印刷标题：${r.header}`) : null,
    h("label", { class: "field" }, h("span", {}, "题型"), h("select", { onchange: ev => { r.kind = ev.target.value; redraw(); } },
      KINDS.map(k => h("option", { value: k, selected: r.kind === k || null }, k)))),
    h("label", { class: "field" }, h("span", {}, "题号（多个用逗号分开）"), qs, h("div", { class: "hint" }, "一道题写在两个框里时，两个框都填这个题号；一个框里有两道题就都填上。")),
    r.kind === "解答题" && !isHW() ? h("label", { class: "field" }, h("span", {}, "分值（选填）"), h("input", { type: "number", min: "0", step: "0.5", value: r.full_score ?? "", oninput: ev => (r.full_score = ev.target.value === "" ? null : +ev.target.value) })) : null,
    r.kind === "解答题" && state.tmpl.generated && (r.questions || []).length === 1 ? (() => {
      const q = r.questions[0];
      const custom = ((state.tmpl.generated.plan || {}).heights || {})[q];
      const inp = h("input", { type: "number", min: "25", max: "800", step: "5", value: Math.round(boxTotalMM(q)), style: "width:90px" });
      return h("div", { class: "field" }, h("span", {}, `第${q}题方框高度（毫米，放不下会接到下一栏）`),
        h("div", { class: "row" }, inp, h("button", { onclick: () => setBoxHeight(q, +inp.value) }, "改"),
          custom ? h("button", { class: "ghost", onclick: () => setBoxHeight(q, null) }, "恢复自动") : null),
        h("div", { class: "hint" }, custom ? "这个高度是老师定的。" : "现在是按分值自动给的高度。也可以直接在图上拖方框的下边框。"));
    })() : null,
    ["填空题", "选择题"].includes(r.kind) ? slotEditor(r, redraw) : null,
    h("div", { class: "row" }, h("button", { onclick: () => redraw() }, "应用"), h("button", { class: "red", onclick: () => { pg.regions.splice(pg.regions.indexOf(r), 1); state.layoutSel = null; redraw(); } }, "删除这个区域"))));
}

function slotEditor(r, redraw) {
  const isFill = r.kind === "填空题";
  const items = (isFill ? r.slots : r.groups) || [];
  const what = isFill ? "条横线" : "道题的选项框";
  const box = h("div", { class: "stack" },
    h("div", {}, items.length ? `找到 ${items.length} ${what}` + (isFill ? "" : `（每题 ${items[0].options.length} 个选项）`) : h("span", { class: "badge red" }, `没找到${isFill ? "横线" : "选项框"}，批改时会把整块区域交给模型识别`)),
    items.length ? h("div", {}, h("div", { class: "muted small", style: "margin-bottom:4px" }, `按${isFill ? "横线" : "行"}的位置（先行后列）对应的题号：`),
      h("div", { class: "slotlist" }, items.map((it, k) => h("label", { class: "slotitem" }, h("small", {}, `${isFill ? "横线" : "第"}${k + 1}${isFill ? "" : "行"}`),
        h("input", { type: "text", value: it.q || "", "aria-label": `${isFill ? "横线" : "第"}${k + 1}${isFill ? "" : "行"}的题号`, oninput: ev => { it.q = ev.target.value.trim(); } }))))) : null,
    h("div", { class: "hint" }, "题号是从答题纸的文字读出来的（扫描件按先行后列的顺序排）。不对的直接改，然后点“应用”“保存版面”。"),
    h("button", { type: "button", onclick: () => { r.redetect = true; toast("点“保存版面”后会重新找横线和选项框"); } }, `重新找${isFill ? "横线" : "选项框"}`));
  return box;
}

async function saveTemplate() {
  try {
    state.exam = await api("PUT", `/api/exams/${state.exam.id}/template`, { pages: state.tmpl.pages.map(p => ({ page: p.page, regions: p.regions })) });
    (state.exam.template.pages || []).forEach(p => p.regions.forEach(r => delete r.redetect));
    state.tmpl = JSON.parse(JSON.stringify(state.exam.template));
    toast((state.exam.sheets || []).length ? "版面已保存。已批的答卷要在“上传扫描件”里点“重新给分”才会按新版面裁切" : "版面已保存");
    route();
  } catch (e) { toast(e.message, true); }
}

// ------------------------------------------------------------------ 第 3 步：上传扫描件
function renderScans(body) {
  const e = state.exam;
  const nums = e.graded_numbers || [];
  if (!e.template || !nums.length) {
    fill(body, h("div", { class: "sheet empty" }, h("h2", {}, "还不能批改"),
      h("p", {}, !e.template ? "先在“答题纸版面”里识别版面（或者按题目生成答题纸）。"
        : isHW() ? "版面上还没有标出题号，或者选择、填空题还没有参考答案（可以在“题目与评分细则”里让 AI 做）。" : "版面上还没有标出解答题的题号，或者题目没有分值。"),
      h("a", { class: "btn", href: `#/exam/${e.id}/${!e.template ? "layout" : "questions"}` }, !e.template ? "去识别版面" : "去检查题目")));
    return;
  }
  const unconfirmed = state.qs.filter(q => ["解答题", "选择题", "填空题"].includes(q.type) && !q.confirmed && (q.answer || "").trim()
    && (q.answer_source === "ai" || (q.checks || []).length)).map(q => q.number);
  const input = h("input", { type: "file", multiple: true, accept: ".pdf,.png,.jpg,.jpeg,.tif,.tiff,.heic,.heif", hidden: true, onchange: () => showFiles(input.files) });
  const list = h("div", { class: "files" });
  const showFiles = files => { list.textContent = files.length ? `已选择 ${files.length} 个文件（共 ${mb([...files].reduce((a, f) => a + f.size, 0))}）：` + [...files].map(f => f.name).slice(0, 12).join("、") + (files.length > 12 ? "……" : "") : ""; };
  const zone = h("div", { class: "dropzone", ondragover: ev => { ev.preventDefault(); zone.classList.add("over"); }, ondragleave: () => zone.classList.remove("over"),
    ondrop: ev => { ev.preventDefault(); zone.classList.remove("over"); input.files = ev.dataTransfer.files; showFiles(input.files); } },
    h("p", {}, "把扫描件拖到这里，或者 ", h("button", { type: "button", onclick: () => input.click() }, "选择文件")),
    h("div", { class: "hint" }, `PDF（扫描仪导出的整叠多页 PDF 也行，几百页、几百 MB 都可以，会一页一页处理）、JPG、PNG、TIFF 都可以，手机拍的照片也行。每份答题纸 ${e.pages_per_sheet || 1} 张图，按扫描顺序排好即可；倒放、歪斜都没关系，空白页会自动跳过，缺页的答卷会标出来。`),
    input, list);
  const go = h("button", { class: "primary", onclick: async () => {
      if (!input.files.length) { toast("请先选择扫描件", true); return; }
      const fd = new FormData(); [...input.files].forEach(f => fd.append("files", f));
      go.disabled = true; go.textContent = "上传中…";
      try {
        const r = await upload(`/api/exams/${e.id}/scans`, fd, (a, b) => {
          go.textContent = a >= b ? "上传完成，正在保存…" : `上传中 ${Math.floor(a / b * 100)}%（${mb(a)} / ${mb(b)}）`;
        });
        watchJob(r.job, j => {
          if (j.status === "done" && j.result && j.result.sheets === 0) { toast("这次上传没有分出答卷，原因见“最近一次上传”", true); route(); }
          else location.hash = `#/exam/${e.id}/review`;
        }); input.value = ""; showFiles([]);
      }
      catch (er) { toast(er.message, true); }
      go.disabled = false; go.textContent = "开始批改";
    } }, "开始批改");
  const st = state.settings;
  fill(body, scanReportBox(e, null, e.scan_report && !e.scan_report.sheets),
    h("div", { class: "sheet stack" }, h("h2", {}, "上传学生答题纸"),
      h("p", { class: "muted" }, (() => {
        const t = n => (state.qs.find(q => q.number === n) || {}).type;
        const parts = [];
        const ch = nums.filter(n => t(n) === "选择题"), fi = nums.filter(n => t(n) === "填空题"), so = nums.filter(n => !["选择题", "填空题"].includes(t(n)));
        if (ch.length) parts.push(`选择题 ${ch.length} 道（按涂卡识别）`);
        if (fi.length) parts.push(`填空题 ${fi.length} 道`);
        if (so.length) parts.push(`解答题第 ${so.join("、")} 题`);
        return "将批改：" + parts.join("，") + "。";
      })()),
      unconfirmed.length ? h("div", { class: "notice" }, `第 ${unconfirmed.join("、")} 题的参考答案是 AI 生成的或者自动核对有疑问，还没确认。建议先到“题目与评分细则”里核对。`) : null,
      zone, h("div", { class: "row" }, go, h("a", { href: "#/settings", class: "muted" }, "批改设置（双评、模型、并发）")),
      h("details", { class: "hint" }, h("summary", {}, "用手机拍照片代替扫描？"), h("p", { style: "margin:6px 0 0" }, PHOTO_TIP))),
    (e.files.scans || []).length ? h("div", { class: "sheet" }, h("h3", {}, `已上传 ${(e.files.scans || []).length} 个文件`),
      h("div", { class: "files" }, e.files.scans.map(f => f.split("/").pop()).join("、")),
      h("div", { class: "row", style: "margin-top:12px" },
        h("button", { onclick: () => regradeAll({}) }, isHW() ? "按最新的答案重新批改" : "按最新的评分细则重新给分"),
        h("button", { onclick: () => regradeAll({ retranscribe: true }) }, isHW() ? "重新识别手写并批改" : "重新识别手写并给分"),
        h("span", { class: "hint" }, "改了评分细则或版面后用第一个：识别好的转写会保留，版面改过的答卷会先按新版面重新裁切。"))) : null);
}

const PHOTO_TIP = "手机拍也可以：整张答题纸都拍进画面，四个角的黑方块都要拍到；尽量正对着拍、纸放平、光线均匀，不要有手指或阴影挡住。"
  + "每张照片一页，按顺序上传（JPG、HEIC，或者在电脑上把照片导出成一个 PDF 都行）。拍得歪一点、纸有点弯没关系，程序会按印刷的线条拉平。";

function scanReportBox(e, linkIds, open) {   // 最近一次上传的情况：几页、分成几份、跳过了哪些页、为什么
  const rep = e.scan_report;
  if (!rep) return null;
  const notes = rep.notes || [];
  const head = `最近一次上传：${rep.pages} 页，分成 ${rep.sheets} 份答卷`;
  if (!notes.length) return h("p", { class: "muted", style: "margin:0 0 12px" }, head + "，没有发现缺页或对不上的页。");
  const pending = notes.filter(n => !n.startsWith("✓"));          // ✓ 开头的是已经合并/删除处理过的
  const link = linkIds || (x => x);
  return h("details", { class: !pending.length ? "notice done" : rep.sheets ? "notice" : "notice err", style: "margin-bottom:12px",
      open: open || (pending.length && notes.length <= 8) || null },
    h("summary", {}, h("b", {}, head), rep.skipped ? `，跳过 ${rep.skipped} 页` : "", rep.partial ? `，${rep.partial} 份缺页` : "",
      pending.length ? `（${pending.length} 条需要看一下）` : "（提醒都已处理）"),
    h("ul", {}, notes.map(n => h("li", { class: n.startsWith("✓") ? "muted" : null }, link(n)))));
}

async function regradeAll(opts) {
  if (!confirm(`重新批改所有答卷？老师改过的${isHW() ? "对错" : "分数"}会保留。`)) return;
  try { const r = await api("POST", `/api/exams/${state.exam.id}/regrade`, opts); watchJob(r.job); } catch (e) { toast(e.message, true); }
}

// ------------------------------------------------------------------ 第 4 步：复核与成绩
async function renderReview(body, sub) {
  const e = state.exam;
  state.results = await api("GET", `/api/exams/${e.id}/results`);
  const res = state.results;
  if (!res.rows.length) {
    const none = e.scan_report && !e.scan_report.sheets;      // 传过了，但一份答卷都没分出来
    fill(body, none ? scanReportBox(e, null, true) : null,
      h("div", { class: "sheet empty" }, h("h2", {}, none ? "这次上传没有分出答卷" : "还没有批改结果"),
        h("p", {}, none ? "上传的页都没能用上，原因见上面。" : "上传学生答题纸的扫描件后，结果会出现在这里。"),
        none ? h("p", { class: "muted" }, PHOTO_TIP) : null,
        h("a", { class: "btn primary", href: `#/exam/${e.id}/scans` }, "上传扫描件")));
    return;
  }
  const nums = res.numbers;
  const types = res.types || {};
  const hw = res.scored === false;          // 平时作业：只有对 / 半对 / 错
  const subj = nums.filter(n => !["选择题", "填空题"].includes(types[n]));
  const objSecs = (res.sections || []).filter(t => t !== "解答题");
  const isReview = r => r.needs_review.length || !r.align_ok || (r.missing_pages || []).length || (r.id_issues || []).length;
  const secAvg = t => res.rows.length ? Math.round(res.rows.reduce((a, r) => a + (r.sections[t] || 0), 0) / res.rows.length * 10) / 10 : null;
  const pct = (a, m) => a == null || !m ? "–" : `${Math.round(a / m * 100)}%`;
  const secReview = (r, t) => nums.some(n => types[n] === t && ["review", "error"].includes(r.states[n]));
  const stats = h("div", { class: "row", style: "margin-bottom:12px" });
  const drawStats = () => fill(stats,
    objSecs.map(t => h("span", { class: "badge" }, hw ? `${t} 正确率 ${pct(secAvg(t), res.section_max[t])}` : `${t} 平均 ${fmt(secAvg(t))}/${fmt(res.section_max[t])}`)),
    subj.map(n => { const s = res.stats[n]; return h("span", { class: "badge" },
      hw ? `第${n}题 对 ${s.right}·半对 ${s.half}·错 ${s.wrong}` : `第${n}题 平均 ${fmt(s.avg)}/${fmt(s.max)}`, s.review ? `，待复核 ${s.review}` : ""); }),
    h("span", { style: "margin-left:auto" }),
    h("a", { class: "btn", href: `/api/exams/${e.id}/export.csv` }, "导出成绩表（CSV）"),
    h("a", { class: "btn", href: `/api/exams/${e.id}/export.zip` }, "下载批改后的答题纸"));
  // 搜索：姓名、学号、班级、答卷编号、拼音全拼或首字母（zs → 张三）
  const matches = (r, q) => {
    if (!q) return true;
    const s = r.student || {};
    const hay = [s.name, s.student_id, s.class, r.id, r.py, r.pyi].filter(Boolean).map(x => String(x).toLowerCase());
    return hay.some(x => x.includes(q));
  };
  const sidNum = r => { const d = String((r.student || {}).student_id || "").replace(/\D/g, ""); return d ? +d : Infinity; };
  const SORTS = { scan: ["扫描顺序", null], sid: ["学号", (a, b) => sidNum(a) - sidNum(b) || a.id.localeCompare(b.id)],
    name: ["姓名", (a, b) => (a.py || a.student.name || "~").localeCompare(b.py || b.student.name || "~", "zh-CN")],
    high: [hw ? "做对从多到少" : "总分从高到低", (a, b) => b.total - a.total], low: [hw ? "做对从少到多" : "总分从低到高", (a, b) => a.total - b.total] };
  state.sort = state.sort && SORTS[state.sort] ? state.sort : "scan";
  const filtered = () => {
    const q = (state.q || "").trim().toLowerCase();
    const rows = res.rows.filter(r => (state.filter === "all" ? true : state.filter === "review" ? isReview(r) : Object.values(r.states).includes("teacher")) && matches(r, q));
    const cmp = SORTS[state.sort][1];
    return cmp ? rows.slice().sort(cmp) : rows;
  };
  const listEl = h("div", {});
  const filtersEl = h("div", { class: "filters" });
  const search = h("input", { type: "search", placeholder: "搜索：姓名、学号或拼音", title: "姓名、学号、班级、答卷编号都可以搜；拼音全拼或首字母也行，比如 zs 找张三", value: state.q || "", "aria-label": "搜索学生",
    oninput: () => { state.q = search.value; drawRoster(); },
    onkeydown: ev => {
      if (ev.key === "Enter") { const f = filtered()[0]; if (f) { ev.preventDefault(); openSheet(f.id); } }
      else if (ev.key === "Escape") { search.value = ""; state.q = ""; drawRoster(); search.blur(); }
    } });
  const sortSel = h("select", { "aria-label": "排序", onchange: () => { state.sort = sortSel.value; drawRoster(); } },
    Object.entries(SORTS).map(([k, [t]]) => h("option", { value: k, selected: state.sort === k || null }, t)));
  state.focusSearch = () => { search.focus(); search.select(); };
  const roster = h("div", { class: "roster" }, h("div", { class: "tools" }, filtersEl, search, h("label", { class: "sortby" }, "排序", sortSel)), listEl);
  const detail = h("div", {});
  const drawRoster = () => {
    const rows = filtered();
    fill(filtersEl, [["all", `全部 ${res.rows.length}`], ["review", `待复核 ${res.rows.filter(isReview).length}`], ["teacher", hw ? "已改判" : "已改分"]].map(([k, t]) =>
      h("button", { class: state.filter === k ? "on" : "", onclick: () => { state.filter = k; drawRoster(); } }, t)));
    fill(listEl,
      !rows.length ? h("div", { class: "nohit" }, state.q ? `没有找到“${state.q}”` : "没有符合条件的答卷") : null,
      ...rows.map(r => h("a", { href: `#/exam/${e.id}/review/${r.id}`, class: r.id === state.sid ? "on" : "", onclick: ev => { ev.preventDefault(); openSheet(r.id); } },
        h("div", { class: "who" }, h("span", {}, h("b", {}, r.student.name || r.id), r.student.name ? h("small", {}, " " + [r.student.student_id, r.id].filter(Boolean).join(" · ")) : null),
          h("span", { class: "num", title: hw ? "做对的题数（半对算半题）" : "总分" }, hw ? `对 ${fmt(r.total)}/${fmt(r.max_total)}` : `${fmt(r.total)}`)),
        h("div", { class: "chips" }, !r.align_ok ? h("span", { class: "chip align_failed" }, "对齐失败") : null,
          (r.missing_pages || []).length ? h("span", { class: "chip align_failed", title: "扫描件里没找到这一面" }, `缺${r.missing_text}`) : null,
          (r.id_issues || []).length ? h("span", { class: "chip review", title: r.id_issues.join("\n") }, "学号待核对") : null,
          objSecs.map(t => h("span", { class: "chip " + (secReview(r, t) ? "review" : ""), title: hw ? `${t}做对几题` : `${t}合计` },
            `${t.slice(0, 2)} ${fmt(r.sections[t])}` + (hw ? `/${fmt(res.section_max[t])}` : ""))),
          subj.map(n => { const v = verdictOf(r.scores[n]);
            return h("span", { class: "chip " + (r.states[n] || "") + (hw && v ? " v-" + VCLS[v] : ""), title: `第${n}题 ${hw && v ? v + " " : ""}${STATE_TEXT[r.states[n]] || ""}` },
              `${n}题 ${hw ? (v ? MARK[v] : "–") : fmt(r.scores[n])}`); })))));
    drawStats();
  };
  async function openSheet(sid, num) {
    state.sid = sid;
    history.replaceState(null, "", `#/exam/${e.id}/review/${sid}`);
    drawRoster();
    const sheet = await api("GET", `/api/exams/${e.id}/sheets/${sid}`);
    state.sheet = sheet;
    const keys = tabKeys(res);
    const needs = k => (k === "填空题" || k === "选择题") ? nums.some(n => types[n] === k && ((sheet.questions[n] || {}).ai || {}).needs_review)
      : ((sheet.questions[k] || {}).ai || {}).needs_review;
    if (!num || !keys.includes(num)) num = keys.includes(state.num) ? state.num : (keys.find(needs) || keys[0]);
    state.num = num;
    drawSheet(detail, sheet, nums, async () => {
      state.results = await api("GET", `/api/exams/${e.id}/results`);
      Object.assign(res, state.results);
      drawRoster();
    }, openSheet);
  }
  state.openSheet = openSheet;
  state.rosterIds = () => filtered().map(r => r.id);
  // 最近一次上传的情况：几页、分成几份、跳过的空白页、缺页的答卷（答卷编号可以点）
  const ids = new Set(res.rows.map(r => r.id));
  const linkIds = text => String(text).split(/(S\d{3,})/).map(part => ids.has(part)
    ? h("a", { href: `#/exam/${e.id}/review/${part}`, onclick: ev => { ev.preventDefault(); openSheet(part); } }, part) : part);
  const repBox = scanReportBox(e, linkIds);
  // 对照名单、没交的人
  const absent = res.absent || [];
  const rinfo = res.roster || {};
  const bindSel = h("select", { "aria-label": "对照哪个班的名单", onchange: async () => {
      try { await api("PUT", `/api/exams/${e.id}/roster`, { classes: bindSel.value ? [bindSel.value] : [] }); toast("已按名单重新对照学生"); await loadExam(e.id); route(); }
      catch (er) { toast(er.message, true); } } });
  api("GET", "/api/rosters").then(list => fill(bindSel, h("option", { value: "" }, "自动（按学号在全部名单里找）"),
    list.map(c => h("option", { value: c.id, selected: (rinfo.bound || []).includes(c.id) || null }, `${c.name}（${c.count} 人）`)))).catch(() => {});
  const rosterBar = rinfo.n ? h("div", { class: "row rosterbar" }, h("span", { class: "muted" }, "对照名单"), bindSel,
    absent.length ? h("details", { class: "absent" }, h("summary", {}, `名单里还有 ${absent.length} 人没有答卷`),
      h("div", { class: "muted" }, absent.map(a => `${a.name}（${a.sid}）`).join("、"), "。可能没交，或者扫描时漏了；也可能是学号涂错了，看一下标“学号待核对”的答卷。"))
      : (rinfo.classes || []).length ? h("span", { class: "muted" }, `${rinfo.classes.join("、")}名单上的人都有答卷`) : null) : null;
  fill(body, repBox, rosterBar, stats, h("div", { class: "review-layout" }, roster, detail));
  drawRoster();
  const first = sub && res.rows.find(r => r.id === sub) ? sub : (state.sid && res.rows.find(r => r.id === state.sid) ? state.sid : res.rows[0].id);
  await openSheet(first);
}

function drawSheet(el, sheet, nums, refresh, openSheet) {
  const e = state.exam;
  const stu = sheet.student || {};
  const nameIn = h("input", { type: "text", value: stu.name || "", placeholder: "姓名", style: "width:120px" });
  const idIn = h("input", { type: "text", value: stu.student_id || "", placeholder: "学号", style: "width:140px" });
  const clsIn = h("input", { type: "text", value: stu.class || "", placeholder: "班级", style: "width:110px" });
  const SRC = { roster: "涂的学号对上了名单", roster_name: "按手写姓名对上了名单", omr: "涂的学号", qr: "答题纸上印的学号", ai: "AI 认的", teacher: "老师填的" };
  const who = h("div", { class: "row", style: "margin-bottom:12px" }, h("h2", {}, sheet.id), nameIn, idIn, clsIn,
    h("button", { onclick: async () => { try { await api("PUT", `/api/exams/${e.id}/sheets/${sheet.id}/student`, { name: nameIn.value, student_id: idIn.value, class: clsIn.value }); toast("已保存"); route(); } catch (er) { toast(er.message, true); } } }, "保存"),
    stu.source && SRC[stu.source] ? h("small", { class: "muted", title: (stu.omr && stu.omr.id) ? `涂卡读到：${stu.omr.id}` : "" }, SRC[stu.source]) : null,
    (stu.qr || []).length ? h("small", { title: "答题纸上识别到的二维码" }, "二维码：" + stu.qr.join("，")) : null,
    h("span", { style: "margin-left:auto" }),
    (sheet.annotated || []).map((a, k) => h("button", { onclick: () => lightbox(fileUrl(a)) }, `看批改后的第 ${k + 1} 页`)),
    !sheet.align_ok ? (sheet.pages || []).map(p => h("button", { class: "red", onclick: () => lightbox(fileUrl(p.display)) }, `原图 ${p.file.split("/").pop()}`)) : null,
    h("button", { class: "ghost", title: "扫错了、重复扫了的答卷可以删掉（原始扫描件不删）", onclick: async () => {
      if (!confirm(`删除答卷 ${sheet.id}${stu.name ? "（" + stu.name + "）" : ""}？它的批改结果和老师改的分都会删掉。`)) return;
      try { await api("DELETE", `/api/exams/${e.id}/sheets/${sheet.id}`); toast(`已删除 ${sheet.id}`); state.sid = null; await loadExam(e.id); route(); }
      catch (er) { toast(er.message, true); } } }, "删除这份"));
  const res = state.results;
  const myRow = (res.rows || []).find(r => r.id === sheet.id) || {};
  const idBox = (myRow.id_issues || []).length ? h("div", { class: "notice", style: "margin-bottom:12px" },
    h("b", {}, "学号待核对："), myRow.id_issues.join("；"), "。看一眼答题纸上的姓名，在上面改好姓名和学号，点“保存”。",
    sheet.pages && sheet.pages[0] ? h("button", { class: "ghost", style: "margin-left:8px", onclick: () => lightbox(fileUrl(sheet.pages[0].display)) }, "看第 1 页") : null) : null;
  const miss = sheet.missing_pages || [];
  let missBox = null;
  if (miss.length) {   // 缺页：说明原因；有“只含缺的那几页”的答卷（比如补扫上传的）就可以并进来
    const cands = (res.rows || []).filter(r => r.id !== sheet.id && (r.present_pages || []).length && r.present_pages.every(p => miss.includes(p)))
      .sort((a, b) => b.id.localeCompare(a.id, "en", { numeric: true }));   // 新上传的（多半是补扫的）排前面
    const sel = h("select", { style: "width:auto;max-width:100%" }, cands.map(r => h("option", { value: r.id },
      `${r.id}${r.student.name ? " " + r.student.name : ""}（只有${r.present_text}${r.first_page ? "，从" + r.first_page + "开始" : ""}）`)));
    const peek = async () => {   // 合并前看一眼那一页，确认是同一个学生的
      try { const s2 = await api("GET", `/api/exams/${e.id}/sheets/${sel.value}`); const p = (s2.pages || [])[0]; if (p && p.display) lightbox(fileUrl(p.display)); }
      catch (er) { toast(er.message, true); }
    };
    const merge = async () => {
      if (!confirm(`把 ${sel.value} 并进 ${sheet.id}？${sel.value} 会合并成这一份，不再单独列出。`)) return;
      try {
        const r = await api("POST", `/api/exams/${e.id}/sheets/${sheet.id}/merge`, { source: sel.value });
        toast(`已把 ${sel.value} 并进 ${sheet.id}`);
        await loadExam(e.id);
        if (r.job) watchJob(r.job, () => route()); else route();
      } catch (er) { toast(er.message, true); }
    };
    missBox = h("div", { class: "notice", style: "margin-bottom:12px" },
      h("b", {}, `这份答卷缺${myRow.missing_text || "页"}`), "：那一面上的题没有作答图片，也没有给分。",
      "常见原因是扫描时漏了一张、两张纸粘在一起进了扫描仪，或者扫描仪开了“跳过空白页”，把学生没写字的一面跳过了。",
      "找到原卷补扫上传后可以在这里合并；如果那一面学生确实没写，直接在打分条上给 0 分即可。",
      cands.length ? h("div", { class: "row", style: "margin-top:8px;flex-wrap:wrap" }, h("span", {}, "把"), sel, h("button", { onclick: peek }, "先看一眼"),
        h("span", {}, "并进这份"), h("button", { class: "primary", onclick: merge }, "合并")) : null);
  }
  const types = res.types || {};
  const hw = res.scored === false;
  const finalOf = qr => qr.teacher && qr.teacher.score != null ? qr.teacher.score : qr.ai ? qr.ai.score : null;
  const tabs = h("div", { class: "qtabs" }, tabKeys(res).map(k => {
    let label, sc;
    if (k === "填空题" || k === "选择题") {
      const ns = nums.filter(n => types[n] === k);
      sc = ns.reduce((a, n) => a + (+finalOf(sheet.questions[n] || {}) || 0), 0);
      const flagged = ns.filter(n => ((sheet.questions[n] || {}).ai || {}).needs_review && !((sheet.questions[n] || {}).teacher || {}).score).length;
      label = [k, " ", h("b", { class: "num" }, (hw ? "对 " : "") + `${fmt(sc)}/${fmt(res.section_max[k])}`), flagged ? h("span", { class: "dot", title: `${flagged} 题待复核` }) : null];
    } else if (hw) {
      const v = verdictOf(finalOf(sheet.questions[k] || {}));
      label = [`第${k}题 `, h("b", { class: v ? "v-" + VCLS[v] : "" }, v ? MARK[v] : "–")];
    } else {
      label = [`第${k}题 `, h("b", { class: "num" }, fmt(finalOf(sheet.questions[k] || {})))];
    }
    return h("button", { class: k === state.num ? "on" : "", onclick: () => { state.num = k; drawSheet(el, sheet, nums, refresh, openSheet); } }, label);
  }));
  const after = async (s2) => { Object.assign(sheet, s2); await refresh(); drawSheet(el, sheet, nums, refresh, openSheet); };
  const view = (state.num === "填空题" || state.num === "选择题")
    ? objectiveView(sheet, state.num, nums.filter(n => types[n] === state.num), after)
    : questionView(sheet, state.num, sheet.questions[state.num] || {}, after);
  fill(el, who, !sheet.align_ok ? h("div", { class: "notice err", style: "margin-bottom:12px" }, "这份答题纸有页面没能和模板对齐（可能放错了答题纸、混进了别的纸或扫描严重变形），请看原图人工批改，在下面的打分条上给分。") : null,
    idBox, missBox, tabs, view,
    h("p", { class: "legend", style: "margin-top:14px" }, "快捷键：", h("span", { class: "kbd" }, "/"), " 搜索学生，", h("span", { class: "kbd" }, "↑"), h("span", { class: "kbd" }, "↓"), " 换学生，", h("span", { class: "kbd" }, "←"), h("span", { class: "kbd" }, "→"), " 换题，", h("span", { class: "kbd" }, "Enter"), hw ? " 确认 AI 的判断。" : " 确认 AI 分数。"));
}

function questionView(sheet, num, qr, after) {
  const e = state.exam;
  const ai = qr.ai || null;
  const teacher = qr.teacher && qr.teacher.score != null ? qr.teacher : null;
  const mx = +(qr.max || (ai && ai.max) || 0);
  const step = (state.settings && +state.settings.score_step) || 1;
  const final = teacher ? teacher.score : ai ? ai.score : null;
  const put = async body => { try { after(await api("PUT", `/api/exams/${e.id}/sheets/${sheet.id}/q/${encodeURIComponent(num)}/teacher`, body)); const hwq = state.results && state.results.scored === false;
    toast(body.clear ? (hwq ? "已撤销改判" : "已撤销改分") : body.accept ? "已确认" : hwq ? `已改判为“${verdictOf(body.score / (mx || 1))}”` : "已改分"); } catch (er) { toast(er.message, true); } };
  state.acceptCurrent = ai && !teacher ? () => put({ accept: true }) : null;

  const crops = h("div", { class: "stack" }, (qr.crops || []).length ? qr.crops.map(c => h("div", { class: "crop" }, h("img", { src: fileUrl(c), alt: `第${num}题作答`, onclick: () => lightbox(fileUrl(c)) })))
    : h("div", { class: "sheet empty" }, qr.status === "align_failed" ? "没有对齐，无法裁切。"
      : qr.status === "missing" ? "这份答卷缺这道题所在的那一页，没有作答图片。学生确实没写的话，直接在下面给 0 分。" : "没有找到这道题的作答区域。"));

  const cells = [];
  for (let v = mx; v >= -1e-9; v -= step) cells.push(Math.round(v * 100) / 100);
  const bar = h("div", { class: "scorebar", role: "group", "aria-label": "打分条" }, cells.map(v => h("button", {
    class: (ai && Math.abs(ai.score - v) < 1e-6 ? "ai " : "") + (final !== null && Math.abs(final - v) < 1e-6 ? "mark" : ""),
    title: `给 ${v} 分`, onclick: () => put({ score: v }) }, String(v))));

  const status = qr.status === "error" ? h("div", { class: "notice err" }, "批改失败：" + (qr.error || ""), " ",
      h("button", { onclick: async () => { try { const r = await api("POST", `/api/exams/${e.id}/regrade`, { sheet_ids: [sheet.id], numbers: [num] }); watchJob(r.job, () => route()); } catch (er) { toast(er.message, true); } } }, "重试"))
    : ["pending", "grading", "transcribing"].includes(qr.status) ? h("div", { class: "notice" }, STATE_TEXT[qr.status] + "…") : null;
  const conf = ai ? h("span", { class: "badge" + (ai.confidence === "low" ? " red" : "") }, { high: "把握大", medium: "把握一般", low: "把握小" }[ai.confidence] || "") : null;
  const reviewNote = ai && ai.needs_review && !teacher ? h("div", { class: "notice" }, h("b", {}, "建议复核"), ai.review_notes ? "：" + ai.review_notes : "") : null;
  if (state.results && state.results.scored === false) {      // 平时作业：判 对 / 半对 / 错，写错在哪
    const v = verdictOf(final);
    const aiV = ai ? (ai.verdict || verdictOf(ai.score)) : null;
    const vbtns = h("div", { class: "verdicts", role: "group", "aria-label": "判对错" },
      [["对", 1], ["半对", 0.5], ["错", 0]].map(([t, val]) => h("button", {
        class: `vb v-${VCLS[t]}` + (aiV === t ? " ai" : "") + (v === t ? " mark" : ""), title: t === "半对" ? "思路对但有错误、过程不完整，或者几个小问有对有错" : `判${t}`,
        onclick: () => put({ score: val * (mx || 1) }) }, `${MARK[t]} ${t}`)));
    const hwBox = h("div", { class: "sheet stack" },
      h("div", { class: "row spread" },
        h("div", { class: "score-big " + (v ? "v-" + VCLS[v] : "") }, v ? `${MARK[v]} ${v}` : "–"),
        h("div", { class: "row" }, teacher ? h("span", { class: "badge ok" }, teacher.accepted ? "已确认 AI 的判断" : "老师改判") : ai ? h("span", { class: "badge" }, "AI 判") : null, conf)),
      reviewNote, status,
      h("div", {}, vbtns, h("div", { class: "legend", style: "margin-top:4px" }, "点一下就改判。", h("i"), "是 AI 的判断，红笔是最终结果。")),
      h("div", { class: "row" },
        ai && !teacher ? h("button", { class: "primary", onclick: () => put({ accept: true }) }, "确认 AI 的判断") : null,
        teacher ? h("button", { onclick: () => put({ clear: true }) }, "撤销改判，用回 AI 的判断") : null));
    const why = ai && (ai.error || ai.comment || (ai.parts || []).length) ? h("div", { class: "sheet stack" },
      ai.error ? h("p", { style: "margin:0" }, h("b", {}, "错在哪："), ai.error) : null,
      ai.comment ? h("p", { style: "margin:0" }, h("b", {}, "订正提示："), ai.comment) : null,
      (ai.parts || []).length ? h("table", { class: "parts-t" },
        h("thead", {}, h("tr", {}, h("th", {}, "小问"), h("th", {}, "AI 判断"), h("th", {}, "说明"))),
        h("tbody", {}, ai.parts.map(p => h("tr", {}, h("td", {}, p.label),
          h("td", {}, p.verdict ? h("span", { class: "v-" + VCLS[p.verdict] }, `${MARK[p.verdict]} ${p.verdict}`) : "–"), h("td", {}, p.comment || ""))))) : null) : null;
    return h("div", { class: "qview" }, crops, h("div", { class: "stack" }, hwBox, why, transcriptView(sheet, num, qr)));
  }
  const source = teacher ? (teacher.accepted ? h("span", { class: "badge ok" }, "已确认 AI 分数") : h("span", { class: "badge ok" }, "老师改分")) : ai ? h("span", { class: "badge" }, "AI 给分") : null;
  const scoreBox = h("div", { class: "sheet stack" },
    h("div", { class: "row spread" },
      h("div", { class: "score-big" }, fmt(final), h("small", {}, ` / ${fmt(mx)} 分`)),
      h("div", { class: "row" }, source, conf)),
    reviewNote,
    status,
    h("div", {}, bar, h("div", { class: "legend", style: "margin-top:4px" }, "点一格就给这个分。", h("i"), "是 AI 给的分，红笔是最终分。")),
    h("div", { class: "row" },
      ai && !teacher ? h("button", { class: "primary", onclick: () => put({ accept: true }) }, "确认 AI 分数") : null,
      teacher ? h("button", { onclick: () => put({ clear: true }) }, "撤销改分，用回 AI 分数") : null,
      ai && ai.samples && ai.samples.length > 1 ? h("small", {}, `${ai.samples.length} 次独立评分：${ai.samples.join("、")}`) : null));

  const partsBox = ai && ai.parts ? h("div", { class: "sheet" }, h("table", { class: "parts-t" },
    h("thead", {}, h("tr", {}, h("th", {}, "小问"), h("th", {}, "AI 给分"), h("th", {}, "说明"))),
    h("tbody", {}, ai.parts.map(p => h("tr", {}, h("td", {}, p.label), h("td", { class: "num" }, `${fmt(p.score)} / ${fmt(p.max)}`),
      h("td", {}, p.comment || "", (p.points || []).length ? h("details", { class: "points" }, h("summary", {}, `得分点 ${p.points.length} 个`),
        h("ul", {}, p.points.map(pt => h("li", {}, `${pt.item}：${fmt(pt.got)}/${fmt(pt.max)}`, pt.reason ? h("small", {}, `（${pt.reason}）`) : null)))) : null))))),
    ai.comment ? h("p", { style: "margin-top:10px" }, h("b", {}, "评语："), ai.comment) : null,
    (ai.deductions || []).length ? h("p", {}, h("b", {}, "扣分："), ai.deductions.join("；")) : null) : null;
  return h("div", { class: "qview" }, crops, h("div", { class: "stack" }, scoreBox, partsBox, transcriptView(sheet, num, qr)));
}

function transcriptView(sheet, num, qr) {   // 手写识别结果：识别错了可以改，再按改过的内容重新批
  const e = state.exam;
  const hw = state.results && state.results.scored === false;
  const trView = h("div", { class: "sheet stack" });
  const drawTr = (editing) => {
    if (!editing) {
      fill(trView, h("div", { class: "row spread" }, h("h3", {}, "手写识别结果", qr.legibility ? h("small", {}, `（清晰度：${qr.legibility}）`) : null, qr.transcription_edited ? h("small", {}, "（老师修改过）") : null),
        h("button", { onclick: () => drawTr(true) }, "修改转写")),
        qr.transcription != null ? mathBlock(qr.transcription, "tr") : h("p", { class: "muted" }, "还没有识别。"));
    } else {
      const ta = h("textarea", { rows: 12, value: qr.transcription || "" });
      fill(trView, h("h3", {}, `修改转写（识别错的地方改过来，再让 AI 按修改后的内容重新${hw ? "判" : "给分"}）`), ta,
        h("div", { class: "row" },
          h("button", { class: "primary", onclick: async () => { try { const r = await api("PUT", `/api/exams/${e.id}/sheets/${sheet.id}/q/${encodeURIComponent(num)}/transcription`, { text: ta.value, regrade: true }); if (r.job) watchJob(r.job, () => route()); } catch (er) { toast(er.message, true); } } }, hw ? "保存并重新判" : "保存并重新给分"),
          h("button", { onclick: () => drawTr(false) }, "取消")));
    }
  };
  drawTr(false);
  return trView;
}

function tabKeys(res) {
  const types = res.types || {};
  const keys = [];
  for (const t of ["选择题", "填空题"]) if (res.numbers.some(n => types[n] === t)) keys.push(t);
  return keys.concat(res.numbers.filter(n => !["选择题", "填空题"].includes(types[n])));
}

function objectiveView(sheet, kind, nums, after) {
  const e = state.exam;
  const qmap = Object.fromEntries((e.questions || []).map(q => [q.number, q]));
  const isFill = kind === "填空题";
  const hw = state.results && state.results.scored === false;
  state.acceptCurrent = null;
  const rows = nums.map(n => {
    const qr = sheet.questions[n] || {};
    const ai = qr.ai || {};
    const teacher = qr.teacher && qr.teacher.score != null ? qr.teacher : null;
    const mx = +(qr.max || ai.max || 0);
    const fin = teacher ? teacher.score : ai.score;
    const right = fin != null && mx > 0 && fin >= mx - 1e-9;
    const base = `/api/exams/${e.id}/sheets/${sheet.id}/q/${encodeURIComponent(n)}`;
    const putT = async body => { try { after(await api("PUT", `${base}/teacher`, body)); } catch (er) { toast(er.message, true); } };
    const putS = async ans => {
      try { const r = await api("PUT", `${base}/student`, { answer: ans }); if (r.job) { toast("正在重新判断…"); watchJob(r.job, () => route()); } else after(r.sheet); }
      catch (er) { toast(er.message, true); }
    };
    const crop = (qr.crops || [])[0];
    const img = crop ? h("img", { src: fileUrl(crop), alt: `第${n}题作答`, onclick: () => lightbox(fileUrl(crop)) }) : h("span", { class: "muted" }, "—");
    const verdict = fin == null ? h("span", { class: "muted" }, STATE_TEXT[qr.status] || "未批") :
      h("span", { class: "verdict " + (right ? "ok" : fin > 0 ? "half" : "bad") }, right ? "✓" : fin > 0 ? "½" : "✗", hw ? null : h("small", {}, ` ${fmt(fin)}/${fmt(mx)}`));
    const flag = ai.needs_review && !teacher ? h("div", { class: "flag" }, ai.review_notes || "建议复核") : null;
    const reason = ai.reason ? h("div", { class: "muted small" }, (teacher ? "老师判分；AI：" : "") + ai.reason) : null;
    let student, actions;
    if (isFill) {
      const showStu = () => qr.student == null ? h("span", { class: "muted" }, "—") : qr.student === "" ? h("span", { class: "muted" }, "（未作答）") : mathBlock("$" + qr.student + "$", "pvmath");
      student = h("div", {}, showStu());
      const edit = h("button", { class: "ghost", type: "button", title: "识别错了就改过来，会重新判断", onclick: () => {
        const inp = h("input", { type: "text", value: qr.student || "", style: "width:100%" });
        fill(student, inp, h("div", { class: "row", style: "margin-top:4px" },
          h("button", { class: "primary", type: "button", onclick: () => putS(inp.value) }, "保存并重判"), h("button", { type: "button", onclick: () => fill(student, showStu()) }, "取消")));
        inp.focus(); } }, "改识别");
      actions = h("div", { class: "row tight" },
        h("button", { type: "button", class: teacher && right ? "on" : "", title: "判对（给满分）", onclick: () => putT({ score: mx }) }, "对"),
        h("button", { type: "button", class: teacher && fin === 0 ? "on" : "", title: "判错（0 分）", onclick: () => putT({ score: 0 }) }, "错"),
        edit, teacher ? h("button", { class: "ghost", type: "button", onclick: () => putT({ clear: true }) }, "撤销") : null,
        ai.needs_review && !teacher ? h("button", { type: "button", title: "AI 判得对，确认", onclick: () => putT({ accept: true }) }, "确认") : null);
    } else {
      const nopt = Math.max(4, ((qr.omr || {}).d || []).length);
      const letters = "ABCDEFGH".slice(0, nopt).split("");
      const cur = qr.student || "";
      student = h("b", { class: "choice" }, cur || "（未涂）");
      actions = h("div", { class: "row tight" }, h("span", { class: "muted small" }, "实际涂的是："),
        letters.map(L => h("button", { type: "button", class: cur.includes(L) ? "on" : "", title: `改成 ${L}`, onclick: () => putS(cur.includes(L) && cur.length > 1 ? cur.replace(L, "") : (qmap[n] && (qmap[n].answer || "").length > 1 && !cur.includes(L) ? cur + L : L)) }, L)),
        h("button", { type: "button", class: !cur ? "on" : "", title: "没涂", onclick: () => putS("") }, "空"),
        ai.needs_review && !teacher ? h("button", { type: "button", title: "识别得对，确认", onclick: () => putT({ accept: true }) }, "确认") : null,
        teacher ? h("button", { class: "ghost", type: "button", onclick: () => putT({ clear: true }) }, "撤销改分") : null);
    }
    if (qr.student_edited) student = h("div", {}, student, h("small", { class: "muted" }, "（老师更正）"));
    const ref = qmap[n] ? (isFill ? mathBlock("$" + (qmap[n].answer || "") + "$", "pvmath") : h("b", { class: "choice" }, qmap[n].answer || "")) : "";
    return h("tr", { class: ai.needs_review && !teacher ? "rv" : "" },
      h("td", { class: "num" }, n), h("td", { class: "cropcell" + (isFill ? " fill" : "") }, img), h("td", {}, student),
      h("td", {}, ref), h("td", {}, verdict, flag, reason), h("td", {}, actions));
  });
  const total = nums.reduce((a, n) => { const qr = sheet.questions[n] || {}; const f = qr.teacher && qr.teacher.score != null ? qr.teacher.score : (qr.ai || {}).score; return a + (+f || 0); }, 0);
  const max = nums.reduce((a, n) => a + (+((sheet.questions[n] || {}).max) || 0), 0);
  return h("div", { class: "sheet" },
    h("div", { class: "row spread", style: "margin-bottom:8px" },
      hw ? h("div", { class: "score-big" }, h("small", {}, "做对 "), fmt(total), h("small", {}, ` / ${fmt(max)} 题`)) : h("div", { class: "score-big" }, fmt(total), h("small", {}, ` / ${fmt(max)} 分`)),
      h("span", { class: "muted" }, isFill ? `识别结果和参考答案写法不同时，由模型判断是否等价。点“对/错”直接${hw ? "改判" : "判分"}；识别错了点“改识别”。`
        : `按涂黑程度识别，不调用模型。识别错了点实际涂的选项，${hw ? "对错" : "分数"}会重新算。`)),
    h("div", { class: "tablewrap" }, h("table", { class: "objt review-objt" },
      h("thead", {}, h("tr", {}, h("th", {}, "题号"), h("th", {}, "学生作答"), h("th", {}, "识别结果"), h("th", {}, "参考答案"), h("th", {}, "判定"), h("th", {}, "更正"))),
      h("tbody", {}, rows))));
}

function onKey(ev) {
  if (!location.hash.includes("/review")) return;
  const tag = (ev.target.tagName || "").toLowerCase();
  if (["input", "textarea", "select"].includes(tag) || ev.metaKey || ev.ctrlKey || ev.altKey) return;
  if (ev.key === "/" && state.focusSearch) { ev.preventDefault(); state.focusSearch(); return; }
  const ids = state.rosterIds ? state.rosterIds() : [];
  const nums = state.results ? tabKeys(state.results) : [];
  if (ev.key === "ArrowDown" || ev.key === "ArrowUp") {
    const i = ids.indexOf(state.sid);
    const j = ev.key === "ArrowDown" ? i + 1 : i - 1;
    if (j >= 0 && j < ids.length) { ev.preventDefault(); state.openSheet(ids[j]); }
  } else if (ev.key === "ArrowRight" || ev.key === "ArrowLeft") {
    const i = nums.indexOf(state.num);
    const j = ev.key === "ArrowRight" ? i + 1 : i - 1;
    if (j >= 0 && j < nums.length) { ev.preventDefault(); state.openSheet(state.sid, nums[j]); }
  } else if (ev.key === "Enter" && state.acceptCurrent) {
    ev.preventDefault(); state.acceptCurrent();
  }
}

// ------------------------------------------------------------------ 设置
async function viewSettings(app) {
  const st = await api("GET", "/api/settings");
  state.settings = st;
  const f = (key, label, el, hint) => h("label", { class: "field" }, h("span", {}, label), el, hint ? h("div", { class: "hint" }, hint) : null);
  const inp = (key, type, extra) => h("input", Object.assign({ type: type || "text", name: key, value: st[key] ?? "" }, extra || {}));
  const sel = (key, opts) => h("select", { name: key }, opts.map(([v, t]) => h("option", { value: v, selected: String(st[key]) === String(v) || null }, t)));
  const chk = (key, label) => h("label", { class: "check" }, h("input", { type: "checkbox", name: key, checked: !!st[key] }), label);
  const models = h("datalist", { id: "models" }, ["deepseek-flash", "deepseek-v4-pro"].map(m => h("option", { value: m })));
  const form = h("form", { class: "sheet stack", onsubmit: async ev => {
      ev.preventDefault();
      const fd = new FormData(form); const body = {};
      for (const [k, v] of fd.entries()) body[k] = v;
      ["grading_attach_image", "transcribe_thinking", "identify_student", "fill_strict"].forEach(k => (body[k] = form[k].checked));
      try { state.settings = await api("PUT", "/api/settings", body); toast("设置已保存"); route(); } catch (er) { toast(er.message, true); }
    } },
    h("h1", {}, "设置"), models,
    h("h2", {}, "DeepSeek 接口"),
    h("div", { class: "grid2" },
      f("api_key", "API Key", inp("api_key", "password", { placeholder: "sk-…", autocomplete: "off" }),
        st.api_key_source === "env" ? "正在使用环境变量 DEEPSEEK_API_KEY。在这里填写会保存到本机 data/settings.json。" : "在 platform.deepseek.com 创建。只保存在你这台电脑的 data/settings.json 里。"),
      f("base_url", "接口地址", inp("base_url"))),
    h("h2", {}, "模型"),
    h("div", { class: "grid2" },
      f("vision_model", "看图模型（录题、识别版面、识别手写）", inp("vision_model", "text", { list: "models" }), "必须支持图片输入。DeepSeek 目前是 deepseek-flash。"),
      f("grading_model", "给分模型", inp("grading_model", "text", { list: "models" }), "deepseek-flash 可以同时看原图核对；deepseek-v4-pro 推理更强但不支持图片，只看转写文字。")),
    h("div", { class: "grid2" },
      f("grading_effort", "给分时的思考强度", sel("grading_effort", [["low", "低（快）"], ["high", "高（推荐）"], ["max", "最高（慢）"]])),
      f("answer_model", "生成参考答案的模型", inp("answer_model", "text", { list: "models" }), "用 deepseek-flash 能看到试卷上的配图。")),
    h("div", { class: "row" }, chk("grading_attach_image", "给分时附上原图"), chk("transcribe_thinking", "识别手写时开启思考（更慢，可能更会“猜”）"), chk("identify_student", "读出学生姓名和学号")),
    h("h2", {}, "批改规则"),
    h("div", { class: "grid2" },
      f("multi_partial", "多选题没选全时", sel("multi_partial", [["proportional", "按选对的比例给分（新高考规则）"], ["none", "不给分"], ["2", "固定给 2 分"], ["3", "固定给 3 分"]]), "有选错的一律 0 分。单选题不受影响。"),
      f("judge_effort", "填空题判断等价时的思考强度", sel("judge_effort", [["low", "低（快）"], ["high", "高（推荐）"], ["max", "最高（慢）"]]))),
    h("div", { class: "row" }, chk("fill_strict", "填空题必须化简到最简形式才算对（默认：数值相等、写法不同也算对）")),
    h("div", { class: "grid2" },
      f("samples", "每道题独立批改几次", sel("samples", [["1", "1 次"], ["2", "2 次（双评，推荐正式考试用）"], ["3", "3 次"]]), "多次时取平均分；分差超过下面的阈值就标记“待复核”。"),
      f("review_gap", "双评分差超过多少分要复核", inp("review_gap", "number", { step: "0.5", min: "0" }))),
    h("div", { class: "grid2" },
      f("score_step", "给分最小单位", sel("score_step", [["1", "1 分"], ["0.5", "0.5 分"]])),
      f("max_workers", "同时进行的请求数", inp("max_workers", "number", { min: "1", max: "64" }), "越大越快。DeepSeek 每个账号的并发上限很高（deepseek-flash 是 2500），一次批几十上百份时可以调到 16~32；网络慢的话调低一点。")),
    h("div", { class: "grid2" },
      f("scan_dpi", "PDF 扫描件的读取分辨率（dpi）", inp("scan_dpi", "number", { min: "100", max: "600" })),
      f("crop_max_side", "裁切图最长边（像素）", inp("crop_max_side", "number", { min: "800", max: "4000" }))),
    h("div", {}, h("button", { class: "primary", type: "submit" }, "保存设置")));
  fill(app, form);
}

// 启动时读一次设置（给分单位等）
api("GET", "/api/settings").then(s => (state.settings = s)).catch(() => {});
