const state = { data: null, sheet: "MomentumLeader", sortKey: null, ascending: false };

const elements = {
  sheet: document.querySelector("#sheet-select"),
  topN: document.querySelector("#top-n"),
  search: document.querySelector("#ticker-search"),
  updated: document.querySelector("#updated-at"),
  title: document.querySelector("#table-title"),
  summary: document.querySelector("#summary"),
  count: document.querySelector("#row-count"),
  head: document.querySelector("#table-head"),
  body: document.querySelector("#table-body"),
  error: document.querySelector("#error"),
};

const label = (name) => name.replaceAll("_", " ");

function displayValue(value, column) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") {
    if (column === "Rank" || column.endsWith("flag")) return String(value);
    return new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(value);
  }
  return String(value);
}

function compareValues(a, b) {
  if (a === null || a === undefined) return 1;
  if (b === null || b === undefined) return -1;
  if (typeof a === "number" && typeof b === "number") return a - b;
  return String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: "base" });
}

function render() {
  const section = state.data.sheets[state.sheet];
  const query = elements.search.value.trim().toUpperCase();
  const limit = Number(elements.topN.value);
  let rows = section.rows.filter((row) => !query || String(row.ticker || "").toUpperCase().includes(query));

  if (state.sortKey) {
    rows = [...rows].sort((a, b) => compareValues(a[state.sortKey], b[state.sortKey]) * (state.ascending ? 1 : -1));
  }
  rows = rows.slice(0, limit);

  const columns = section.rows.length ? Object.keys(section.rows[0]) : [];
  elements.title.textContent = `${state.sheet} — Top ${limit}`;
  elements.summary.textContent = section.summary;
  elements.count.textContent = `${rows.length} of ${section.rows.length} names`;
  elements.head.replaceChildren();
  elements.body.replaceChildren();

  const headerRow = document.createElement("tr");
  ["Rank", ...columns].forEach((column) => {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = `${label(column)}${state.sortKey === column ? (state.ascending ? " ▲" : " ▼") : ""}`;
    if (column !== "Rank") {
      th.addEventListener("click", () => {
        state.ascending = state.sortKey === column ? !state.ascending : false;
        state.sortKey = column;
        render();
      });
    }
    headerRow.appendChild(th);
  });
  elements.head.appendChild(headerRow);

  rows.forEach((row, index) => {
    const tr = document.createElement("tr");
    [["Rank", index + 1], ...columns.map((column) => [column, row[column]])].forEach(([column, value]) => {
      const td = document.createElement("td");
      td.textContent = displayValue(value, column);
      tr.appendChild(td);
    });
    elements.body.appendChild(tr);
  });
}

async function initialize() {
  try {
    const response = await fetch("data/rankings.json", { cache: "no-cache" });
    if (!response.ok) throw new Error(`Data request failed (${response.status})`);
    state.data = await response.json();
    Object.keys(state.data.sheets).forEach((name) => elements.sheet.add(new Option(name, name)));
    elements.updated.textContent = new Intl.DateTimeFormat("en-US", {
      year: "numeric", month: "short", day: "numeric", hour: "numeric",
      minute: "2-digit", timeZoneName: "short"
    }).format(new Date(state.data.updated_at));
    render();
  } catch (error) {
    elements.error.hidden = false;
    elements.error.textContent = `The rankings could not be loaded. ${error.message}`;
    elements.title.textContent = "Data unavailable";
  }
}

elements.sheet.addEventListener("change", (event) => {
  state.sheet = event.target.value;
  state.sortKey = null;
  state.ascending = false;
  render();
});
elements.topN.addEventListener("change", render);
elements.search.addEventListener("input", render);
initialize();
