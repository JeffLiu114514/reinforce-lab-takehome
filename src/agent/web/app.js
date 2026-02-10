const form = document.getElementById("run-form");
const statusText = document.getElementById("status-text");
const runBtn = document.getElementById("run-btn");
const sampleBtn = document.getElementById("sample-btn");
const clearBtn = document.getElementById("clear-btn");
const summary = document.getElementById("summary");
const filePaths = document.getElementById("file-paths");

const reportContent = document.getElementById("report-content");
const graphContent = document.getElementById("graph-content");
const ledgerContent = document.getElementById("ledger-content");
const traceContent = document.getElementById("trace-content");
const conclusion = document.getElementById("conclusion");
const graphRender = document.getElementById("graph-render");

const SAMPLE_PROMPT = "What are the real-world risks and benefits of using synthetic data to train or fine-tune large language models? Focus on data quality, bias, and evaluation.";
let pollTimer = null;

function setStatus(text, mode = "") {
  statusText.textContent = text;
  statusText.className = `status ${mode}`.trim();
}

function readPayload() {
  const data = new FormData(form);
  return {
    prompt: (data.get("prompt") || "").toString(),
    k_per_query: Number(data.get("k_per_query") || 8),
    max_urls: Number(data.get("max_urls") || 30),
    out_dir: (data.get("out_dir") || "artifacts").toString(),
    config_path: (data.get("config_path") || "config/source_weights.json").toString(),
    google_model: (data.get("google_model") || "").toString() || null,
    google_api_key: (data.get("google_api_key") || "").toString() || null,
  };
}

function renderSummary(payload) {
  const items = [
    ["sources", payload.summary?.sources ?? 0],
    ["evidence", payload.summary?.evidence ?? 0],
    ["claims", payload.summary?.claims ?? 0],
    ["edges", payload.summary?.edges ?? 0],
    ["resolutions", payload.summary?.resolutions ?? 0],
    ["ece", Number(payload.metrics?.ece ?? 0).toFixed(3)],
  ];

  summary.innerHTML = items
    .map(([name, value]) => `<div class="metric"><div class="name">${name}</div><div class="value">${value}</div></div>`)
    .join("");

  const fileMap = payload.files || {};
  filePaths.innerHTML = [
    `<div><strong>report:</strong> ${fileMap.report || "-"}</div>`,
    `<div><strong>graph:</strong> ${fileMap.graph || "-"}</div>`,
    `<div><strong>ledger:</strong> ${fileMap.ledger || "-"}</div>`,
    `<div><strong>trace:</strong> ${fileMap.trace || "-"}</div>`,
  ].join("");
}

function renderConclusion(ledger) {
  const claims = ledger?.graph?.claims || [];
  if (!claims.length) {
    conclusion.innerHTML = "<div class=\"conclusion-item\"><h3>No claims yet.</h3><p>Run the agent to generate claims.</p></div>";
    return;
  }
  const items = claims
    .slice()
    .sort((a, b) => (b.confidence || 0) - (a.confidence || 0))
    .map((claim) => {
      const status = claim.needs_more_evidence ? "needs_more_evidence" : "";
      return `
        <div class="conclusion-item">
          <h3>${claim.id} - Confidence ${claim.confidence}/5</h3>
          <p>${claim.statement}</p>
          ${status ? `<p><strong>Status:</strong> ${status}</p>` : ""}
        </div>
      `;
    })
    .join("");
  conclusion.innerHTML = items;
}

function renderGraph(graphText) {
  graphRender.innerHTML = "";
  if (!graphText) {
    graphRender.innerHTML = "<p>No graph available.</p>";
    return;
  }

  const wrapper = document.createElement("div");
  wrapper.className = "mermaid";
  wrapper.textContent = graphText;
  graphRender.appendChild(wrapper);

  if (window.mermaid) {
    window.mermaid.initialize({ startOnLoad: false });
    window.mermaid.run({ nodes: [wrapper] }).catch(() => {
      graphRender.innerHTML = `<pre class="viewer">${graphText}</pre>`;
    });
  } else {
    graphRender.innerHTML = `<pre class="viewer">${graphText}</pre>`;
  }
}

function clearOutputs() {
  summary.innerHTML = "";
  filePaths.innerHTML = "";
  reportContent.textContent = "";
  graphContent.textContent = "";
  ledgerContent.textContent = "";
  traceContent.textContent = "";
  conclusion.innerHTML = "";
  graphRender.innerHTML = "";
  setStatus("Idle.");
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function pollStatus(runId) {
  const response = await fetch(`/api/status/${runId}`);
  const status = await response.json();
  if (!response.ok) {
    throw new Error(status.detail || "Run status failed");
  }

  if (status.progress?.length) {
    setStatus(status.progress.join("\n"));
  }

  if (status.status === "error") {
    stopPolling();
    setStatus(`Run failed: ${status.error || "Unknown error"}`, "error");
    runBtn.disabled = false;
    return;
  }

  if (status.status === "complete" && status.result) {
    stopPolling();
    runBtn.disabled = false;
    const result = status.result;
    setStatus(`Run complete. Output directory: ${result.out_dir}`, "ok");
    renderSummary(result);
    renderConclusion(result.ledger);
    renderGraph(result.graph_mermaid);

    reportContent.textContent = result.report_markdown || "";
    graphContent.textContent = result.graph_mermaid || "";
    ledgerContent.textContent = JSON.stringify(result.ledger || {}, null, 2);
    traceContent.textContent = JSON.stringify(result.trace || {}, null, 2);
  }
}

async function submitRun(event) {
  event.preventDefault();
  const payload = readPayload();

  if (!payload.prompt.trim()) {
    setStatus("Prompt is required.", "error");
    return;
  }

  runBtn.disabled = true;
  setStatus("Run queued...");

  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.detail || "Run failed");
    }

    stopPolling();
    pollTimer = setInterval(() => {
      pollStatus(result.run_id).catch((error) => {
        stopPolling();
        setStatus(`Run failed: ${error.message}`, "error");
        runBtn.disabled = false;
      });
    }, 1200);
  } catch (error) {
    runBtn.disabled = false;
    setStatus(`Run failed: ${error.message}`, "error");
  }
}

function setupTabs() {
  const tabs = [...document.querySelectorAll(".tab")];
  const panels = [...document.querySelectorAll(".tab-content")];

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((item) => item.classList.remove("active"));
      panels.forEach((panel) => panel.classList.remove("active"));

      tab.classList.add("active");
      const target = tab.dataset.tab;
      const panel = document.getElementById(target);
      if (panel) {
        panel.classList.add("active");
      }
    });
  });
}

form.addEventListener("submit", submitRun);
sampleBtn.addEventListener("click", () => {
  document.getElementById("prompt").value = SAMPLE_PROMPT;
  setStatus("Sample prompt loaded.");
});
clearBtn.addEventListener("click", () => {
  stopPolling();
  clearOutputs();
});

setupTabs();
setStatus("Idle.");
