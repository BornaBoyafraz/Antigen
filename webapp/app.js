// Talks to the little FastAPI backend. Plain fetch, nothing clever.

const input = document.getElementById("input");
const result = document.getElementById("result");
const verdict = document.getElementById("verdict");
const scoreFill = document.getElementById("score-fill");
const scoreText = document.getElementById("score-text");
const signalsBox = document.getElementById("signals");

document.getElementById("check").addEventListener("click", checkText);
document.getElementById("clear").addEventListener("click", () => {
  input.value = "";
  result.classList.add("hidden");
});

async function checkText() {
  const text = input.value.trim();
  if (!text) {
    return;
  }
  verdict.textContent = "Checking...";
  verdict.className = "";
  result.classList.remove("hidden");
  signalsBox.innerHTML = "";
  scoreFill.style.width = "0";
  scoreText.textContent = "";

  try {
    const res = await fetch("/api/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text }),
    });
    const data = await res.json();
    showResult(data);
  } catch (err) {
    verdict.textContent = "Something went wrong (is the server running?)";
  }
}

function showResult(data) {
  const pct = Math.round(data.score * 100);
  verdict.textContent = data.label === "injection" ? "Looks like an injection" : "Looks benign";
  verdict.className = data.label;
  scoreFill.style.width = pct + "%";
  scoreText.textContent = "Injection score: " + pct + "%";

  // Show whichever clues actually fired.
  const exp = data.explanation || {};
  const groups = [
    ["Override phrases", exp.triggered_phrases],
    ["Fake role markers", exp.triggered_role_markers],
    ["Talking to the assistant", exp.triggered_addressed_phrases],
    ["Third-party content markers", exp.triggered_indirect_frames],
    ["Hidden in a code comment", exp.triggered_code_comment_directives],
    ["Hidden in tool arguments", exp.triggered_json_tool_argument_directives],
    ["Hidden in a URL", exp.triggered_url_query_directives],
    ["Reads as quoted / discussed", exp.discussion_context_phrases],
  ];

  signalsBox.innerHTML = "";
  let anyShown = false;
  for (const [label, items] of groups) {
    if (items && items.length) {
      anyShown = true;
      const div = document.createElement("div");
      div.className = "signal";
      const chips = items.map((t) => `<span class="chip">${escapeHtml(t)}</span>`).join("");
      div.innerHTML = `<b>${label}:</b> ${chips}`;
      signalsBox.appendChild(div);
    }
  }
  if (!anyShown) {
    signalsBox.innerHTML = "<p class='signal'>No obvious injection signals fired.</p>";
  }
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

// Load the example gallery so people can try things quickly.
async function loadExamples() {
  const box = document.getElementById("examples");
  try {
    const res = await fetch("/api/examples");
    const examples = await res.json();
    box.innerHTML = "";
    for (const ex of examples) {
      const btn = document.createElement("button");
      btn.className = "example";
      btn.innerHTML = `<div class="cat">${escapeHtml(ex.category)}</div>${escapeHtml(ex.title)}`;
      btn.addEventListener("click", () => {
        input.value = ex.text;
        checkText();
        window.scrollTo({ top: 0, behavior: "smooth" });
      });
      box.appendChild(btn);
    }
  } catch (err) {
    box.textContent = "Couldn't load examples.";
  }
}

loadExamples();
