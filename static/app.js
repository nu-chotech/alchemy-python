let gameId = null;
let currentState = null;
let activeCandidateSet = null;
let selectedCandidate = null;
let camera = { rotX: -18, rotY: 28, zoom: 1 };
let dragStart = null;

const elements = {
  currentWord: document.querySelector("#current-word"),
  targetWord: document.querySelector("#target-word"),
  difficultyLabel: document.querySelector("#difficulty-label"),
  combo: document.querySelector("#combo"),
  turn: document.querySelector("#turn"),
  resultOutput: document.querySelector("#result-output"),
  historyList: document.querySelector("#history-list"),
  eventList: document.querySelector("#event-list"),
  candidateList: document.querySelector("#candidate-list"),
  candidateMeta: document.querySelector("#candidate-meta"),
  vocabulary: document.querySelector("#vocabulary"),
  alphaInput: document.querySelector("#alpha-input"),
  alphaOutput: document.querySelector("#alpha-output"),
  scene: document.querySelector("#map-scene"),
  viewport: document.querySelector("#map-viewport"),
  selectedLabel: document.querySelector("#selected-label"),
};

elements.alphaInput.addEventListener("input", () => {
  elements.alphaOutput.textContent = Number(elements.alphaInput.value).toFixed(2);
});

document.querySelector("#new-game-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  await startGame();
});

document.querySelector("#candidate-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!gameId) {
    showError("先にゲームを開始してください。");
    return;
  }
  const payload = {
    material_a: document.querySelector("#material-a-input").value.trim(),
    material_b: document.querySelector("#material-b-input").value.trim(),
    alpha: Number(elements.alphaInput.value),
  };
  await callApi(`/api/craft/games/${gameId}/candidates`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
});

document.querySelector("#reset-camera-button").addEventListener("click", () => {
  camera = { rotX: -18, rotY: 28, zoom: 1 };
  renderMap();
});

elements.viewport.addEventListener("pointerdown", (event) => {
  dragStart = { x: event.clientX, y: event.clientY, rotX: camera.rotX, rotY: camera.rotY };
  elements.viewport.setPointerCapture(event.pointerId);
});

elements.viewport.addEventListener("pointermove", (event) => {
  if (!dragStart) return;
  camera.rotY = dragStart.rotY + (event.clientX - dragStart.x) * 0.35;
  camera.rotX = dragStart.rotX - (event.clientY - dragStart.y) * 0.35;
  renderMap();
});

elements.viewport.addEventListener("pointerup", () => {
  dragStart = null;
});

elements.viewport.addEventListener("wheel", (event) => {
  event.preventDefault();
  camera.zoom = Math.max(0.55, Math.min(2.3, camera.zoom + (event.deltaY > 0 ? -0.08 : 0.08)));
  renderMap();
});

async function startGame() {
  activeCandidateSet = null;
  selectedCandidate = null;
  const payload = {
    target: document.querySelector("#target-input").value.trim(),
    difficulty: document.querySelector("#difficulty-input").value,
    seed: Number(document.querySelector("#seed-input").value),
    use_mock: document.querySelector("#mock-input").checked,
    combo_enabled: document.querySelector("#combo-enabled-input").checked,
    goal_bias_enabled: document.querySelector("#goal-bias-input").checked,
  };
  await callApi("/api/craft/games", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await loadVocabulary(payload.use_mock);
}

async function callApi(url, options) {
  try {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok) {
      showApiError(data.detail || "APIエラーが発生しました。");
      return;
    }
    render(data);
  } catch (error) {
    showError(error.message);
  }
}

function render(data) {
  if (data.state) {
    currentState = data.state;
    gameId = data.state.game_id;
    renderState(data.state);
  }
  if (data.candidate_set) {
    activeCandidateSet = data.candidate_set;
    selectedCandidate = null;
    renderCandidates(data.candidate_set);
  }
  if (data.result) {
    activeCandidateSet = null;
    selectedCandidate = null;
    document.querySelector("#material-a-input").value = data.result.word;
    renderCandidates(null);
  }
  elements.resultOutput.classList.remove("error");
  elements.resultOutput.textContent = JSON.stringify(data, null, 2);
  renderMap();
}

function renderState(state) {
  elements.currentWord.textContent = state.current;
  elements.targetWord.textContent = state.target;
  elements.difficultyLabel.textContent = state.difficulty;
  elements.combo.textContent = state.combo;
  elements.turn.textContent = state.turn;
  elements.historyList.replaceChildren(
    ...state.history.map((word) => {
      const item = document.createElement("li");
      item.textContent = word;
      return item;
    })
  );
  elements.eventList.replaceChildren(
    ...state.events.slice().reverse().map((entry) => {
      const item = document.createElement("li");
      item.textContent = `Turn ${entry.turn}: ${entry.word} / ${entry.target_similarity.toFixed(3)} / Combo ${entry.combo}`;
      return item;
    })
  );
}

function renderCandidates(candidateSet) {
  if (!candidateSet) {
    elements.candidateMeta.textContent = "候補を見ると最大3語を表示します。";
    elements.candidateList.replaceChildren();
    return;
  }
  elements.candidateMeta.textContent = `候補セット ${candidateSet.candidate_set_id.slice(0, 8)} / beta ${candidateSet.beta.toFixed(3)} / pool ${candidateSet.pool_size}`;
  elements.candidateList.replaceChildren(
    ...candidateSet.candidates.map((candidate) => {
      const card = document.createElement("article");
      card.className = "candidate-card";
      card.tabIndex = 0;
      card.dataset.candidateId = candidate.id;
      if (selectedCandidate?.id === candidate.id) card.classList.add("selected");

      const title = document.createElement("h3");
      title.textContent = candidate.word;
      const meta = document.createElement("p");
      const targetText =
        candidate.target_similarity === null ? "" : ` / 目標 ${candidate.target_similarity.toFixed(3)}`;
      meta.textContent = `合成 ${candidate.blend_similarity.toFixed(3)}${targetText}`;

      const actions = document.createElement("div");
      actions.className = "candidate-actions";
      const confirmButton = document.createElement("button");
      confirmButton.type = "button";
      confirmButton.textContent = "確定";
      confirmButton.addEventListener("click", async (event) => {
        event.stopPropagation();
        await confirmCandidate(candidate);
      });
      const reuseAButton = document.createElement("button");
      reuseAButton.type = "button";
      reuseAButton.textContent = "Aへ";
      reuseAButton.addEventListener("click", (event) => {
        event.stopPropagation();
        document.querySelector("#material-a-input").value = candidate.word;
      });
      const reuseBButton = document.createElement("button");
      reuseBButton.type = "button";
      reuseBButton.textContent = "Bへ";
      reuseBButton.addEventListener("click", (event) => {
        event.stopPropagation();
        document.querySelector("#material-b-input").value = candidate.word;
      });
      actions.append(confirmButton, reuseAButton, reuseBButton);

      card.addEventListener("click", () => {
        selectedCandidate = candidate;
        renderCandidates(candidateSet);
        renderMap();
      });
      card.append(title, meta, actions);
      return card;
    })
  );
}

async function confirmCandidate(candidate) {
  const payload = {
    candidate_set_id: activeCandidateSet.candidate_set_id,
    candidate_id: candidate.id,
  };
  await callApi(`/api/craft/games/${gameId}/confirm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

function renderMap() {
  if (!currentState) return;
  const nodes = [...currentState.nodes];
  if (activeCandidateSet) {
    activeCandidateSet.candidates.forEach((candidate) => {
      nodes.push({
        word: candidate.word,
        role: selectedCandidate?.id === candidate.id ? "selected" : "candidate",
        point: candidate.point,
      });
    });
  }
  elements.scene.replaceChildren(
    ...nodes.map((node) => {
      const point = document.createElement("button");
      point.type = "button";
      point.className = `map-point ${node.role}`;
      point.title = node.word;
      const projected = projectPoint(node.point);
      point.style.left = `${projected.x}%`;
      point.style.top = `${projected.y}%`;
      point.style.transform = `translate(-50%, -50%) scale(${projected.scale})`;
      point.style.zIndex = String(projected.zIndex);
      point.addEventListener("click", () => {
        elements.selectedLabel.textContent = `${node.word} (${node.role})`;
      });
      if (node.role === "target" || node.role === "current" || node.role === "selected") {
        const label = document.createElement("span");
        label.textContent = node.word;
        point.append(label);
      }
      return point;
    })
  );
  const selectedWord = selectedCandidate?.word || currentState.current;
  elements.selectedLabel.textContent = selectedWord;
}

function projectPoint(point) {
  const scale = 190 * camera.zoom;
  const rx = (camera.rotX * Math.PI) / 180;
  const ry = (camera.rotY * Math.PI) / 180;
  const y1 = point.y * Math.cos(rx) - point.z * Math.sin(rx);
  const z1 = point.y * Math.sin(rx) + point.z * Math.cos(rx);
  const x2 = point.x * Math.cos(ry) + z1 * Math.sin(ry);
  const z2 = -point.x * Math.sin(ry) + z1 * Math.cos(ry);
  const perspective = 1 / (1 + z2 * 0.25);
  return {
    x: 50 + x2 * scale * perspective,
    y: 50 - y1 * scale * perspective,
    scale: Math.max(0.72, Math.min(1.7, perspective)),
    zIndex: Math.round(1000 + z2 * 100),
  };
}

function showApiError(detail) {
  if (typeof detail === "object") {
    const suggestions = detail.suggestions?.length ? `\n候補: ${detail.suggestions.join(", ")}` : "";
    showError(`${detail.message || detail.error}${suggestions}`);
    return;
  }
  showError(detail);
}

function showError(message) {
  elements.resultOutput.classList.add("error");
  elements.resultOutput.textContent = message;
}

async function loadVocabulary(useMock = true) {
  const response = await fetch(`/api/vocabulary?use_mock=${useMock ? "true" : "false"}`);
  if (!response.ok) return;
  const data = await response.json();
  elements.vocabulary.replaceChildren(
    ...data.words.map((word) => {
      const option = document.createElement("option");
      option.value = word;
      return option;
    })
  );
}

loadVocabulary(true);
startGame();
