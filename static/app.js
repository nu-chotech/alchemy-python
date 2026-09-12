let gameId = null;

const elements = {
  currentWord: document.querySelector("#current-word"),
  targetWord: document.querySelector("#target-word"),
  similarity: document.querySelector("#similarity"),
  temperature: document.querySelector("#temperature"),
  combo: document.querySelector("#combo"),
  scoreTotal: document.querySelector("#score-total"),
  resultOutput: document.querySelector("#result-output"),
  historyList: document.querySelector("#history-list"),
  candidateList: document.querySelector("#candidate-list"),
  eventList: document.querySelector("#event-list"),
  gameMeta: document.querySelector("#game-meta"),
  vocabulary: document.querySelector("#vocabulary"),
  strengthInput: document.querySelector("#strength-input"),
  strengthOutput: document.querySelector("#strength-output"),
};

elements.strengthInput.addEventListener("input", () => {
  elements.strengthOutput.textContent = Number(elements.strengthInput.value).toFixed(2);
});

document.querySelector("#new-game-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = {
    target: document.querySelector("#target-input").value.trim(),
    seed: Number(document.querySelector("#seed-input").value),
    use_mock: document.querySelector("#mock-input").checked,
  };
  await callApi("/api/games", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
});

document.querySelector("#step-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!gameId) {
    showError("先にゲームを開始してください。");
    return;
  }
  const payload = {
    operation: document.querySelector("#operation-input").value,
    ingredient: document.querySelector("#ingredient-input").value.trim(),
    strength: Number(elements.strengthInput.value),
  };
  await callApi(`/api/games/${gameId}/steps`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
});

async function callApi(url, options) {
  try {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok) {
      showError(data.detail || "APIエラーが発生しました。");
      return;
    }
    render(data);
  } catch (error) {
    showError(error.message);
  }
}

function render(data) {
  const { state, result } = data;
  gameId = state.game_id;
  elements.currentWord.textContent = state.current;
  elements.targetWord.textContent = state.target;
  elements.similarity.textContent = state.similarity.toFixed(3);
  elements.temperature.textContent = state.temperature;
  elements.combo.textContent = state.combo;
  elements.scoreTotal.textContent = Math.round(state.score_total).toLocaleString("ja-JP");
  elements.gameMeta.textContent = `${state.vector_source} / Turn ${state.turn} of ${state.turn_limit} / Best ${state.best_similarity.toFixed(3)}`;
  elements.resultOutput.classList.remove("error");
  elements.resultOutput.textContent = JSON.stringify(result || state, null, 2);

  elements.historyList.replaceChildren(
    ...state.history.map((word) => {
      const item = document.createElement("li");
      item.textContent = word;
      return item;
    })
  );

  const candidates = result?.candidates || [];
  elements.candidateList.replaceChildren(
    ...candidates.map((candidate) => {
      const item = document.createElement("li");
      item.textContent = `${candidate.word} (${candidate.score.toFixed(3)})`;
      return item;
    })
  );

  elements.eventList.replaceChildren(
    ...state.recent_events.slice().reverse().map((eventText) => {
      const item = document.createElement("li");
      item.textContent = eventText;
      return item;
    })
  );
}

function showError(message) {
  elements.resultOutput.classList.add("error");
  elements.resultOutput.textContent = message;
}

async function loadVocabulary() {
  const response = await fetch("/api/vocabulary?use_mock=true");
  const data = await response.json();
  elements.vocabulary.replaceChildren(
    ...data.words.map((word) => {
      const option = document.createElement("option");
      option.value = word;
      return option;
    })
  );
}

loadVocabulary();
