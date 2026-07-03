const boardEl = document.getElementById("board");
const newGameEl = document.getElementById("newGame");
const statusEl = document.getElementById("status");
const statsEl = document.getElementById("stats");
const modalEl = document.getElementById("modal");
const modalTitleEl = document.getElementById("modalTitle");
const modalTextEl = document.getElementById("modalText");
const modalActionsEl = document.getElementById("modalActions");

let state = null;
let busy = false;
let selectedColor = "black";

newGameEl.addEventListener("click", () => showColorDialog());
renderEmptyBoard(15);
showColorDialog();

async function newGame(humanColor) {
  selectedColor = humanColor;
  hideModal();
  busy = true;
  newGameEl.disabled = true;
  try {
    state = await sendJson("/api/new-game", {
      player_id: "anonymous",
      human_color: selectedColor
    });
    render();
    await refreshStats();
  } catch (error) {
    statusEl.textContent = error.message;
    showMessage("Start failed", error.message);
  } finally {
    busy = false;
    newGameEl.disabled = false;
    if (state) render();
  }
}

async function play(row, col) {
  if (!state || busy || state.result) return;
  if (state.current_player !== state.human_color) return;
  if (state.board[row][col] !== 0) return;
  const previousState = cloneState(state);
  applyOptimisticHumanMove(row, col);
  busy = true;
  newGameEl.disabled = true;
  statusEl.textContent = "AI thinking";
  render();
  try {
    state = await sendJson("/api/move", {
      game_id: previousState.game_id,
      row,
      col
    });
    render();
    await refreshStats();
    showResultIfDone();
  } catch (error) {
    state = previousState;
    statusEl.textContent = error.message;
    showMessage("Move rejected", error.message);
  } finally {
    busy = false;
    newGameEl.disabled = false;
    if (state) render();
  }
}

async function sendJson(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  const data = await response.json();
  if (!response.ok) throw new Error(formatError(data.detail));
  return data;
}

async function refreshStats() {
  const response = await fetch("/api/stats");
  const data = await response.json();
  const total = data.totals;
  statsEl.textContent = `${total.games} games | H ${total.human_win} / AI ${total.ai_win} / D ${total.draw}`;
}

function render() {
  if (!state) return;
  boardEl.style.gridTemplateColumns = `repeat(${state.board_size}, 1fr)`;
    boardEl.innerHTML = "";
  state.board.forEach((rowValues, row) => {
    rowValues.forEach((value, col) => {
      const cell = document.createElement("button");
      cell.className = "cell";
      if (isStarPoint(state.board_size, row, col)) cell.classList.add("star");
      if (value === 1) cell.classList.add("black");
      if (value === 2) cell.classList.add("white");
      if (isLastAiMove(row, col)) cell.classList.add("last-ai");
      cell.disabled = value !== 0 || state.result || state.current_player !== state.human_color || busy;
      cell.setAttribute("aria-label", `${row}, ${col}`);
      cell.addEventListener("click", () => play(row, col));
      boardEl.appendChild(cell);
    });
  });

  if (state.result) {
    statusEl.textContent = resultText(state.result);
  } else if (state.current_player === state.human_color) {
    statusEl.textContent = "Your turn";
  } else {
    statusEl.textContent = "AI thinking";
  }
}

function applyOptimisticHumanMove(row, col) {
  const value = state.human_color === "black" ? 1 : 2;
  state.board[row][col] = value;
  state.move_count += 1;
  state.moves.push({player: state.human_color, row, col});
  state.current_player = state.ai_color;
}

function cloneState(value) {
  return JSON.parse(JSON.stringify(value));
}

function isLastAiMove(row, col) {
  if (!state || !state.moves) return false;
  for (let index = state.moves.length - 1; index >= 0; index -= 1) {
    const move = state.moves[index];
    if (move.player === state.ai_color) return move.row === row && move.col === col;
  }
  return false;
}

function renderEmptyBoard(size) {
  boardEl.style.gridTemplateColumns = `repeat(${size}, 1fr)`;
  boardEl.innerHTML = "";
  for (let row = 0; row < size; row += 1) {
    for (let col = 0; col < size; col += 1) {
      const cell = document.createElement("button");
      cell.className = "cell";
      if (isStarPoint(size, row, col)) cell.classList.add("star");
      cell.disabled = true;
      boardEl.appendChild(cell);
    }
  }
}

function isStarPoint(size, row, col) {
  if (size < 9) return false;
  const edge = size === 9 ? 2 : 3;
  const center = Math.floor(size / 2);
  return [edge, center, size - 1 - edge].includes(row) && [edge, center, size - 1 - edge].includes(col);
}

function resultText(result) {
  if (result === "human_win") return "Human win";
  if (result === "ai_win") return "AI win";
  return "Draw";
}

function showResultIfDone() {
  if (!state || !state.result) return;
  if (state.result === "human_win") {
    showMessage("You Win", "Game saved.");
  } else if (state.result === "ai_win") {
    showMessage("You Lose", "Game saved.");
  } else {
    showMessage("Draw", "Game saved.");
  }
}

function formatError(detail) {
  if (Array.isArray(detail)) return detail.map((item) => item.msg || JSON.stringify(item)).join(" / ");
  if (typeof detail === "string") return detail;
  if (detail) return JSON.stringify(detail);
  return "Request failed";
}

function showColorDialog() {
  modalTitleEl.textContent = "Choose color";
  modalTextEl.textContent = "Play as black first or white second.";
  modalActionsEl.innerHTML = "";
  const black = document.createElement("button");
  black.textContent = "Black first";
  black.addEventListener("click", () => newGame("black"));
  const white = document.createElement("button");
  white.textContent = "White second";
  white.addEventListener("click", () => newGame("white"));
  modalActionsEl.appendChild(black);
  modalActionsEl.appendChild(white);
  modalEl.classList.add("open");
}

function showMessage(title, message) {
  modalTitleEl.textContent = title;
  modalTextEl.textContent = message;
  modalActionsEl.innerHTML = "";
  const close = document.createElement("button");
  close.textContent = "OK";
  close.addEventListener("click", hideModal);
  modalActionsEl.appendChild(close);
  modalEl.classList.add("open");
}

function hideModal() {
  modalEl.classList.remove("open");
}

refreshStats();
