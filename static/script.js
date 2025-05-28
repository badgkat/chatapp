let lightMode = true;
let recorder = null;
let recording = false;
let voiceOption = "default";
const responses = [];
const botRepeatButtonIDToIndexMap = {};
const userRepeatButtonIDToRecordingMap = {};
const baseUrl = window.location.origin;

async function showBotLoadingAnimation() {
  await sleep(500);
  $(".loading-animation")[1].style.display = "inline-block";
}

function hideBotLoadingAnimation() {
  $(".loading-animation")[1].style.display = "none";
}

async function showUserLoadingAnimation() {
  await sleep(100);
  $(".loading-animation")[0].style.display = "flex";
}

function hideUserLoadingAnimation() {
  $(".loading-animation")[0].style.display = "none";
}

const getSpeechToText = async (userRecording) => {
  let response = await fetch(baseUrl + "/speech-to-text", {
    method: "POST",
    body: userRecording.audioBlob,
  });
  response = await response.json();
  return response.text;
};

const processUserMessage = async (userMessage) => {
  let response = await fetch(baseUrl + "/process-message", {
    method: "POST",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify({ userMessage: userMessage, voice: voiceOption }),
  });
  response = await response.json();
  return response;
};

const cleanTextInput = (value) => {
  return value
    .trim()
    .replace(/[\n\t]/g, "")
    .replace(/<[^>]*>/g, "")
    .replace(/[<>&;]/g, "");
};

const recordAudio = () => {
  return new Promise(async (resolve, reject) => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      const audioChunks = [];

      mediaRecorder.addEventListener("dataavailable", (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
        }
      });

      const start = () => mediaRecorder.start();

      const stop = () =>
        new Promise((resolveStop) => {
          mediaRecorder.addEventListener("stop", () => {
            const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
            const audioUrl = URL.createObjectURL(audioBlob);
            const audio = new Audio(audioUrl);
            const play = () => audio.play();
            resolveStop({ audioBlob, audioUrl, play });
          });
          mediaRecorder.stop();
        });

      resolve({ start, stop });
    } catch (err) {
      console.error("Microphone access denied or unavailable:", err);
      alert("Microphone access is required to record audio.");
      reject(err);
    }
  });
};

const sleep = (time) => new Promise((resolve) => setTimeout(resolve, time));

const toggleRecording = async () => {
  if (!recording) {
    recorder = await recordAudio();
    recording = true;
    recorder.start();
  } else {
    const audio = await recorder.stop();
    sleep(1000);
    return audio;
  }
};

const playResponseAudio = (function () {
  const df = document.createDocumentFragment();
  return function Sound(src) {
    const snd = new Audio(src);
    df.appendChild(snd);
    snd.addEventListener("ended", function () {
      df.removeChild(snd);
    });
    snd.play();
    return snd;
  };
})();

const getRandomID = () => {
  return Date.now().toString(36) + Math.random().toString(36).substr(2);
};

const scrollToBottom = () => {
  $("#chat-window").animate({
    scrollTop: $("#chat-window")[0].scrollHeight,
  });
};

const populateUserMessage = (userMessage, userRecording) => {
  $("#message-input").val("");

  if (userRecording) {
    const userRepeatButtonID = getRandomID();
    userRepeatButtonIDToRecordingMap[userRepeatButtonID] = userRecording;
    hideUserLoadingAnimation();
    $("#message-list").append(
      `<div class='message-line my-text'><div class='message-box my-text${
        !lightMode ? " dark" : ""
      }'><div class='me'>${userMessage}</div></div>
            <button id='${userRepeatButtonID}' class='btn volume repeat-button' onclick='userRepeatButtonIDToRecordingMap[this.id].play()'><i class='fa fa-volume-up'></i></button>
            </div>`
    );
  } else {
    $("#message-list").append(
      `<div class='message-line my-text'><div class='message-box my-text${
        !lightMode ? " dark" : ""
      }'><div class='me'>${userMessage}</div></div></div>`
    );
  }

  scrollToBottom();
};

const populateBotResponse = async (userMessage) => {
  await showBotLoadingAnimation();
  const response = await processUserMessage(userMessage);
  responses.push(response);

  const repeatButtonID = getRandomID();
  botRepeatButtonIDToIndexMap[repeatButtonID] = responses.length - 1;
  hideBotLoadingAnimation();
  $("#message-list").append(
    `<div class='message-line'><div class='message-box${
      !lightMode ? " dark" : ""
    }'>${response.ResponseText}</div>
    <button id='${repeatButtonID}' class='btn volume repeat-button' onclick='playResponseAudio("data:audio/wav;base64," + responses[botRepeatButtonIDToIndexMap[this.id]].ResponseSpeech);'><i class='fa fa-volume-up'></i></button></div>`
  );

  playResponseAudio("data:audio/wav;base64," + response.ResponseSpeech);
  scrollToBottom();
};

// === MAIN UI INIT ===
$(document).ready(function () {
  // Load voice options dynamically
  fetch(baseUrl + "/list-voices")
    .then((res) => res.json())
    .then((data) => {
      const select = $("#voice-options");
      select.empty();
      select.append(`<option value="">default</option>`);
      data.voices.forEach((voice) => {
        const label = voice.replace("af_", "").toUpperCase();
        select.append(`<option value="${voice}">${label}</option>`);
      });
    })
    .catch((err) => {
      console.error("Failed to load voices:", err);
    });

  // Message input key handler
  $("#message-input").keyup(function (event) {
    let inputVal = cleanTextInput($("#message-input").val());

    if (event.keyCode === 13 && inputVal != "") {
      const message = inputVal;
      populateUserMessage(message, null);
      populateBotResponse(message);
    }

    inputVal = $("#message-input").val();

    if (!inputVal) {
      $("#send-button")
        .removeClass("send")
        .addClass("microphone")
        .html("<i class='fa fa-microphone'></i>");
    } else {
      $("#send-button")
        .removeClass("microphone")
        .addClass("send")
        .html("<i class='fa fa-paper-plane'></i>");
    }
  });

  // Send button click handler
  $("#send-button").click(async function () {
    const $icon = $(".fa-microphone");

    if ($(this).hasClass("microphone") && !recording) {
      try {
        recorder = await recordAudio();
        recorder.start();
        recording = true;
        $icon.css("color", "#f44336");
      } catch {
        recording = false;
        $icon.css("color", "#125ee5");
      }
    } else if (recording) {
      try {
        const userRecording = await recorder.stop();
        recording = false;
        $icon.css("color", "#125ee5");

        await showUserLoadingAnimation();
        const userMessage = await getSpeechToText(userRecording);
        populateUserMessage(userMessage, userRecording);
        populateBotResponse(userMessage);
      } catch (err) {
        console.error("Failed to stop recording or process audio:", err);
      }
    } else {
      const message = cleanTextInput($("#message-input").val());
      populateUserMessage(message, null);
      populateBotResponse(message);
      $(this)
        .removeClass("send")
        .addClass("microphone")
        .html("<i class='fa fa-microphone'></i>");
    }
  });

  // Toggle dark/light mode
  $("#light-dark-mode-switch").change(function () {
    $("body").toggleClass("dark-mode");
    $(".message-box").toggleClass("dark");
    $(".loading-dots").toggleClass("dark");
    $(".dot").toggleClass("dark-dot");
    lightMode = !lightMode;
  });

  // Update selected voice
  $("#voice-options").change(function () {
    voiceOption = $(this).val();
    console.log("Selected voice:", voiceOption);
  });
});

document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("model-form");
  if (!form) return; // Only run on setup.html

  const progressBar = document.getElementById("download-progress");
  const progressContainer = document.getElementById("progress-container");
  const progressText = document.getElementById("progress-text");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const model = document.getElementById("model").value;
    progressBar.value = 0;
    progressText.textContent = "Starting download...";
    progressContainer.style.display = "block";

    try {
      const response = await fetch("/download-model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model }),
      });

      if (!response.ok || !response.body) {
        progressText.textContent = "Download failed.";
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let received = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        received += decoder.decode(value, { stream: true });

        const lines = received.trim().split("\n");
        const lastLine = lines[lines.length - 1];

        try {
          const { progress, status } = JSON.parse(lastLine);
          progressBar.value = progress;
          progressText.textContent = status;
        } catch {
          // ignore malformed JSON (e.g., incomplete chunk)
        }
      }

      progressText.textContent = "Download complete.";
      form.querySelector("input[type='submit']").value = "Downloaded";
      form.querySelector("input[type='submit']").disabled = true;

      const backLink = document.getElementById("back-link");
      if (backLink) {
        backLink.style.display = "block";
      }

    } catch (err) {
      progressText.textContent = "Error: " + err.message;
    }
  });
});