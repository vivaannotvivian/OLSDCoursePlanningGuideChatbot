function sendMessage() {
    const inputField = document.getElementById("input");
    const input = inputField.value.trim();
    if (input !== "") {
      getResponseFromPython(input);
      inputField.value = "";
    }
  }
  
  document.addEventListener("DOMContentLoaded", () => {
    const inputField = document.getElementById("input");
    inputField.addEventListener("keydown", (e) => {
      if (e.code === "Enter") {
        const input = inputField.value.trim();
        if (input !== "") {
          getResponseFromPython(input);
          inputField.value = "";
        }
      }
    });
  });
  
  function getResponseFromPython(userInput) {
    addUserMessage(userInput);
  
    fetch("http://127.0.0.1:5000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ message: userInput })
    })
      .then((response) => response.json())
      .then((data) => {
        const reply = data.reply;
        addBotMessage(reply);
      })
      .catch((error) => {
        console.error("Error:", error);
        const fallback = "Sorry, I couldn't connect to the Python server.";
        addBotMessage(fallback);
      });
  }
  
  function addUserMessage(text) {
    const mainDiv = document.getElementById("message-section");
    const userDiv = document.createElement("div");
    userDiv.classList.add("message");
    userDiv.id = "user";
    userDiv.innerHTML = `<span id="user-response">${text}</span>`;
    mainDiv.appendChild(userDiv);
    mainDiv.scrollTop = mainDiv.scrollHeight;
  }
  
  function addBotMessage(text) {
    const mainDiv = document.getElementById("message-section");
    const botDiv = document.createElement("div");
    botDiv.classList.add("message");
    botDiv.id = "bot";
    botDiv.innerHTML = `<span id="bot-response">${text}</span>`;
    mainDiv.appendChild(botDiv);
    mainDiv.scrollTop = mainDiv.scrollHeight;
  }
  