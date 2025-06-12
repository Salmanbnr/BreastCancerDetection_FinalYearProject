let currentImageBase64 = null;

function showLoading(show) {
  const loading = document.getElementById("loading");
  const predictButton = document.getElementById("predictButton");
  const explainButton = document.getElementById("explainButton");
  
  if (show) {
    loading.classList.remove("hidden");
    predictButton.disabled = true;
    predictButton.classList.add("opacity-50", "cursor-not-allowed");
    if (explainButton) {
      explainButton.disabled = true;
      explainButton.classList.add("opacity-50", "cursor-not-allowed");
    }
  } else {
    loading.classList.add("hidden");
    predictButton.disabled = false;
    predictButton.classList.remove("opacity-50", "cursor-not-allowed");
    if (explainButton) {
      explainButton.disabled = false;
      explainButton.classList.remove("opacity-50", "cursor-not-allowed");
    }
  }
}

function predictImage() {
  const input = document.getElementById("imageInput");
  if (input.files.length === 0) {
    alert("Please select an image first.");
    return;
  }

  showLoading(true);

  const file = input.files[0];
  const formData = new FormData();
  formData.append("file", file);

  fetch("/predict", {
    method: "POST",
    body: formData,
  })
    .then((res) => res.json())
    .then((data) => {
      showLoading(false);
      if (data.error) {
        alert(data.error);
        return;
      }

      document.getElementById("label").textContent = data.predicted_label;
      document.getElementById("confidence").textContent = (data.confidence * 100).toFixed(2) + "%";
      document.getElementById("originalImage").src = "data:image/jpeg;base64," + data.image_base64;
      document.getElementById("result").classList.remove("hidden");
      document.getElementById("explanations").classList.add("hidden");
      currentImageBase64 = data.image_base64;
    })
    .catch((err) => {
      showLoading(false);
      console.error(err);
      alert("Prediction failed.");
    });
}

function explainImage() {
  if (!currentImageBase64) {
    alert("No image to explain. Please predict an image first.");
    return;
  }

  showLoading(true);

  fetch("/explain", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ image_base64: currentImageBase64 }),
  })
    .then((res) => res.json())
    .then((data) => {
      showLoading(false);
      if (data.error) {
        alert(data.error);
        return;
      }

      document.getElementById("gradcamImage").src = "data:image/jpeg;base64," + data.gradcam;
      document.getElementById("limeImage").src = "data:image/jpeg;base64," + data.lime;
      document.getElementById("urlabImage").src = "data:image/jpeg;base64," + data.urlab;
      document.getElementById("explanations").classList.remove("hidden");
    })
    .catch((err) => {
      showLoading(false);
      console.error(err);
      alert("Explanation failed.");
    });
}