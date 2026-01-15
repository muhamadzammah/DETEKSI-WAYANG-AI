function upload() {
    const file = document.getElementById("imageInput").files[0];
    const formData = new FormData();
    formData.append("file", file);

    document.getElementById("progress").innerHTML =
        "Step 1: Upload image...<br>" +
        "Step 2: Preprocessing...<br>" +
        "Step 3: CNN Feature Extraction...<br>" +
        "Step 4: Classification...<br>" +
        "Step 5: Result generated...";

    const reader = new FileReader();
    reader.onload = () => {
        document.getElementById("preview").src = reader.result;
    };
    reader.readAsDataURL(file);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("result").textContent =
            JSON.stringify(data, null, 2);
    });
}
