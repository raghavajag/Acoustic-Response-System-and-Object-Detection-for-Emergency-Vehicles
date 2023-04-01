function predictAudio(event) {
    event.preventDefault();

    // Get the uploaded audio file
    const file = document.getElementById('audio').files[0];

    // Create a FormData object and add the file to it
    const formData = new FormData();
    formData.append('file', file);

    // Make the POST request to /predict
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
        .then(response => response.text())
        .then(result => {
            console.log("predicted result ", result);
        })
        .catch(error => {
            console.log(error);
        });
}
