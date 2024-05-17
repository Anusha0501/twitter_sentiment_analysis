function analyzeSentiment() {
  var textInput = document.getElementById('textInput').value;
  fetch('/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ text: textInput })
  })
  .then(response => response.text())
  .then(data => {
    var resultDiv = document.getElementById('result');
    resultDiv.innerText = 'Sentiment: ' + data;
    var characterImg = document.getElementById('characterImg');
    var characterDiv = document.getElementById('character');
    if (data === 'Positive') {
      characterImg.src = 'positive.png';
      characterDiv.style.display = 'block';
    } else if (data === 'Negative') {
      characterImg.src = 'negative.png';
      characterDiv.style.display = 'block';
    } else {
      characterImg.src = 'neutral.png';
      characterDiv.style.display = 'block';
    }
  })
  .catch(error => console.error('Error:', error));
}
