document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();  // Impede o envio padrão do formulário

    // Coletando dados do formulário
    const features = [
        parseFloat(document.getElementById('age').value),
        document.getElementById('workclass').value,
        parseFloat(document.getElementById('fnlwgt').value),
        document.getElementById('education').value,
        parseFloat(document.getElementById('education-num').value),
        document.getElementById('marital-status').value,
        document.getElementById('occupation').value,
        document.getElementById('relationship').value,
        document.getElementById('race').value,
        document.getElementById('sex').value,
        parseFloat(document.getElementById('capital-gain').value),
        parseFloat(document.getElementById('capital-loss').value),
        parseFloat(document.getElementById('hours-per-week').value),
        document.getElementById('native-country').value
    ];

    // Enviando os dados para a API
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ features })
    });

    const result = await response.json();
    document.getElementById('result').innerText = `Previsão: ${result.prediction === 1 ? 'Renda >50K' : 'Renda <=50K'}`;
});
