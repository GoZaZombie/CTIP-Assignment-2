<script setup>
import { ref } from 'vue'

const modelChoice = ref('LR')
const message = ref('')
const predictionLabel = ref('')
const confidence = ref(null)
const error = ref('')
const resultColor = ref('orange')

const modelOptions = ['LR', 'NBE', 'NBSMS', 'SVM', 'GRU']

async function submitForm() {
  predictionLabel.value = ''
  confidence.value = null
  error.value = ''
  if (message.value === '6 7') {
        window.location.href = 'https://youtu.be/XnygT6ANLzQ?list=RDXnygT6ANLzQ&t=30';
        return;
      }
    
  try {
    const response = await fetch('http://127.0.0.1:8000/CLASSIFY/Detection', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ModelChoice: modelChoice.value,
        Message: message.value,
      }),
    })

    if (!response.ok) throw new Error('Request failed')
    
    const data = await response.json()
    //const pred = data.Prediction //commented this out too since idk i forgot to just change the existing code
    const label = data.Prediction;
    const conf = data.Confidence;
 
    console.log('API data:', data);
    predictionLabel.value = label || '';
    confidence.value = conf !== undefined && conf !== null ? conf.toFixed(2) : '';
    if (label === 'Spam'){ // now this handles just the spam string rather than the array 
      resultColor.value = 'red';
    }
    else if (label === 'Safe') {
      resultColor.value = 'green';
      
    }
    else {
        resultColor.value = 'white' 
    }
      
  } catch (err) {
    error.value = err.message
  }
}
</script>

<template>
  <h1>Spam Checker</h1>
  <Transition name="fade">
  <div v-if="predictionLabel" class="result" :style="{ color: resultColor }">
      <p>
        Prediction using {{ modelChoice }} model: {{ predictionLabel }} <br></br>
        <span v-if="confidence"> (Confidence: {{ confidence }})</span>

      </p>
      
  </div>
  </Transition>

  <div v-if="error">Error: {{ error }}</div>
 

    <form @submit.prevent="submitForm" class="input-row">

      <select v-model="modelChoice" >
        <option v-for="option in modelOptions" :key="option" :value="option">
          {{ option }}
        </option>
      </select>

 
      <textarea v-model="message" placeholder="Enter message" ></textarea>


      <button type="submit" >Check!</button>

    </form>

    

    
  

</template>

