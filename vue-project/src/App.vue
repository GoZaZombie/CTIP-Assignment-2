<script setup lang="ts">
import { ref } from 'vue'
import { reactive } from "vue";
const modelChoice = ref('LR')
const message = ref('')
const predictionLabel = ref('')
const confidence = ref<number | null>(null)
const error = ref('')
const resultColor = ref('orange')

const modelOptions = ['LR', 'NBE', 'NBSMS', 'SVM', 'GRU']
interface Result {
  id: number;
  type: string;
  message: string;
  model: string;
  prediction: string;
  confidence: number | null;
  onClick: () => void;   // function each element can run
}
const results = reactive<Result[]>([]);
function addResult(type: string, message: string, model: string, prediction: string, confidence: number | null)  {
  results.push({
    id: Date.now(),
    type,
    message,
    model,
    prediction,
    confidence,
    onClick: () => {
      alert(`You clicked: ${type}`);
    }
  });
}
async function submitForm() {
  predictionLabel.value = ''
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
    const pred = data.Prediction

    // Handle both cases: with or without confidence
    if (Array.isArray(pred)) {
      predictionLabel.value = pred[0] || ''
      confidence.value = pred[1] !== undefined ? pred[1] : null
    } else {
      predictionLabel.value = pred
    }
    if(pred[0] === 'Spam'){
        resultColor.value = 'red'
      } else if (pred[0] === 'Safe'){
        resultColor.value = 'green'
      } else {
        resultColor.value = 'white'
      }
  } catch (err: unknown) {
    error.value = err instanceof Error ? err.message : 'An unknown error occurred'
  }
  addResult("not done yet", message.value, modelChoice.value, predictionLabel.value, confidence.value);
}
</script>

<template>
  <h1>Spam Checker</h1>
  <div class="results-container">
  <div v-for="result in results" :key="result.id" class="result" 
    :class="{
      spam: result.prediction === 'Spam',
      safe: result.prediction === 'Safe'
    }">
    <p class="message">
      <img src="@/assets/email.png" alt="email icon" width="50">
      {{ result.message }}
    </p>
    <p class="metrics">
      Prediction using {{ result.model }} model: {{ result.prediction }} <br />
      <span v-if="result.confidence !== null"> (Confidence: {{ result.confidence.toFixed(2) }})</span>
    </p>
  </div>
  </div>


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

