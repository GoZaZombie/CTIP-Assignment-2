<script setup>
import { ref } from 'vue'

const modelChoice = ref('ModelA')
const message = ref('')
const predictionLabel = ref('')
const confidence = ref(null)
const error = ref('')

const modelOptions = ['LR', 'NBE', 'NBSMS', 'SVM', 'GRU']

async function submitForm() {
  predictionLabel.value = ''
  confidence.value = null
  error.value = ''

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
  } catch (err) {
    error.value = err.message
  }
}
</script>

<template>
  <div>
    <form @submit.prevent="submitForm">
      <select v-model="modelChoice">
        <option v-for="option in modelOptions" :key="option" :value="option">
          {{ option }}
        </option>
      </select>

      <input v-model="message" type="text" placeholder="Enter message" />

      <button type="submit">Submit</button>
    </form>

    <div v-if="predictionLabel">
      Prediction: {{ predictionLabel }}
      <span v-if="confidence !== null"> (Confidence: {{ confidence }})</span>
    </div>

    <div v-if="error">Error: {{ error }}</div>
  </div>
</template>
