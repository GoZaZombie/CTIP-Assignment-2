<script setup lang="ts">
import { ref, nextTick } from 'vue'
import { reactive } from "vue";
const modelChoice = ref('sms')
const message = ref('')
const prediction = ref('')
const confidence = ref<number | null>(null)
const error = ref('')
import emailIcon from '@/assets/email.png'
import smsIcon from '@/assets/sms.png'
const multiResults = ref<{ [key: string]: any } | null>(null)

const modelOptions = ['email','sms']
interface ModelResult {
  label: string;
  confidence: number | null;
}
interface ApiResponse {
  [key: string]: ModelResult; // Dynamic keys like "NBSMS", "LR", etc.
}
interface Result {
  id: number;
  type: string;
  message: string;
  modelresults: { [key: string]: ModelResult };
  prediction: string;
  confidence?: number | null;
  isExpanded: boolean;
}

const results = reactive<Result[]>([]);
function addResult(type: string, message: string, modelresults: { [key: string]: ModelResult }, prediction: string, confidence?: number | null) {
    results.push({
    id: Date.now(),
    type,
    message,
    modelresults,
    prediction,
    confidence: confidence ?? null,
    isExpanded: false,
  });
}
const scrollContainer = ref<HTMLElement | null>(null);

const scrollToBottom = () => {
  if (scrollContainer.value) {
    scrollContainer.value.scrollTo({
      top: scrollContainer.value.scrollHeight,
      behavior: 'smooth'
    });
  }
};
async function submitForm() { 
  try {
    const response = await fetch('http://127.0.0.1:8000/CLASSIFY/Detection', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ModelChoice: modelChoice.value === 'email' ? "ALLE" : "ALLSMS",
        Message: message.value,
      }),
    })

    if (!response.ok) throw new Error('Request failed')
    
    const data = await response.json()
    multiResults.value = data.results
    console.log(multiResults.value)
    if (multiResults.value && modelChoice.value === 'sms') {
      prediction.value = multiResults.value.LR.label;
      confidence.value = multiResults.value.LR.confidence;
    }
    else if (multiResults.value && modelChoice.value === 'email') {
      prediction.value = multiResults.value.NBE.label;
      confidence.value = multiResults.value.NBE.confidence;
    }
    addResult(modelChoice.value, message.value, multiResults.value ?? {}, prediction.value, confidence.value);
  } catch (err: unknown) {
    error.value = err instanceof Error ? err.message : 'An unknown error occurred'
  }


  
  await nextTick();
  scrollToBottom();

}

</script>

<template>
  <h1>Spam Checker</h1>
  <div class="results-container" ref="scrollContainer">
  <div v-for="result in results" :key="result.id" class="result" 
    :class="{
      spam: result.prediction === 'Spam',
      safe: result.prediction === 'Safe'
    }" >
    <div class="result-content">
    <p class="message">
      <img :src="result.type === 'email' ? emailIcon : smsIcon" 
           alt="icon" 
           width="50" />
      {{ result.message }}<br />
    </p>
    <div class="metrics">
      <p>Prediction using {{ result.type }} models: {{ result.prediction }} <br />
       <span v-if="result.confidence != null"> Confidence: {{ (result.confidence*100).toFixed(0) }}%</span>
      </p>
      <div class="bar" v-if="result.confidence != null">
      <div class="bar_fill" :style="{ '--final-width': (result.confidence*100) + '%'}" :class="{
      spambar: result.prediction === 'Spam',
      safebar: result.prediction === 'Safe'
      }">
      </div>
      </div>
    <button @click="result.isExpanded = !result.isExpanded">
    {{ result.isExpanded ? 'Show Less' : 'Show More' }}
    </button>  
    </div>
    <div v-if="result.isExpanded" class="detailed-results">
      <div v-for="(modelResult, modelName) in result.modelresults" :key="modelName" class="model-result">
        {{ modelName }}:
        {{ modelResult.label }}
        <p v-if="modelResult.confidence != null">Confidence: {{ (modelResult.confidence * 100).toFixed(2) }}%</p>
      </div>
    </div>
    
    
    </div>
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

