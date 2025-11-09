<script setup lang="ts">
import { ref, nextTick } from 'vue' //importing vue features
import { reactive } from "vue";
const modelChoice = ref('sms') //declaring variables as references
const message = ref('')
const prediction = ref('')
const confidence = ref<number | null>(null)
const error = ref('')
import emailIcon from '@/assets/email.png' //Import images and icons
import smsIcon from '@/assets/sms.png'
import instructions from '@/assets/instructions.svg'
const multiResults = ref<{ [key: string]: any } | null>(null) //array for results from querying multiple models
const modelNames: Record<string, string> = {
  GRU: 'Gated Recurrent Unit',
  LR: 'Logistic Regression',
  NBSMS: 'Naive Bayes SMS',
  SVM: 'Support Vector Machine',
  NBE: 'Naive Bayes Email'
}; //Dictionary for model full names
const modelOptions = ['email','sms'] // model options for dropdown
interface ModelResult {
  label: string;
  confidence: number | null;
}
interface Result { //result object with all of the necessary variables
  id: number;
  type: string;
  message: string;
  modelresults: { [key: string]: ModelResult };
  prediction: string;
  confidence?: number | null;
  isExpanded: boolean;
}
const getPieClass = (result: any) => { //function to get pie chart class based on confidence and label, which determines the background gif
  // High confidence safe
  if (result.label === 'Safe' && result.confidence > 0.8) {
    return 'pie-safe-high';
  }
  // High confidence spam
  if (result.label === 'Spam' && result.confidence > 0.8) {
    return 'pie-spam-high';
  }
  // Low confidence spam
  return 'pie-spam-low';
};
const results = reactive<Result[]>([]);
function addResult(type: string, message: string, modelresults: { [key: string]: ModelResult }, prediction: string, confidence?: number | null) { //function to add a result to the results array
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

const scrollToBottom = () => { //function to force-scroll to the bottom of the results container
  if (scrollContainer.value) {
    scrollContainer.value.scrollTo({
      top: scrollContainer.value.scrollHeight,
      behavior: 'smooth'
    });
  }
};
async function submitForm() {  //function to submit the form and get results from the backend
  if (!message.value.trim()) { //check for empty message
    alert('Please enter a message before submitting.');
    return;
  }
  
  if (message.value.length > 300) { //check for message length over 300 characters
    alert('Message is too long. Please keep it under 300 characters.');
    return;
  }
  try { //tries to query the API Detection endpoint
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
    console.log(multiResults.value) //console log for debugging reasons
    if (multiResults.value && modelChoice.value === 'sms') { //sets primary classification to the results from LR model if sms is chosen
      prediction.value = multiResults.value.LR.label;
      confidence.value = multiResults.value.LR.confidence;
    }
    else if (multiResults.value && modelChoice.value === 'email') {
      prediction.value = multiResults.value.NBE.label; //sets primary classification to the results from NBE model if email is chosen
      confidence.value = multiResults.value.NBE.confidence;
    }
    addResult(modelChoice.value, message.value, multiResults.value ?? {}, prediction.value, confidence.value); //adds the result to the results array
  } catch (err: unknown) {
    error.value = err instanceof Error ? err.message : 'An unknown error occurred'
  }


  
  await nextTick();
  scrollToBottom(); //scrolls to the bottom after adding a new result

}

</script>

<template>
  <h1>Spam Checker</h1>
  <img v-if="results.length < 1" :src="instructions" alt="Instructions" style="width:40vmin;"/><!--shows instruction card if there are no results in the results array-->
  <div class="results-container" ref="scrollContainer"> <!--results container for scrolling purposes-->

  <div v-for="result in results" :key="result.id" class="result" :class="{
      spam: result.prediction === 'Spam',
      safe: result.prediction === 'Safe'
    }" ><!--adds a div for every result, sets the class of the result div based on prediction which determines styling-->

    <div class="result-content">
    <p class="message"><!--card which shows the inputted message and the icon corresponding to the type-->
      <img :src="result.type === 'email' ? emailIcon : smsIcon" 
           alt="icon" 
           class="icon" />
      {{ result.message }}<br />
    </p>
    <div class="metrics"><!--displays model output, with bar indicating confidence level-->
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
    <button @click="result.isExpanded = !result.isExpanded"><!--show more button-->
    {{ result.isExpanded ? 'Show Less' : 'Show More' }}
    </button>  
    </div>
    </div>

    <Transition name="expand">
    <div v-if="result.isExpanded" class="detailed-results"><!--shows detailed results when expanded-->
      <div v-for="(modelResult, modelName) in result.modelresults" :key="modelName" class="model-result">
        {{ modelNames[modelName] }}:
        {{ modelResult.label }}
        <!--pie charts-->
        <div v-if="modelResult.confidence != null" class="pie" :class="getPieClass(modelResult)" :style="{  
    '--percentage': (modelResult.confidence * 100).toFixed(0),
    '--end-percentage': (modelResult.confidence * 100).toFixed(0),
    '--fill-color': modelResult.label === 'Spam' ? 'red' : 'green',
    '--back-color': modelResult.label === 'Spam' ? 'black' : 'black',

  }">
          {{ (modelResult.confidence * 100).toFixed(0) }}%
        </div>
      </div>
    </div>
    </Transition>
    

  </div>
  </div>


  <div v-if="error">Error: {{ error }}</div>
 

    <form @submit.prevent="submitForm" class="input-row"> <!--input form (search bar style)-->

      <select v-model="modelChoice" ><!--drop down menu to select type of model, takes from pre-defined list-->
        <option v-for="option in modelOptions" :key="option" :value="option">
          {{ option }}
        </option>
      </select>

      <div style="position: relative; display: contents;">
      <textarea v-model="message" placeholder="Enter message" maxlength="300"></textarea><!--input box where you can input the message-->
      <div 
      style="position: absolute; align-self: flex-end; font-size: 2vmin; pointer-events: none;"
      :style="{ color: message.length >= 300 ? 'red' : message.length > 250 ? 'orange' : 'lightgreen' }"
      >
      {{ message.length }} / 300 <!--shows the amount of characters that have been used-->
      </div>
      </div>



      <button type="submit" >Check!</button><!--CHECK button to send the query-->

    </form>

    

    
  

</template>

