const { request } = require('playwright');
const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');

// Load environment variables
dotenv.config();

// CONFIGURATION
const BASE_URL = 'http://127.0.0.1:8000';
const BATCH_IDS = [
  '8a861151-5d27-4daf-bb1b-6cf9deeff22c',
  'b33f8473-db92-494d-9216-0cde7e1854e5'
];

// Load all 10 API keys
const API_KEYS = [];
for (let i = 1; i <= 10; i++) {
  const key = process.env[`GEMINI_API_KEY_${i}`];
  if (key) API_KEYS.push(key);
}

// Fallback to primary if numbered keys are missing
if (API_KEYS.length === 0 && process.env.GEMINI_API_KEY) {
  API_KEYS.push(process.env.GEMINI_API_KEY);
}

if (API_KEYS.length === 0) {
  console.error("❌ No API keys found. Please set GEMINI_API_KEY_1 to GEMINI_API_KEY_10 in .env");
  process.exit(1);
}

const LEARNERS = ['nithish-learner', 'abirami-learner', 'sathish-learner'];

// QUESTION BANK
const QUESTIONS = {
  '8a861151-5d27-4daf-bb1b-6cf9deeff22c': [
    "What is Postman and what is its primary purpose in API development?",
    "How does Postman simplify the process of building and testing APIs?",
    "Which HTTP methods are supported by Postman?",
    "What is the role of a GET request in API communication?",
    "What are collections in Postman and why are they useful?",
    "What is Node.js and why is it considered cross-platform?",
    "How can Node.js handle thousands of concurrent connections efficiently?",
    "What is ReactJS and how does it differ from other frameworks like AngularJS?",
    "What architectural pattern is commonly associated with ReactJS?"
  ],
  'b33f8473-db92-494d-9216-0cde7e1854e5': [
    "What is cloud computing?",
    "How are cloud services delivered to clients?",
    "What types of resources are included in cloud computing?",
    "How does virtualization contribute to cloud computing?",
    "What does scalability mean in cloud computing?",
    "What is elasticity in cloud systems?",
    "What is a hypervisor?",
    "What are Type 1 and Type 2 hypervisors?",
    "What is KVM and how does it work in Linux?"
  ]
};

const BATCH_LEARNERS = {
  '8a861151-5d27-4daf-bb1b-6cf9deeff22c': ['nithish-learner', 'sathish-learner', 'abirami-learner', 'sakthiram-learner'],
  'b33f8473-db92-494d-9216-0cde7e1854e5': ['sathish-learner', 'abirami-learner', 'sakthiram-learner', 'harshini-learner']
};

// LOGGING
const LOG_FILE = path.join(__dirname, '../logs/stress_test_results.csv');
if (!fs.existsSync(path.dirname(LOG_FILE))) fs.mkdirSync(path.dirname(LOG_FILE));
fs.writeFileSync(LOG_FILE, 'Timestamp,User_Name,Concurrency,Batch_ID,Status,Total_RTT,Retrieval_ms,LLM_ms,Key_Idx\n');

function logResult(data) {
  const row = `${new Date().toISOString()},${data.userName},${data.concurrency},${data.batchId},${data.status},${data.rtt},${data.retrieval_ms},${data.llm_ms},${data.keyIdx}\n`;
  fs.appendFileSync(LOG_FILE, row);
}

// SINGLE REQUEST ACTION
let globalRequestCount = 0;
async function fireRequest(reqId, concurrency) {
  const context = await request.newContext();

  // 1. Pick Batch
  const batchId = BATCH_IDS[reqId % BATCH_IDS.length];

  // 2. Pick Learner based on Batch
  const learnerList = BATCH_LEARNERS[batchId];
  const userName = learnerList[reqId % learnerList.length];

  // 3. Pick Question
  const questionList = QUESTIONS[batchId];
  const message = questionList[Math.floor(Math.random() * questionList.length)];

  // 4. Pick API Key
  const keyIdx = globalRequestCount % API_KEYS.length;
  const apiKey = API_KEYS[keyIdx];
  globalRequestCount++;

  const t0 = Date.now();
  try {
    const response = await context.post(`${BASE_URL}/chat`, {
      data: {
        username: userName,
        active_batch_id: batchId,
        message: message,
        complexity: "Undergrad",
        tutor_mode: false
      },
      headers: {
        'X-Gemini-API-Key': apiKey,
        'Content-Type': 'application/json'
      },
      timeout: 120000 // 120s timeout for heavy stress
    });

    const rtt = Date.now() - t0;
    const status = response.status();
    let debug = { retrieval_ms: 0, llm_ms: 0 };
    
    if (status === 200) {
      const body = await response.json();
      debug = body.debug_timings || debug;
    }

    logResult({
      userName,
      concurrency,
      batchId,
      status,
      rtt,
      retrieval_ms: debug.retrieval_ms || 0,
      llm_ms: debug.llm_ms || 0,
      keyIdx
    });

    const timingInfo = status === 200 
      ? `(Retrieval: ${debug.retrieval_ms}ms, LLM: ${debug.llm_ms}ms)` 
      : `(ERROR: ${status})`;

    console.log(`   [REQ] User: ${userName.padEnd(16)} | Batch: ${batchId.slice(0,8)}... | Status: ${status} | RTT: ${rtt}ms ${timingInfo}`);

    return { status, rtt };
  } catch (err) {
    logResult({
      userName,
      concurrency,
      batchId,
      status: 'ERR',
      rtt: Date.now() - t0,
      retrieval_ms: 0,
      llm_ms: 0,
      keyIdx
    });
    return { status: 'ERR', rtt: Date.now() - t0 };
  }
}

async function runStage(concurrency, durationSec) {
  console.log(`\n🚀 STAGE START: ${concurrency} Concurrent Burst for ${durationSec}s`);
  const startTime = Date.now();
  let waveCount = 1;

  while (Date.now() - startTime < durationSec * 1000) {
    console.log(`🌊 Wave ${waveCount}: Firing ${concurrency} simultaneous requests...`);
    const t_wave_start = Date.now();
    
    const requests = [];
    for (let i = 0; i < concurrency; i++) {
      requests.push(fireRequest(i, concurrency));
    }

    // REVOKED WAIT: We fire all concurrency at once.
    // We wait for the WHOLE wave to complete to measure the 10s goal for the "slowest" user.
    const results = await Promise.all(requests);
    const waveDuration = Date.now() - t_wave_start;
    
    const success = results.filter(r => r.status === 200).length;
    console.log(`📊 Wave ${waveCount} Complete: ${success}/${concurrency} succeeded. Max Latency: ${waveDuration}ms`);

    waveCount++;
    // Interval between bursts to allow backend recovery
    await new Promise(r => setTimeout(r, 2000));
  }
}

async function main() {
  const stages = [5, 10, 15, 20, 30, 40, 50, 100];
  const stageDuration = 60; // 1 minute per stage

  console.log("🔥 STARTING SOPHISTICATED BURST CONCURRENCY TEST");
  console.log(`👥 Using users: ${LEARNERS.join(', ')}`);
  console.log(`🔑 Using ${API_KEYS.length} Round-Robin Keys`);
  console.log("--------------------------------------------------");

  for (const concurrency of stages) {
    await runStage(concurrency, stageDuration);
    console.log("⏸️ Cooling down for 10s...");
    await new Promise(r => setTimeout(r, 10000));
  }

  console.log("\n🏆 TEST COMPLETE. Results in logs/stress_test_results.csv");
}

main().catch(console.error);
