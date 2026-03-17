const BASE_URL = "http://127.0.0.1:8000";

/* ======================
   AUTH
   ====================== */
export async function login(username) {
  const res = await fetch(`${BASE_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Login failed");
  return data;
}

export async function switchBatch(username, batchId) {
  const res = await fetch(`${BASE_URL}/auth/switch-batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username: username, batch_id: batchId })
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Switching course failed");
  return data;
}

/**
 * Polls the job status endpoint until completion or failure.
 */
async function pollJob(jobId) {
  // For long videos, we might need longer. Let's do exponential backoff or just longer timeout.
  // Actually, YouTube ingestion can take minutes. 
  // Let's loop indefinitely with a reasonable cap (e.g., 10 minutes).
  
  const startTime = Date.now();
  const timeoutMs = 10 * 60 * 1000; // 10 minutes

  while (Date.now() - startTime < timeoutMs) {
    const res = await fetch(`${BASE_URL}/jobs/${jobId}`);
    if (!res.ok) throw new Error("Failed to check job status");
    
    const job = await res.json();
    
    if (job.status === "completed") {
      return job;
    }
    
    if (job.status === "failed") {
      throw new Error(job.message || "Job failed");
    }
    
    // Wait 2s before next poll
    await new Promise(r => setTimeout(r, 2000));
  }
  
  throw new Error("Operation timed out");
}

/* ======================
   CHAT
   ====================== */
export async function sendMessage(username, activeBatchId, message, complexity = "Undergrad", tutorMode = false) {
  const res = await fetch(`${BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username: username, active_batch_id: activeBatchId, message, complexity, tutor_mode: tutorMode })
  });

  const data = await res.json();

  if (!res.ok) {
    throw new Error(data.detail || "Unknown server error");
  }

  return data.reply;
}

/* ======================
   DOCUMENT UPLOAD
   ====================== */
export async function uploadDocument(file, batchId = "default_batch") {
  const formData = new FormData();
  formData.append("file", file);

  // 1. Initiate Upload (Returns Job ID)
  const res = await fetch(`${BASE_URL}/upload?batch_id=${batchId}`, {
    method: "POST",
    body: formData
  });

  const data = await res.json();

  if (!res.ok) {
    throw new Error(data.detail || "Upload failed");
  }

  // 2. Poll for Completion
  return await pollJob(data.job_id);
}

/* ======================
   YOUTUBE INGESTION
   ====================== */
export async function ingestYoutube(url, batchId = "default_batch") {
  // 1. Initiate Ingestion (Returns Job ID)
  const res = await fetch(`${BASE_URL}/ingest/url`, { // Standardized to universal URL endpoint
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, batch_id: batchId })
  });

  const data = await res.json();

  if (!res.ok) {
    throw new Error(data.detail || "YouTube ingestion failed");
  }

  // 2. Poll for Completion
  return await pollJob(data.job_id);
}

/* ======================
   WEB INGESTION
   ====================== */
export async function ingestWeb(url, mode = "single", batchId = "default_batch") {
  // 1. Initiate Ingestion (Returns Job ID)
  const res = await fetch(`${BASE_URL}/ingest/url`, { // Standardized to universal URL endpoint
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, mode, batch_id: batchId })
  });

  const data = await res.json();

  if (!res.ok) {
    throw new Error(data.detail || "Web ingestion failed");
  }

  // 2. Poll for Completion
  return await pollJob(data.job_id);
}

/* ======================
   UNIVERSAL URL INGESTION (SMART)
   ====================== */
export async function ingestSmartUrl(url, mode = "single", batchId = "default_batch") {
  const res = await fetch(`${BASE_URL}/ingest/url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, mode, batch_id: batchId })
  });

  const data = await res.json();

  if (!res.ok) {
    throw new Error(data.detail || "Smart ingestion failed");
  }

  return await pollJob(data.job_id);
}

/* ======================
   BATCH URL INGESTION (S3 PRESIGNED)
   ====================== */
export async function ingestBatchUrls(urls, batchId = "default_batch") {
  const res = await fetch(`${BASE_URL}/ingest/batch-urls`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ urls, batch_id: batchId })
  });

  const data = await res.json();

  if (!res.ok) {
    throw new Error(data.detail || "Batch ingestion failed");
  }

  return await pollJob(data.job_id);
}

/* ======================
   METRICS
   ====================== */
export async function fetchMetrics(range = '24h') {
  const res = await fetch(`${BASE_URL}/metrics?range=${range}`);
  const data = await res.json();
  
  if (!res.ok) {
    throw new Error("Failed to fetch metrics");
  }
  
  return data;
}

export async function clearMetrics() {
  const res = await fetch(`${BASE_URL}/metrics`, {
    method: "DELETE"
  });
  if (!res.ok) throw new Error("Failed to clear metrics");
  return await res.json();
}

/* ======================
   STUDIO (FLASHCARDS)
   ====================== */
export async function fetchFlashcards(username, activeBatchId) {
  const res = await fetch(`${BASE_URL}/studio/flashcards?username=${username}&batch_id=${activeBatchId}`);
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Failed to fetch flashcards");
  return data.cards;
}

export async function deleteFlashcards(username, activeBatchId, topic) {
  const res = await fetch(`${BASE_URL}/studio/flashcards?username=${username}&batch_id=${activeBatchId}&topic=${encodeURIComponent(topic)}`, {
    method: "DELETE"
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Failed to delete flashcards");
  return data;
}

export async function generateFlashcards(username, activeBatchId, topics = null) {
  const res = await fetch(`${BASE_URL}/studio/flashcards`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username: username, active_batch_id: activeBatchId, topics })
  });
  
  const data = await res.json();
  
  if (!res.ok) {
    throw new Error(data.detail || "Failed to generate flashcards");
  }
  
  return data.cards;
}

/* ======================
   STUDIO (QUIZZES)
   ====================== */
export async function fetchQuizzes(username, activeBatchId) {
  const res = await fetch(`${BASE_URL}/studio/quizzes?username=${username}&batch_id=${activeBatchId}`);
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Failed to fetch quizzes");
  return data.quizzes;
}

export async function deleteQuiz(username, activeBatchId, topic) {
  const res = await fetch(`${BASE_URL}/studio/quiz?username=${username}&batch_id=${activeBatchId}&topic=${encodeURIComponent(topic)}`, {
    method: "DELETE"
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Failed to delete quiz");
  return data;
}

export async function generateQuiz(username, activeBatchId, topics) {
  const res = await fetch(`${BASE_URL}/studio/quiz/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username: username, active_batch_id: activeBatchId, topics })
  });
  
  const data = await res.json();
  
  if (!res.ok) {
    throw new Error(data.detail || "Failed to generate quiz");
  }
  
  return data;
}

/* ======================
   STUDIO (PODCASTS)
   ====================== */
export async function fetchPodcasts(username, activeBatchId) {
  const res = await fetch(`${BASE_URL}/studio/podcasts?username=${username}&batch_id=${activeBatchId}`);
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Failed to fetch podcasts");
  return data.podcasts;
}

export async function deletePodcast(username, activeBatchId, topic) {
  const res = await fetch(`${BASE_URL}/studio/podcast?username=${username}&batch_id=${activeBatchId}&topic=${encodeURIComponent(topic)}`, {
    method: "DELETE"
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Failed to delete podcast");
  return data;
}

export async function generatePodcast(username, activeBatchId, topic) {
  const res = await fetch(`${BASE_URL}/studio/podcast/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username: username, active_batch_id: activeBatchId, topic })
  });
  
  const data = await res.json();
  
  if (!res.ok) {
    throw new Error(data.detail || "Failed to generate podcast");
  }
  
  // Wait for the background process to finish before returning
  return await pollJob(data.job_id);
}

/* ======================
   SOURCE MANAGEMENT
   ====================== */
export async function fetchSources(batchId = null) {
  let url = `${BASE_URL}/sources`;
  if (batchId) {
    url += `?batch_id=${batchId}`;
  }
  const res = await fetch(url);
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Failed to fetch sources");
  return data.sources;
}

export async function deleteSource(documentId) {
  const res = await fetch(`${BASE_URL}/sources/${documentId}`, {
    method: "DELETE"
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Failed to delete source");
  return data;
}

export async function clearAllSources() {
  const res = await fetch(`${BASE_URL}/sources`, {
    method: "DELETE"
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Failed to clear sources");
  return data;
}