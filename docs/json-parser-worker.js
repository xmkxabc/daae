// Enhanced Web Worker for JSON parsing with performance optimizations
let workerState = {
    isProcessing: false,
    lastProgressTime: Date.now(),
    totalPapers: 0,
    processedCount: 0,
    startTime: 0
};

// Self-monitoring: send heartbeat to detect stuck state
function sendHeartbeat() {
    if (workerState.isProcessing) {
        self.postMessage({
            type: 'heartbeat',
            timestamp: Date.now(),
            processed: workerState.processedCount,
            total: workerState.totalPapers
        });
    }
}

// Start heartbeat monitoring
setInterval(sendHeartbeat, 5000); // Every 5 seconds

self.onmessage = function(e) {
    const { url, month, config = {} } = e.data;
    
    // Reset worker state
    workerState = {
        isProcessing: true,
        lastProgressTime: Date.now(),
        totalPapers: 0,
        processedCount: 0,
        startTime: Date.now()
    };
    
    // Send initial status
    self.postMessage({
        type: 'started',
        month: month,
        timestamp: workerState.startTime
    });
    
    fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            // Report fetch completion
            self.postMessage({
                type: 'fetch_complete',
                month: month,
                contentLength: response.headers.get('content-length')
            });
            
            return response.json();
        })
        .then(papers => {
            workerState.totalPapers = papers.length;
            
            // Calculate dynamic batch size based on data size and configuration
            const dynamicBatchSize = calculateOptimalBatchSize(papers.length, config);
            
            self.postMessage({
                type: 'processing_start',
                month: month,
                totalPapers: papers.length,
                batchSize: dynamicBatchSize
            });
            
            const batches = [];
            for (let i = 0; i < papers.length; i += dynamicBatchSize) {
                batches.push(papers.slice(i, i + dynamicBatchSize));
            }
            
            // Process batches with improved progress reporting
            processBatchesWithTimeSlicing(batches, month, dynamicBatchSize);
        })
        .catch(error => {
            workerState.isProcessing = false;
            self.postMessage({
                type: 'error',
                month: month,
                error: error.message,
                timestamp: Date.now()
            });
        });
};

// Calculate optimal batch size based on data volume and system capabilities
function calculateOptimalBatchSize(totalPapers, config) {
    const baseBatchSize = config.baseBatchSize || 1000;
    const maxBatchSize = config.maxBatchSize || 2000;
    const minBatchSize = config.minBatchSize || 100;
    
    // Adjust batch size based on total data size
    let optimalSize = baseBatchSize;
    
    if (totalPapers < 1000) {
        // Small dataset: use smaller batches for more frequent progress updates
        optimalSize = Math.max(minBatchSize, Math.floor(totalPapers / 10));
    } else if (totalPapers > 10000) {
        // Large dataset: use larger batches for efficiency
        optimalSize = Math.min(maxBatchSize, Math.floor(totalPapers / 20));
    } else {
        // Medium dataset: scale batch size proportionally
        optimalSize = Math.floor(baseBatchSize * (totalPapers / 5000));
    }
    
    return Math.max(minBatchSize, Math.min(maxBatchSize, optimalSize));
}

// Process batches with time slicing to avoid blocking and provide better progress updates
function processBatchesWithTimeSlicing(batches, month, batchSize) {
    let currentBatchIndex = 0;
    
    function processNextBatch() {
        if (currentBatchIndex >= batches.length) {
            // All batches completed
            workerState.isProcessing = false;
            self.postMessage({
                type: 'complete',
                month: month,
                totalPapers: workerState.totalPapers,
                processingTime: Date.now() - workerState.startTime,
                timestamp: Date.now()
            });
            return;
        }
        
        const batch = batches[currentBatchIndex];
        const batchStartTime = Date.now();
        
        // Update progress tracking
        workerState.processedCount += batch.length;
        workerState.lastProgressTime = Date.now();
        
        // Send batch data with enhanced progress information
        const progressPercentage = Math.round(((currentBatchIndex + 1) / batches.length) * 100);
        const processingSpeed = workerState.processedCount / ((Date.now() - workerState.startTime) / 1000); // papers per second
        const estimatedTimeRemaining = (workerState.totalPapers - workerState.processedCount) / processingSpeed;
        
        self.postMessage({
            type: 'batch',
            month: month,
            papers: batch,
            progress: {
                current: workerState.processedCount,
                total: workerState.totalPapers,
                percentage: progressPercentage,
                batchIndex: currentBatchIndex + 1,
                totalBatches: batches.length,
                processingSpeed: Math.round(processingSpeed),
                estimatedTimeRemaining: Math.round(estimatedTimeRemaining),
                batchProcessingTime: Date.now() - batchStartTime
            }
        });
        
        currentBatchIndex++;
        
        // Use setTimeout to create a time slice, allowing for heartbeat and other operations
        setTimeout(processNextBatch, 0);
    }
    
    // Start processing
    processNextBatch();
}
