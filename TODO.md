- Add slice for batch generation in triple extraction and concept generation
- Support Chinese Prompt

The slice on texts first approach is generally better because:

Better Parallelization:

Each worker gets complete documents to process

Avoids partial document processing across workers

Improved Fault Tolerance:

If a worker fails, it's clear which documents need reprocessing

No half-processed documents across workers

More Balanced Load:

Documents are more evenly distributed than chunks

Avoids situations where one worker gets many small chun