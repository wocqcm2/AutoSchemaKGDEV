#!/bin/bash
# kg_extraction_parallel.sh - Run knowledge graph extraction across 3 parallel slices

TOTAL_SLICES=3
LOG_DIR="/home/httsangaj/projects/AutoSchemaKG/log"
SCRIPT_DIR="/home/httsangaj/projects/AutoSchemaKG/example_scripts"
# Function to run a single slice
run_slice() {
    local slice_num=$1
    echo "Starting slice ${slice_num}/${TOTAL_SLICES}"
    python $SCRIPT_DIR/1_slice_kg_extraction.py \
        --slice "${slice_num}" \
        --total_slices "${TOTAL_SLICES}" \
        > "${LOG_DIR}/slice_${slice_num}.log" 2>&1 &
}

# Run all slices in parallel
for ((i=0; i<TOTAL_SLICES; i++)); do
    run_slice $i
done

# Wait for all processes to complete
echo "All slices started. Waiting for completion..."
wait
echo "All slices completed."

# Merge results (if needed)
echo "Merging results..."
# Add your merge commands here if needed

echo "Knowledge graph extraction complete."