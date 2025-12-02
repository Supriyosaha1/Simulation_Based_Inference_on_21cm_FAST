#!/bin/bash
# Monitor improved training job

JOB_ID="1445759"
LOG_DIR="/user1/supriyo/ml_project/SBI_step_by_step/UGMRT_500h"

echo "=========================================="
echo "MONITORING IMPROVED TRAINING JOB"
echo "=========================================="
echo "Job ID: $JOB_ID"
echo "=========================================="
echo ""

# Check job status
echo "📊 Job Status:"
qstat $JOB_ID 2>/dev/null || echo "   (Job may be completed)"
echo ""

# Check output log
echo "📝 Output Log:"
if [ -f "$LOG_DIR/step_04_train_improved.log" ]; then
    echo "   Found! Last 30 lines:"
    echo "   ---"
    tail -30 "$LOG_DIR/step_04_train_improved.log"
else
    echo "   Not yet created..."
fi
echo ""

# Check training log
echo "🔍 Training Details (training_log.txt):"
TRAINING_LOG="$LOG_DIR/Outputs/training_log.txt"
if [ -f "$TRAINING_LOG" ]; then
    echo "   Found! Last 40 lines:"
    echo "   ---"
    tail -40 "$TRAINING_LOG"
    echo "   ---"
else
    echo "   Not yet created (training may still be initializing)..."
fi
echo ""

# Check if model was saved
echo "💾 Model Files:"
POSTERIOR_FILE="$LOG_DIR/Outputs/posterior_snpe.pt"
if [ -f "$POSTERIOR_FILE" ]; then
    SIZE=$(du -h "$POSTERIOR_FILE" | cut -f1)
    echo "   ✓ posterior_snpe.pt exists (size: $SIZE)"
else
    echo "   ⏳ Still training..."
fi
echo ""

echo "=========================================="
echo "Tip: Run this script periodically to monitor progress"
echo "=========================================="
