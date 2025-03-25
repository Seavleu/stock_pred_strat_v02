echo "🔁 Running dynamic feature selection..."
python src/dynamic_feature_selection.py

if [ $? -ne 0 ]; then
    echo "❌ Dynamic feature selection failed. Aborting."
    exit 1
fi

echo "✅ Dynamic feature selection completed."

echo "🚀 Training LSTM + Attention model..."
python src/lstm_att_model_training.py

if [ $? -ne 0 ]; then
    echo "❌ Model training failed."
    exit 1
fi

echo "🎉 Model training completed successfully."