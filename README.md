# Fine-tuning-gpt-model
OpenAI GPT model fine-tuning and evaluation workflows. This github inlcudes model training automation, and performance evaluation with confidence scoring.

## 🚀 Features

- **Automated Fine-Tuning**: Streamlined pipeline for training custom GPT models
- **Multi-Client Support**: Manage fine-tuning for multiple clients/projects simultaneously
- **Batch Processing**: Efficient async processing for large datasets
- **Confidence Scoring**: Evaluate model predictions with probability-based confidence metrics
- **Comprehensive Logging**: Detailed logs for monitoring training progress
- **Flexible Configuration**: YAML-based configuration for easy customization

## 📁 Project Structure

```
openai-finetuning-suite/
├── README.md
├── requirements.txt
├── config.yaml
├── finetuning_automator.py    # Main fine-tuning automation script
├── model_tester.py            # Model evaluation and testing script
├── data/                      # Training and test data
│   ├── client_a/
│   └── test_data/
├── output/                    # Fine-tuning outputs
├── results/                   # Evaluation results
├── logs/                      # Application logs
└── examples/                  # Example configurations and data
```

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/openai-finetuning-suite.git
cd openai-finetuning-suite
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up your OpenAI API key:**
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

4. **Create configuration file:**
```bash
python finetuning_automator.py
```
This will create a `config.yaml` file that you can customize.

## 📋 Requirements

```
pandas>=1.5.0
openai>=1.0.0
scikit-learn>=1.0.0
PyYAML>=6.0
tqdm>=4.64.0
nest-asyncio>=1.5.0
openpyxl>=3.1.0
```



## 🚀 Usage

### Fine-Tuning Models

#### 1. Prepare Your Data

Ensure your training data CSV files have these columns:
- `sentence__pre_process_desc`: Input text for training
- `topic__description`: Expected output/label

Place files in the structure:
```
data/client_name/feedback_type/Client Name_feedback_type_training.csv
```

#### 2. Run Fine-Tuning

**Option 1: Process all clients with both feedback types**
```python
from finetuning_automator import OpenAIFineTuningAutomator

automator = OpenAIFineTuningAutomator()
results = automator.process_all_clients()
```

**Option 2: Process single feedback type**
```python
# Process only negative feedback
results = automator.process_single_feedback_type()

# Process only positive feedback
results = automator.process_single_feedback_type()
```

**Option 3: Process specific client**
```python
result = automator.process_client("client_a")
```

#### 3. Monitor Training Jobs

```python
# Check job status
job_status = automator.check_job_status("your-job-id")

# List recent jobs
recent_jobs = automator.list_recent_jobs(limit=10)

# Cancel a job if needed
cancelled_job = automator.cancel_job("your-job-id")
```

### Model Evaluation

#### 1. Prepare Test Data

Create test files with the same column structure as training data:
- Excel (`.xlsx`) or CSV (`.csv`) format
- Must contain `sentence__pre_process_desc` column

#### 2. Run Evaluation

```python
from model_tester import run_model_evaluation

# Evaluate model performance
results = run_model_evaluation(
    test_file_path="data/test_data/",
    model_id="ft:gpt-4o-mini-2024-07-18:your-org:your-model:abc123",
    output_file_path="results/"
)
```

#### 3. Analyze Results

The evaluation outputs a CSV file with:
- Original test data
- `original_text`: Input text
- `predicted_category`: Model prediction
- `confidence_score`: Prediction confidence (0.0-1.0)

## 📊 Understanding Outputs

### Fine-Tuning Outputs

Each fine-tuning job creates:
- **Training/Validation JSONL files**: OpenAI-formatted training data
- **Job info JSON**: Complete job metadata and configuration
- **Logs**: Detailed processing logs

### Evaluation Outputs

Model evaluation provides:
- **Predictions**: Model outputs for each test sample
- **Confidence Scores**: Probability-based confidence metrics
- **Merged Results**: Original data combined with predictions

## 🔧 Advanced Usage

### Custom Base Models

Use previously fine-tuned models as base models:


### Batch Processing Configuration

Adjust batch sizes for your API limits:

```python
# In model_tester.py, modify:
api_call_batch_size = 5  # Reduce for lower rate limits
```

### Custom Data Preprocessing

Override the preprocessing method:

```python
class CustomAutomator(OpenAIFineTuningAutomator):
    def preprocess_data(self, df):
        # Add your custom preprocessing logic
        df = super().preprocess_data(df)
        # Additional processing...
        return df
```

## 🚨 Important Notes

### Rate Limits
- OpenAI has rate limits for API calls and fine-tuning jobs
- The async processing includes built-in batching to respect limits
- Monitor your usage in the OpenAI dashboard

### Data Privacy
- Ensure your training data complies with OpenAI's usage policies
- Remove any sensitive information before training
- Consider data retention policies for uploaded files

### Cost Management
- Fine-tuning costs depend on model size and training data volume
- Monitor costs in your OpenAI dashboard
- Delete unused fine-tuned models to avoid storage costs

## 🐛 Troubleshooting

### Common Issues

**"Config file not found"**
```bash
# Run the script once to generate default config
python finetuning_automator.py
```

**"API key not set"**
```bash
# Set environment variable
export OPENAI_API_KEY="your-key-here"
# Or update config.yaml directly
```

**"Data file not found"**
- Check file paths in config.yaml
- Ensure data files follow the naming convention
- Verify directory structure matches configuration

**"Rate limit exceeded"**
- Reduce batch sizes in the code
- Add delays between API calls
- Check your OpenAI plan limits

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

---

**Note**: This tool is designed for educational and research purposes. Always follow OpenAI's usage policies and guidelines when fine-tuning models.
