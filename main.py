import pandas as pd
import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from openai import OpenAI
from pathlib import Path
import yaml
from dataclasses import dataclass


@dataclass
class ClientConfig:
    name: str
    base_model: str
    data_path: str
    output_dir: str
    suffix_template: str


class OpenAIFineTuningAutomator:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.client = OpenAI(api_key=self.config['openai']['api_key'])
        self._setup_logging()
        
    def _load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found. Using default config.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        return {
            'openai': {
                'api_key': 'your-openai-api-key-here'
            },
            'clients': {
                'client_a': {
                    'name': 'Client A',
                    'base_model': 'gpt-4o-mini-2024-07-18',
                    'data_path': 'data/client_a/',
                    'output_dir': 'output/client_a/',
                    'suffix_template': 'ClientA_{feedback_type}_{date}'
                },
                'client_b': {
                    'name': 'Client B',
                    'base_model': 'gpt-4o-mini-2024-07-18',
                    'data_path': 'data/client_b/',
                    'output_dir': 'output/client_b/',
                    'suffix_template': 'ClientB_{feedback_type}_{date}'
                },
                'client_c': {
                    'name': 'Client C',
                    'base_model': 'gpt-4o-mini-2024-07-18',
                    'data_path': 'data/client_c/',
                    'output_dir': 'output/client_c/',
                    'suffix_template': 'ClientC_{feedback_type}_{date}'
                },
                'client_d': {
                    'name': 'Client D',
                    'base_model': 'gpt-4o-mini-2024-07-18',
                    'data_path': 'data/client_d/',
                    'output_dir': 'output/client_d/',
                    'suffix_template': 'ClientD_{feedback_type}_{date}'
                },
                'client_e': {
                    'name': 'Client E',
                    'base_model': 'gpt-4o-mini-2024-07-18',
                    'data_path': 'data/client_e/',
                    'output_dir': 'output/client_e/',
                    'suffix_template': 'ClientE_{feedback_type}_{date}'
                },
            },
            'training': {
                'test_size': 0.1,
                'random_state': 42,
                'encoding': 'latin1'
            }
        }
    
    def _setup_logging(self):
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/finetuning_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Original dataset shape: {df.shape}")
        
        # Remove null values
        df = df.dropna(how='all')
        self.logger.info(f"After removing null values: {df.shape}")
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['sentence__pre_process_desc'])

        duplicates_removed = initial_count - len(df)
        self.logger.info(f"Duplicates removed: {duplicates_removed}")
        self.logger.info(f"Final dataset shape: {df.shape}")
        
        return df
    
    def convert_to_gpt_format(self, dataset: pd.DataFrame) -> List[Dict]:
        fine_tuning_data = []
        
        for _, row in dataset.iterrows():
            json_response = row['topic__description']
            fine_tuning_data.append({
                "messages": [
                    {"role": "user", "content": row['sentence__pre_process_desc']},
                    {"role": "assistant", "content": json_response}
                ]
            })
        
        self.logger.info(f"Converted {len(fine_tuning_data)} examples to GPT format")
        return fine_tuning_data
    
    def split_data(self, converted_data: List[Dict], dataset: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        # Filter data for stratification (only classes with more than 1 sample)
        filtered_data = dataset[dataset['topic__description'].map(
            dataset['topic__description'].value_counts()) > 1]
        
        train_data, val_data = train_test_split(
            converted_data,
            test_size=self.config['training']['test_size'],
            random_state=self.config['training']['random_state']
        )
        
        self.logger.info(f"Training samples: {len(train_data)}")
        self.logger.info(f"Validation samples: {len(val_data)}")
        
        return train_data, val_data
    
    def write_to_jsonl(self, data: List[Dict], file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as file:
            for entry in data:
                json.dump(entry, file)
                file.write('\n')
        
        self.logger.info(f"Written {len(data)} entries to {file_path}")
    
    def upload_files(self, training_file_path: str, validation_file_path: str) -> Tuple[str, str]:
        training_file = self.client.files.create(
            file=open(training_file_path, "rb"),
            purpose="fine-tune"
        )
        
        validation_file = self.client.files.create(
            file=open(validation_file_path, "rb"),
            purpose="fine-tune"
        )
        
        self.logger.info(f"Training file uploaded: {training_file.id}")
        self.logger.info(f"Validation file uploaded: {validation_file.id}")
        
        return training_file.id, validation_file.id
    
    def create_fine_tuning_job(self, training_file_id: str, validation_file_id: str,
                              base_model: str, suffix: str) -> str:
        response = self.client.fine_tuning.jobs.create(
            training_file=training_file_id,
            validation_file=validation_file_id,
            model=base_model,
            suffix=suffix,
        )
        print("FINE-TUNING RESPONSE", response)
        
        self.logger.info(f"Fine-tuning job created: {response.id}")
        return response.id
    
    def process_client(self, client_name: str, feedback_type: str = "negative"):
        if client_name not in self.config['clients']:
            raise ValueError(f"Client {client_name} not found in configuration")
        
        client_config = self.config['clients'][client_name]
        self.logger.info(f"Processing client: {client_config['name']} - {feedback_type}")
        
        # Create feedback-specific output directory
        output_dir = Path(client_config['output_dir']) / feedback_type
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load feedback-specific data file
        data_file = Path(client_config['data_path']) / feedback_type / f"{client_config['name']}_{feedback_type}_training.csv"
        
        if not data_file.exists():
            self.logger.error(f"Data file not found: {data_file}")
            return None
        
        df = pd.read_csv(data_file, encoding=self.config['training']['encoding'])
        self.logger.info(f"Loaded data file: {data_file}")
        self.logger.info(f"Data shape: {df.shape}")

        df = self.preprocess_data(df)
        
        # Convert to GPT format
        converted_data = self.convert_to_gpt_format(df)
        
        # Split data
        train_data, val_data = self.split_data(converted_data, df)
        
        # Generate feedback-specific file names with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_file = output_dir / f"train_{feedback_type}_{timestamp}.jsonl"
        val_file = output_dir / f"val_{feedback_type}_{timestamp}.jsonl"
        
        # Write JSONL files
        self.write_to_jsonl(train_data, str(train_file))
        self.write_to_jsonl(val_data, str(val_file))
        
        # Upload files
        training_file_id, validation_file_id = self.upload_files(str(train_file), str(val_file))

        # Get feedback-specific base model
        base_model = client_config.get(f'base_model_{feedback_type}', client_config['base_model'])
        self.logger.info(f"Using base model: {base_model}")
        
        # Create suffix
        date_str = datetime.now().strftime("%b%d").lower()
        # Format string example: ClientA_{feedback_type}_{date}
        # Result example: ClientA_positive_08aug
        suffix = client_config['suffix_template'].format(
            feedback_type=feedback_type,
            date=date_str
        )
        self.logger.info(f"Using suffix: {suffix}")
        
        # Create fine-tuning job
        job_id = self.create_fine_tuning_job(
            training_file_id,
            validation_file_id,
            base_model,
            suffix
        )
        
        # Save job info with feedback type
        job_info = {
            'client': client_config['name'],
            'feedback_type': feedback_type,
            'job_id': job_id,
            'training_file_id': training_file_id,
            'validation_file_id': validation_file_id,
            'base_model': base_model,
            'suffix': suffix,
            'timestamp': timestamp,
            'data_file': str(data_file),
            'train_samples': len(train_data),
            'val_samples': len(val_data)
        }

        job_file = output_dir / f"job_info_{feedback_type}_{timestamp}.json"

        with open(job_file, 'w') as f:
            json.dump(job_info, f, indent=2)
        
        self.logger.info(f"Job information saved to: {job_file}")
        return job_info
    
    def process_all_clients(self, feedback_types: List[str] = None):
        if feedback_types is None:
            feedback_types = ["negative", "positive"]
        
        results = {}
        
        for client_name in self.config['clients'].keys():
            results[client_name] = {}  # Initialize client results dictionary
            
            for feedback_type in feedback_types:
                try:
                    self.logger.info(f"Processing {client_name} - {feedback_type}")
                    result = self.process_client(client_name, feedback_type)
                    results[client_name][feedback_type] = result
                except Exception as e:
                    self.logger.error(f"Error processing {client_name} - {feedback_type}: {str(e)}")
                    results[client_name][feedback_type] = None
        
        return results

    def process_single_feedback_type(self, feedback_type: str = "negative"):
        """Process all clients for a single feedback type"""
        results = {}
        
        for client_name in self.config['clients'].keys():
            try:
                self.logger.info(f"Processing {client_name} - {feedback_type}")
                result = self.process_client(client_name, feedback_type)
                results[client_name] = result
            except Exception as e:
                self.logger.error(f"Error processing {client_name} - {feedback_type}: {str(e)}")
                results[client_name] = None
        
        return results

    def cancel_job(self, job_id: str):
        job = self.client.fine_tuning.jobs.cancel(job_id)
        self.logger.info(f"Job {job_id} status: {job.status}")
        return job
    
    def check_job_status(self, job_id: str):
        job = self.client.fine_tuning.jobs.retrieve(job_id)
        self.logger.info(f"Job {job_id} status: {job.status}")
        
        if job.fine_tuned_model:
            self.logger.info(f"Fine-tuned model: {job.fine_tuned_model}")
        
        return job
    
    def list_recent_jobs(self, limit: int = 10):
        jobs = self.client.fine_tuning.jobs.list(limit=limit)
        
        for job in jobs.data:
            self.logger.info(f"Job ID: {job.id}, Status: {job.status}, Model: {job.fine_tuned_model}")
        
        return jobs
    
    def delete_model(self, model: str):
        response = self.client.models.delete(model=model)
    
        self.logger.info(f"Model: {response.id}, Deleted Status: {response.deleted}")
        
        return response


def create_config_file(): 
    """Create a sample configuration file"""
    config = {
        'openai': {
            'api_key': 'your-openai-api-key-here'
        },
        'clients': {
            'client_a': {
                'name': 'Client A',
                'base_model': 'gpt-4o-mini-2024-07-18',
                'base_model_negative': 'gpt-4o-mini-2024-07-18',
                'base_model_positive': 'gpt-4o-mini-2024-07-18',
                'data_path': 'data/client_a/',
                'output_dir': 'output/client_a/',
                'suffix_template': 'ClientA_{feedback_type}_{date}'
            },
            'client_b': {
                'name': 'Client B',
                'base_model': 'gpt-4o-mini-2024-07-18',
                'base_model_negative': 'gpt-4o-mini-2024-07-18',
                'base_model_positive': 'gpt-4o-mini-2024-07-18',
                'data_path': 'data/client_b/',
                'output_dir': 'output/client_b/',
                'suffix_template': 'ClientB_{feedback_type}_{date}'
            },
            'client_c': {
                'name': 'Client C',
                'base_model': 'gpt-4o-mini-2024-07-18',
                'base_model_negative': 'gpt-4o-mini-2024-07-18',
                'base_model_positive': 'gpt-4o-mini-2024-07-18',
                'data_path': 'data/client_c/',
                'output_dir': 'output/client_c/',
                'suffix_template': 'ClientC_{feedback_type}_{date}'
            }
        },
        'training': {
            'test_size': 0.1,
            'random_state': 42,
            'encoding': 'latin1'
        }
    }
    
    with open('config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    
    print("Configuration file 'config.yaml' created successfully!")


def main():
    """Main function to run the automation"""
    # Create config file if it doesn't exist
    if not os.path.exists('config.yaml'):
        create_config_file()
        print("Please update the config.yaml file with your OpenAI API key and correct paths.")
        return
    
    # Initialize automator
    automator = OpenAIFineTuningAutomator()
    
    # Choose processing method:
    
    # Option 1: Process both positive and negative for all clients
    # print("Starting fine-tuning automation for all clients (both positive and negative)...")
    # results = automator.process_all_clients(feedback_types=["negative", "positive"])
    
    # Option 2: Process only negative feedback for all clients
    print("Starting fine-tuning automation for all clients (negative only)...")
    results = automator.process_single_feedback_type(feedback_type="negative")
    
    # Option 3: Process only positive feedback for all clients
    # print("Starting fine-tuning automation for all clients (positive only)...")
    # results = automator.process_single_feedback_type(feedback_type="positive")
    
    # Print results
    print("\n" + "="*60)
    print("FINE-TUNING RESULTS")
    print("="*60)
    
    for client, client_results in results.items():
        print(f"\n{client.upper()}:")
        
        if isinstance(client_results, dict):
            # Multiple feedback types
            for feedback_type, result in client_results.items():
                if result:
                    print(f"  {feedback_type.upper()}:")
                    print(f"    Job ID: {result['job_id']}")
                    print(f"    Base Model: {result['base_model']}")
                    print(f"    Suffix: {result['suffix']}")
                    print(f"    Training samples: {result['train_samples']}")
                    print(f"    Validation samples: {result['val_samples']}")
                else:
                    print(f"  {feedback_type.upper()}: FAILED")
        else:
            # Single feedback type
            if client_results:
                result = client_results
                print(f"  Job ID: {result['job_id']}")
                print(f"  Base Model: {result['base_model']}")
                print(f"  Suffix: {result['suffix']}")
                print(f"  Training samples: {result['train_samples']}")
                print(f"  Validation samples: {result['val_samples']}")
            else:
                print(f"  FAILED")


if __name__ == "__main__":
    main()