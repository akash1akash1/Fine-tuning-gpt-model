import time
import asyncio
import pandas as pd
from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm
import os
from pathlib import Path

# Load API key from environment variable or config
OPENAI_KEY = os.getenv('OPENAI_API_KEY', 'your-openai-api-key-here')

client = AsyncOpenAI(api_key=OPENAI_KEY)

# Example fine-tuned model ID (replace with your actual model ID)
fine_tuned_model_id = 'ft:gpt-4o-mini-2024-07-18:your-org:your-model-name:abc123'
print("\nFine-tuned model id:", fine_tuned_model_id)

from decimal import Decimal, ROUND_HALF_UP
import math

def convert_to_linear_probs(logprobs):
    """Convert log probabilities to linear probabilities and calculate average"""
    # Convert log probabilities to linear probabilities by exponentiating each value
    linear_probs = [math.exp(logprob) for logprob in logprobs]

    # Calculate the average of the linear probabilities
    avg_prob = sum(linear_probs) / len(linear_probs)

    # Convert the average probability to a Decimal and round it to 2 decimal places
    avg_prob_rounded = Decimal(avg_prob).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    # Return the list of linear probabilities and the rounded average probability as a float
    return linear_probs, float(avg_prob_rounded)

def process_api_response(api_response):
    """Process API response to extract logprobs and responses"""
    all_logprobs = []
    custom_responses = []
    avg_probabilities = []

    if not api_response:
        print("Warning: API response is empty.")
        return all_logprobs, custom_responses

    for completion in api_response:
        if not completion.choices:
            print(f"Warning: No choices in completion {completion.id}")
            continue

        for choice in completion.choices:
            if not choice.logprobs or not choice.logprobs.content:
                print(f"Warning: No logprobs content in choice {choice.index} of completion {completion.id}")
                custom_responses.append("")  # Empty string as custom response
                continue

            logprobs = [token.logprob for token in choice.logprobs.content if token.logprob is not None]

            linear_probs, avg_prob = convert_to_linear_probs(logprobs)
            
            if not logprobs:
                print(f"Warning: No valid logprobs in choice {choice.index} of completion {completion.id}")
                custom_responses.append("")  # Empty string as custom response
                continue
            
            all_logprobs.extend(logprobs)
            avg_probabilities.append(avg_prob)
            custom_responses.append(choice.message.content if choice.message else "")

    if not all_logprobs:
        print("Warning: No valid logprobs found in the entire API response.")

    return all_logprobs, custom_responses, avg_probabilities


async def chat_completion_async(client, text, **kwargs):
    """Async chat completion with proper text formatting"""
    text = "\"" + text + "\""

    messages = [{"role": "user", "content": text}]
    return await client.chat.completions.create(
        messages=messages,
        **kwargs
    )

async def run_chat_completions_batch(prompts, **kwargs):
    """Run chat completions in batches to avoid rate limits"""
    async with AsyncOpenAI(api_key=OPENAI_KEY) as client:
        tasks = []
        responses = []
        api_call_batch_size = 10
        progress_bar = tqdm(total=len(prompts), desc="Processing")
        
        for prompt in prompts:
            task = asyncio.create_task(
                chat_completion_async(client, prompt, **kwargs))
            tasks.append(task)
            if len(tasks) >= api_call_batch_size:
                print("Starting batch")
                responses.extend(await asyncio.gather(*tasks))
                tasks = []
                progress_bar.update(api_call_batch_size)
        
        # Process the last batch
        if tasks:
            responses.extend(await asyncio.gather(*tasks))
            progress_bar.update(len(tasks))
        
        progress_bar.close()
        return responses
    
async def predict_async(sentences, model_id):
    """Make predictions using the fine-tuned model"""
    API_RESPONSE = await run_chat_completions_batch(
        sentences, 
        model=model_id, 
        temperature=0,
        max_tokens=20, 
        logprobs=True
    )
    response_text = [
        response.choices[0].message.content for response in API_RESPONSE
    ]
    logprobs, response_text, avg_probabilities = process_api_response(API_RESPONSE)
    merged_data = zip(sentences, response_text, avg_probabilities)
    return merged_data

def format_test_message(row):
    """Format test data for API call"""
    formatted_message = [
        {
            "role": "user",
            "content": row['sentence__pre_process_desc']
        }
    ]
    return formatted_message

def store_predictions(test_df, model_id):
    """Store predictions from the fine-tuned model"""
    sentences = test_df['sentence__pre_process_desc'].to_list()
    
    # Create new event loop and run prediction
    async def run():
        return await predict_async(sentences, model_id)
    
    # Handle different runtime environments
    try:
        import nest_asyncio
        nest_asyncio.apply()
        merged_data = asyncio.run(run())
    except RuntimeError:
        # If nest_asyncio is not available or doesn't work
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        merged_data = loop.run_until_complete(run())
        loop.close()
    
    return merged_data

def run_model_evaluation(test_file_path, model_id, output_file_path):
    """Main function to run model evaluation"""
    # Load test data
    if test_file_path.endswith('.xlsx'):
        test_df = pd.read_excel(test_file_path)
    elif test_file_path.endswith('.csv'):
        test_df = pd.read_csv(test_file_path)
    else:
        raise ValueError("Unsupported file format. Use .xlsx or .csv")
    
    print(f"Test data shape: {test_df.shape}")
    
    # Optional: limit for testing
    # test_df = test_df[:5]
    
    # Get predictions
    merged_data = store_predictions(test_df, model_id)
    
    # Process predictions
    prediction_data = []
    for original_text, prediction, confidence in merged_data:
        prediction_data.append({
            'original_text': original_text,
            'predicted_category': prediction,
            'confidence_score': confidence
        })
    
    # Convert prediction_data to DataFrame
    prediction_df = pd.DataFrame(prediction_data)

    # Merge with original test data
    result_df = test_df.merge(
        prediction_df, 
        left_index=True, 
        right_index=True
    )
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    result_df.to_csv(output_file_path, index=False)
    print(f"Results saved to: {output_file_path}")
    
    return result_df

if __name__ == "__main__":
    # Configuration
    test_file = "data/test_data/client_e_positive_test.xlsx"  # Updated path
    output_file = "results/client_e_positive_results.csv"    # Updated output
    
    # Check if test file exists
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        print("Please update the test_file path to point to your actual test data.")
        exit(1)
    
    # Check if API key is set
    if OPENAI_KEY == 'your-openai-api-key-here':
        print("Please set your OpenAI API key in the OPENAI_KEY environment variable.")
        exit(1)
    
    # Run evaluation
    try:
        results = run_model_evaluation(test_file, fine_tuned_model_id, output_file)
        print(f"Evaluation completed successfully!")
        print(f"Processed {len(results)} samples")
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")