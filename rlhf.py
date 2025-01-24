import numpy as np
import jax
import jax.numpy as jnp
import optax
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM, FlaxAutoModelForCausalLM
from datasets import load_dataset
import random, collections
from ipdb import set_trace

def inference_test(model_name, dataset_name, inference_cap=None):
    model = FlaxAutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset(dataset_name)

    prompts = dataset['train']['instruction']
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="np")
        #training_algorithm(model, tokenizer, inputs)
        outputs = model(**inputs)
        logits = outputs.logits
        generated_ids = logits.argmax(axis=-1)
        decoded_output = tokenizer.decode(generated_ids[0])

        print("\nPROMPT #" + str(i) + ": \n" + prompt)
        print("\nRESPONSE: ", decoded_output)
        print("\n\n")

        if inference_cap is not None:
            if i >= inference_cap:
                print("INFERENCE CAP REACHED")
                break
        
def dummy_reward(context):
    return jax.random.uniform(jax.random.PRNGKey(0), shape=())

#DM-TODO: try to utilize @jax.jit; requires feedforward pass of LLM, which isn't a pure JAX object... use policy.module?
def REINFORCE(policy, inputs, reward, optimizer, opt_state):
    params = policy.params
    def loss_fn(params): #function needs to be completely differentiable
        #feed-forward pass
        logits = policy(**inputs, params=params).logits
        log_probs = jax.nn.log_softmax(logits, axis=-1)

        # Select token log-probs corresponding to the generated sequence (from GPT)
        generated_tokens = inputs.input_ids[0]
        token_log_probs = jnp.sum(jnp.take_along_axis(log_probs, generated_tokens[..., None], axis=-1)) #DM-TODO: investigate (from GPT)

        #return loss
        return -reward * token_log_probs

    # Compute gradients
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
def train(policy, tokenizer, dataset, reward_model, reward_tokenizer, training_algorithm, optimizer):    
    prompts = dataset['train']['instruction']
    opt_state = optimizer.init(policy.params)
    for i, prompt in enumerate(prompts):
        #get policy response
        inputs = tokenizer(prompt, return_tensors="jax", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        #outputs = policy.generate(input_ids=input_ids, max_length=50, params=params) #runs into memory issues?
        outputs = policy(**inputs).logits.argmax(axis=-1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        #calculate reward
        full_context = f"Prompt: {prompt}\nResponse: {response}" #requires prompt and response
        reward = reward_model(full_context)
        
        #perform optimization step
        training_algorithm(policy, inputs, reward, optimizer, opt_state)


def main():
    print("~~START MAIN~~")

    #identify HF models
    MODEL_NAME = "bigscience/bloom-560m" #"allenai/OLMo-1B" #"allenai/tulu-2-7b"
    REWARD_MODEL_NAME = ""
    DATASET_NAME = "openbmb/UltraFeedback"

    #load models, datasets, and algorithm(s)
    model = FlaxAutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding=True)
    dataset = load_dataset(DATASET_NAME)
    reward_model = dummy_reward
    reward_tokenizer = None
    training_algorithm = REINFORCE
    optimizer = optax.adam(learning_rate=1e-5)


    #run training
    train(model, tokenizer, dataset, reward_model, reward_tokenizer, training_algorithm, optimizer)
    print("~~END MAIN~~")

if __name__ == "__main__":
    pass
    #main()