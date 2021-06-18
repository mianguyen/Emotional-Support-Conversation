# Emotional-Support-Conversation
Data and codes for ACL 2021 paper: Towards Emotional Support Dialog Systems


## News
- (2021-04) We have collected more conversations with more topics. ESConv now contians 1,410 conversations with 10 topic problems.
- (2021-06) Fix bugs of the implementation in orginal paper and update new results.

- (2021-06) @chujiezheng has integrated the model implementations in his [repo](https://github.com/chujiezheng/UniModel), where he provides a finer tuning and hyper-parameter settings.



# Environment

Dependencies:
| package name | version |
| ---- | ---- |
| transformers | 4.2.2 |
| torch | 1.7.1 |
| python | 3.7 |

**Please use transformers from our repo, because we adapted transformers to our generation task. The main files we changed are:**
* transformer/generation_utils.py
* transformer/generation_logits_process.py
* transformer/models/blenderbot_small




# Interact with Blender Joint Model


1. Change the parameters
 ```python
 class Args():
    def __init__(self):    
        self.output_dir = './blender_strategy/checkpoint-2130'
        self.model_type = 'mymodel'
        self.model_name_or_path = './blender-small'        
        self.config_name = './blender-small'        
        self.tokenizer_name = './blender-small'        
        self.data_path = "./dataset"
        self.train_file_name = "trainWithStrategy_short.tsv"
        self.eval_file_name = "testWithStrategy_short.tsv"
        self.cache_dir = 'cached'
        self.block_size = 512
        self.do_train = False
        self.do_eval = False
        self.generation = True
        self.generate_and_eval = False
        self.evaluate_during_training = True
        self.per_gpu_train_batch_size = 4
        self.per_gpu_eval_batch_size = 1
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-5
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.num_train_epochs = 5
        self.max_steps = -1
        self.warmup_steps = 120
        self.logging_steps = 30
        self.save_steps = 30
        self.save_total_limit = None
        self.eval_all_checkpoints = False
        self.no_cuda = False
        self.overwrite_output_dir = True
        self.overwrite_cache = True
        self.should_continue = False
        self.seed = 42
        self.local_rank = -1
        self.fp16 = False
        self.fp16_opt_level = 'O1'
        self.strategy = True
        self.turn = False
        self.role = False
 ```
2. run:
`python BlenderEmotionalSupport.py `
# Train\Eval\Generate

We integrate codes for training, evaluating, generating, interacting in ONE file. So **just change the parameters** for training, evaluating...

<details>
<summary>parameters for training (click to show code)</summary>

 ```python
 class Args():
    def __init__(self):    
        self.output_dir = './blender_strategy'
        self.model_type = 'mymodel'
        self.model_name_or_path = './blender-small'        
        self.config_name = './blender-small'        
        self.tokenizer_name = './blender-small'        
        self.data_path = "./dataset"
        self.train_file_name = "trainWithStrategy_short.tsv"
        self.eval_file_name = "devWithStrategy_short.tsv"
        self.cache_dir = 'cached'
        self.block_size = 512
        self.do_train = True
        self.do_eval = False
        self.generation = False
        self.generate_and_eval = False
        self.evaluate_during_training = True
        self.per_gpu_train_batch_size = 4
        self.per_gpu_eval_batch_size = 1
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-5
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.num_train_epochs = 5
        self.max_steps = -1
        self.warmup_steps = 120
        self.logging_steps = 30
        self.save_steps = 30
        self.save_total_limit = None
        self.eval_all_checkpoints = False
        self.no_cuda = False
        self.overwrite_output_dir = True
        self.overwrite_cache = True
        self.should_continue = False
        self.seed = 42
        self.local_rank = -1
        self.fp16 = False
        self.fp16_opt_level = 'O1'
        self.strategy = True
        self.turn = False
        self.role = False
 ```

</details>


<details>
<summary>parameters for evaluating (click to show code)</summary>

 ```python
 class Args():
    def __init__(self):    
        self.output_dir = './blender_strategy/checkpoint-2130'
        self.model_type = 'mymodel'
        self.model_name_or_path = './blender-small'        
        self.config_name = './blender-small'        
        self.tokenizer_name = './blender-small'        
        self.data_path = "./dataset"
        self.train_file_name = "trainWithStrategy_short.tsv"
        self.eval_file_name = "devWithStrategy_short.tsv"
        self.cache_dir = 'cached'
        self.block_size = 512
        self.do_train = False
        self.do_eval = True
        self.generation = False
        self.generate_and_eval = False
        self.evaluate_during_training = True
        self.per_gpu_train_batch_size = 4
        self.per_gpu_eval_batch_size = 1
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-5
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.num_train_epochs = 5
        self.max_steps = -1
        self.warmup_steps = 120
        self.logging_steps = 30
        self.save_steps = 30
        self.save_total_limit = None
        self.eval_all_checkpoints = False
        self.no_cuda = False
        self.overwrite_output_dir = True
        self.overwrite_cache = True
        self.should_continue = False
        self.seed = 42
        self.local_rank = -1
        self.fp16 = False
        self.fp16_opt_level = 'O1'
        self.strategy = True
        self.turn = False
        self.role = False
 ```
</details>


<details>
<summary>parameters for generating (click to show code)</summary>

 ```python
 class Args():
    def __init__(self):    
        self.output_dir = './blender_strategy/checkpint-2130'
        self.model_type = 'mymodel'
        self.model_name_or_path = './blender-small'        
        self.config_name = './blender-small'        
        self.tokenizer_name = './blender-small'        
        self.data_path = "./dataset"
        self.train_file_name = "trainWithStrategy_short.tsv"
        self.eval_file_name = "testWithStrategy_short.tsv"
        self.cache_dir = 'cached'
        self.block_size = 512
        self.do_train = False
        self.do_eval = False
        self.generation = True
        self.generate_and_eval = True
        self.evaluate_during_training = True
        self.per_gpu_train_batch_size = 4
        self.per_gpu_eval_batch_size = 1
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-5
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.num_train_epochs = 5
        self.max_steps = -1
        self.warmup_steps = 120
        self.logging_steps = 30
        self.save_steps = 30
        self.save_total_limit = None
        self.eval_all_checkpoints = False
        self.no_cuda = False
        self.overwrite_output_dir = True
        self.overwrite_cache = True
        self.should_continue = False
        self.seed = 42
        self.local_rank = -1
        self.fp16 = False
        self.fp16_opt_level = 'O1'
        self.strategy = True
        self.turn = False
        self.role = False
 ```
 </details>

run:
`python BlenderEmotionalSupport.py `

# Bugs in original implementation

1. We wrongly adopted sentence-level ppl (perplexity) to evaluate generation results on test set, which should have been corpus-level. Commonly sentence-level ppl seems lower than corpus-level. Hense, we update the new ppl results here:

    Blenderbot Joint/Oracle : 18.61 (originally 16.03)

    DialoGPT Joint/Oracle : 19.09 (originally 15.19)



