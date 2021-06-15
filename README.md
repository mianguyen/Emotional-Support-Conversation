# Emotional-Support-Conversation
Data and codes for ACL 2021 paper: Towards Emotional Support Dialog Systems


## News
- (2021-04) Collect more conversations with more topics. ESConv now contians 1410 conversations with 10 topic problems.





Interact with Blender Joint Model:


python BlenderEmotionalSupport.py 
Parameter:

 
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
   
