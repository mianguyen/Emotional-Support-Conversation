CUDA_VISIBLE_DEVICES=0 python interact.py \
    --config_name strat \
    --inputter_name strat \
    --seed 3 \
    --load_checkpoint ./DATA/strat.strat/2021-10-31191442.3e-05.16.1gpu/epoch-1.bin \
    --fp16 false \
    --max_length 50 \
    --min_length 10 \
    --temperature 0.7 \
    --top_k 0 \
    --top_p 0.9 \
    --num_beams 1 \
    --repetition_penalty 1 \
    --no_repeat_ngram_size 3\
    --strategy_path ./_reformat/strategy.json


    # --max_src_len 150 \
    # --max_tgt_len 50 \