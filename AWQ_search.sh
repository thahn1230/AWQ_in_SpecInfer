python -m awq.entry --model_path models/llama2-7b --w_bit 1 --q_group_size 128 --run_awq --dump_awq awq_cache/llama2-7b-w1-g128.pt
python -m awq.entry --model_path models/llama2-7b --w_bit 2 --q_group_size 128 --run_awq --dump_awq awq_cache/llama2-7b-w2-g128.pt
python -m awq.entry --model_path models/llama2-7b --w_bit 3 --q_group_size 128 --run_awq --dump_awq awq_cache/llama2-7b-w3-g128.pt
python -m awq.entry --model_path models/llama2-7b --w_bit 5 --q_group_size 128 --run_awq --dump_awq awq_cache/llama2-7b-w5-g128.pt
python -m awq.entry --model_path models/llama2-7b --w_bit 6 --q_group_size 128 --run_awq --dump_awq awq_cache/llama2-7b-w6-g128.pt
python -m awq.entry --model_path models/llama2-7b --w_bit 7 --q_group_size 128 --run_awq --dump_awq awq_cache/llama2-7b-w7-g128.pt
python -m awq.entry --model_path models/llama2-7b --w_bit 8 --q_group_size 128 --run_awq --dump_awq awq_cache/llama2-7b-w8-g128.pt