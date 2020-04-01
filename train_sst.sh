python main.py --dir=model_sst_1 --data=sst --num_layers=1 --train;\
python main.py --dir=model_sst_1_no --data=sst --num_layers=1 --layer_norm=no --train;\
python main.py --dir=model_sst_1_standard --data=sst --num_layers=1 --layer_norm=standard --train

python main.py --dir=model_sst_2 --data=sst --num_layers=2 --train;\
python main.py --dir=model_sst_2_no --data=sst --num_layers=2 --layer_norm=no --train;\
python main.py --dir=model_sst_2_standard --data=sst --num_layers=2 --layer_norm=standard --train

python main.py --dir=model_sst_3 --data=sst --num_layers=3 --train;\
python main.py --dir=model_sst_3_no --data=sst --num_layers=3 --layer_norm=no --train;\
python main.py --dir=model_sst_3_standard --data=sst --num_layers=3 --layer_norm=standard --train

python main.py --dir=model_sst_small_1 --data=sst --num_layers=1 --hidden_size=64 --intermediate_size=128 --train