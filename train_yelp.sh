python main.py --dir=model_yelp_1 --data=yelp --num_layers=1 --train;\
python main.py --dir=model_yelp_1_no --data=yelp --num_layers=1 --layer_norm=no --train;\
python main.py --dir=model_yelp_1_standard --data=yelp --num_layers=1 --layer_norm=standard --train

python main.py --dir=model_yelp_2 --data=yelp --num_layers=2 --train;\
python main.py --dir=model_yelp_2_no --data=yelp --num_layers=2 --layer_norm=no --train;\
python main.py --dir=model_yelp_2_standard --data=yelp --num_layers=2 --layer_norm=standard --train

python main.py --dir=model_yelp_3 --data=yelp --num_layers=3 --train;\
python main.py --dir=model_yelp_3_no --data=yelp --num_layers=3 --layer_norm=no --train;\
python main.py --dir=model_yelp_3_standard --data=yelp --num_layers=3 --layer_norm=standard --train

python main.py --dir=model_yelp_small_1 --data=yelp --num_layers=1 --hidden_size=64 --intermediate_size=128 --train
