# compute bounds under one-word perturbation
python run_bounds.py --data=sst --model model_sst_1 model_sst_2 model_sst_3 --p 1 2 100 --method baf discrete ibp --perturbed_words=1
python run_bounds.py --data=yelp --model model_yelp_1 model_yelp_2 model_yelp_3 --p 1 2 100 --method baf discrete ibp --perturbed_words=1

# compute bounds under two-word perturbation
python run_bounds.py --data=sst --model model_sst_1 model_sst_2 model_sst_3 --p 2 --method baf ibp --perturbed_words=2
python run_bounds.py --data=yelp --model model_yelp_1 model_yelp_2 model_yelp_3 --p 2 --method baf ibp --perturbed_words=2

# compare fully-forward, fully-backward, and backward & forward configurations
python run_bounds.py --data=yelp --model model_yelp_small_1 --p 1 2 100 --method backward baf forward --perturbed_words=1
python run_bounds.py --data=sst --model model_sst_small_1 --p 1 2 100 --method backward baf forward --perturbed_words=1

# for quantitative analysis on SST in identifying words important to prediction
python run_bounds.py --data=sst --model model_sst_1 --p 2 --method baf discrete --perturbed_words=1 --samples=100

# train models with other layer normalization settings
python run_bounds.py --data=yelp --model model_yelp_1_no model_yelp_2_no model_yelp_3_no  model_yelp_1_no --p 1 2 100 --method baf discrete --perturbed_words=1 --suffix=no
python run_bounds.py --data=yelp --model model_yelp_1_standard model_yelp_2_standard model_yelp_3_standard --p 1 2 100 --method baf discrete --perturbed_words=1 --suffix=standard
python run_bounds.py --data=sst --model model_sst_1_no model_sst_2_no  model_sst_3_no --p 1 2 100 --method baf discrete --perturbed_words=1 --suffix=no
python run_bounds.py --data=sst --model model_sst_1_standard model_sst_2_standard model_sst_3_standard --p 1 2 100 --method baf discrete --perturbed_words=1 --suffix=standard
