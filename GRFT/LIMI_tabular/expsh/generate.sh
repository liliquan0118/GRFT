cd ..
root='./exp/'

# CUDA_VISIBLE_DEVICES=0 python generate_data.py --load_path=$root'gans/census/census_gan.pth' \
#  --exp_name='census'  \
#  --num_samples=1000000 --exp_flag='orig'

CUDA_VISIBLE_DEVICES=3 python generate_data.py --load_path=$root'gans/credit/credit_gan.pth' \
 --exp_name='credit'  \
 --num_samples=1000000 --exp_flag='orig'

CUDA_VISIBLE_DEVICES=3 python generate_data.py --load_path=$root'gans/bank/bank_gan.pth' \
 --exp_name='bank'  \
 --num_samples=1000000 --exp_flag='orig'
 CUDA_VISIBLE_DEVICES=3 python generate_data.py --load_path=$root'gans/compas/compas_gan.pth' \
 --exp_name='compas'  \
 --num_samples=1000000 --exp_flag='orig'

CUDA_VISIBLE_DEVICES=3 python generate_data.py --load_path=$root'gans/lsac/lsac_gan.pth' \
 --exp_name='lsac'  \
 --num_samples=100000 --exp_flag='orig'

# CUDA_VISIBLE_DEVICES=3 python generate_data.py --load_path=$root'gans/meps/meps_gan.pth' \
#  --exp_name='meps'  \
#  --num_samples=1000000 --exp_flag='orig'