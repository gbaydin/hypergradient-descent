#!/usr/bin/env bash

python train.py --cuda --model logreg --method sgd     --save --epochs 10 --alpha_0 0.001 --beta 0.001
python train.py --cuda --model logreg --method sgd_hd  --save --epochs 10 --alpha_0 0.001 --beta 0.001
python train.py --cuda --model logreg --method sgdn    --save --epochs 10 --alpha_0 0.001 --beta 0.001
python train.py --cuda --model logreg --method sgdn_hd --save --epochs 10 --alpha_0 0.001 --beta 0.001
python train.py --cuda --model logreg --method adam    --save --epochs 10 --alpha_0 0.001 --beta 1e-7
python train.py --cuda --model logreg --method adam_hd --save --epochs 10 --alpha_0 0.001 --beta 1e-7

python train.py --cuda --model mlp --method sgd     --save --epochs 100 --alpha_0 0.001 --beta 0.001
python train.py --cuda --model mlp --method sgd_hd  --save --epochs 100 --alpha_0 0.001 --beta 0.001
python train.py --cuda --model mlp --method sgdn    --save --epochs 100 --alpha_0 0.001 --beta 0.001
python train.py --cuda --model mlp --method sgdn_hd --save --epochs 100 --alpha_0 0.001 --beta 0.001
python train.py --cuda --model mlp --method adam    --save --epochs 100 --alpha_0 0.001 --beta 1e-7
python train.py --cuda --model mlp --method adam_hd --save --epochs 100 --alpha_0 0.001 --beta 1e-7

python train.py --cuda --model vgg --method sgd     --save --epochs 100 --alpha_0 0.001 --beta 0.001
python train.py --cuda --model vgg --method sgd_hd  --save --epochs 100 --alpha_0 0.001 --beta 0.001
python train.py --cuda --model vgg --method sgdn    --save --epochs 100 --alpha_0 0.001 --beta 0.001
python train.py --cuda --model vgg --method sgdn_hd --save --epochs 100 --alpha_0 0.001 --beta 0.001
python train.py --cuda --model vgg --method adam    --save --epochs 100 --alpha_0 0.001 --beta 1e-8
python train.py --cuda --model vgg --method adam_hd --save --epochs 100 --alpha_0 0.001 --beta 1e-8
