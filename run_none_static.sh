# activate pip / conda environment first

# Train SimCLR model
python main_none_static.py

# Train linear model and run test
python finetune_none_static.py --dataset=CIFAR10 --model_path=save --epoch_num=100
python finetune_none_static.py --dataset=STL10 --model_path=save --epoch_num=100