# activate pip / conda environment first

# Train SimCLR model
python main.py --static=False --model=resnet18

# Train linear model and run test
python -m testing.logistic_regression \
    with \
    model_path=./logs/0 \
    static=False \
    model=resnet18