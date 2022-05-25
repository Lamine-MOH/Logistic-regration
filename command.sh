# split the data to training and testing data with equals yes and no outputs
echo "Prepear the data..."
python "src/prepear_data.py" "data/data.csv"

# training the module
echo "Start the training"
python "src/main.py" new train_data="data/training_data.csv" hypo_level=2 mixing iterations_num=100 saving_rate=10

# training with testing the module
echo "Start the training"
python "src/main.py" new train_data="data/training_data.csv" test_data="data/test_data.csv" hypo_level=2 mixing iterations_num=100 saving_rate=10

# contune the training
echo "Contune the training"
python "src/main.py" contune "result/module.json" train_data="data/training_data.csv" test_data="data/test_data.csv"

# test the module
echo "Test the module"
python "src/main.py" test "result/module.json" test_data="data/test_data.csv"