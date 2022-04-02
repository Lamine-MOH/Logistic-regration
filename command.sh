# split the data to training and testing data with equals yes and no outputs
echo "Prepear the data..."
python "src/prepear_data.py" "data/data.csv"

# training the module
echo "Start the training"
python "src/main.py" new "data/training_data.csv" hypo_level=2 mixing iterations_num=100 saving_rate=10

# test the module
echo "Test the module"
python "src/main.py" test "data/learned module.json" "data/test_data.csv"