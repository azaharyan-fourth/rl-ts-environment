from argparse import ArgumentParser, Namespace
import json

def parse_command_args():
    parser = ArgumentParser()

    parser.add_argument("-dp", "--data_path", dest="data_path",required=True,
                        help="specify the path to the data file stored in CSV format")
    parser.add_argument("-m", "--model", dest="model", required=True)
    parser.add_argument("-ws", "--window_size", dest="window_size", default=30)
    parser.add_argument("-t", "--target", dest="target", required=True)
    parser.add_argument("-l", "--labor_feature", dest="labor_feature", required=True)
    parser.add_argument("-na", "--number_actions", dest="number_actions", required=True)
    parser.add_argument("-sa", "--start_action", dest="start_action", required=True)
    parser.add_argument("-st", "--stop_action", dest="stop_action", required=True)
    parser.add_argument("-tp", "--test_period", dest="start_test_period", required=True)
    parser.add_argument("-pt", "--target_params", dest="target_params")
    parser.add_argument("-pl", "--labor_params", dest="labor_params")
    parser.add_argument("-c", "--cost_labor", dest="cost_labor", default=10)

    args = parser.parse_args()

    return args

def get_json_params(file_name):
    if file_name != "":
        f = open(file_name)
        params = json.load(f)

        return params