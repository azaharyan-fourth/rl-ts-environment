#from agent_factory import AgentFactory
from environment import TSEnvironment
from utils import parse_command_args, get_json_params

def run_process(args):
    target_params = get_json_params(args.target_params)
    labor_params = get_json_params(args.labor_params)

    env = TSEnvironment(args.data_path,
                        args.start_test_period,
                        args.target,
                        args.labor_feature,
                        number_actions=int(args.number_actions),
                        start_action=float(args.start_action),
                        stop_action=float(args.stop_action),
                        target_model_params=target_params,
                        labor_model_params=labor_params,
                        cost_feature=int(args.cost_feature),
                        window=int(args.window_size)
                        )
    env.train_environment_and_evaluate()

    #agent = AgentFactory.get_agent(env, args.model)

    #agent.train(151)

if __name__ == '__main__':
    args = parse_command_args()
    run_process(args)
    