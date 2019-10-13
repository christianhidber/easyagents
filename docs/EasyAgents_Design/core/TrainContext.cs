
namespace EasyAgents.core
{
    public class TrainContext
    {
        int episodes_done_in_iteration;
        int episodes_done_in_training;
        int eval_rewards;
        int eval_steps;
        int iterations_done_in_training;
        int learning_rate;
        int max_steps_in_buffer;
        int max_steps_per_episode;
        int num_episodes_per_eval;
        int num_epochs_per_iteration;
        int reward_discount_gamma;
        int steps_done_in_iteration;
        int steps_done_in_training;
        int training_done;
        int loss;
    }
}