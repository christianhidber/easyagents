namespace EasyAgents.core
{
    public class DqnTrainContext : TrainContext
    {
        int num_steps_per_iteration;
        int num_steps_initial_bufferload;
        int num_steps_sampled_from_buffer;
        int max_steps_in_buffer;
    }
}