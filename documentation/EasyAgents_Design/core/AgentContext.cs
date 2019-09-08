using EasyAgents_Design;

namespace EasyAgents.core
{
    public class AgentContext
    {
        TrainContext train;
        GymContext gym;
        PyPlotContext pyplot;
        PlayContext play;

        bool is_eval { get; }

        bool is_play { get; }

        bool is_train { get; }
    }
}