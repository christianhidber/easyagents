using System;
using System.Collections.Generic;
using System.Text;

namespace EasyAgents_Design
{
    public class PlayContext
    {
        int actions;
        int episodes_done;
        GymEnv gym_env;
        int max_steps_per_episode;
        int num_episodes;
        int play_done;
        int rewards;
        int steps_done;
        int steps_done_in_episode;
        int sum_of_rewards;
    }
}