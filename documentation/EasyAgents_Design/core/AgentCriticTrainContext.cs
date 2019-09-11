using System;
using System.Collections.Generic;
using System.Text;
using EasyAgents.core;

namespace EasyAgents_Design
{
    public class AgentCriticTrainContext : EpisodesTrainContext
    {

        int actor_loss;
        int critic_loss;
    }
}