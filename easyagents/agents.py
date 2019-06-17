
class EasyAgent(object):
    """ Abstract base class for all easy reinforcment learning agents.
    """

    def __init__(   self,
                    gym_env_name,
                    fc_layers ):
        if (fc_layers == None):
            fc_layers = (75,75)
        assert isinstance(gym_env_name,str), "passed gym_env_name not a string."
        assert gym_env_name != "", "gym environment name is empty."
        assert fc_layers != None, "fc_layers not set"
        self.gym_env_name = gym_env_name
        self.fc_layers = fc_layers
        return

    def __str__(self):
        """ yields a human readable representation of the agents/algorithms current configuration
        """
        result = "gym_env_name=" + self.gym_env_name + " fc_layers=" + str(self.fc_layers)
        return result



