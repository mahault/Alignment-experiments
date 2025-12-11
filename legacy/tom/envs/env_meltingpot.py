import dm_env
import meltingpot

meltingpot.scenario.PERMITTED_OBSERVATIONS = frozenset(
    {
        # The primary visual input.
        "RGB",
        # Extra observations used in some substrates.
        "HUNGER",
        "INVENTORY",
        "MY_OFFER",
        "OFFERS",
        "READY_TO_SHOOT",
        "STAMINA",
        "VOTING",
        # An extra observation that is never necessary but could perhaps help.
        "COLLECTIVE_REWARD",
        # Add world observation so we can render it.
        "WORLD.RGB",
    }
)


class MeltingPotEnv:

    def __init__(self, substrate="clean_up", scenario=None, roles=None):
        if scenario is not None:
            env_config = meltingpot.scenario.get_config(scenario)
            self._env = meltingpot.scenario.build_from_config(config=env_config)
        else:
            env_config = meltingpot.substrate.get_config(substrate)
            if roles is None:
                roles = env_config.default_player_roles

            self._env = meltingpot.substrate.build_from_config(
                config=env_config, roles=roles
            )

        self.num_players = len(self._env.action_spec())
        # assume all players have same action space?!
        self.num_actions = self._env.action_spec()[0].num_values
        self.ts = None

    def as_observation(self, ts: dm_env.TimeStep):
        """
        Convert TimeStep to an observation for each agent that PYMDP can ingest
        """
        # TODO determine what this should look like, for now an array with RGB observations
        return [ts.observation[i]["RGB"] for i in range(self.num_players)]

    def reset(self):
        """
        Reset the environment and return inital observation
        """
        self.ts = self._env.reset()
        return self.as_observation(self.ts)

    def step(self, actions):
        """
        Take a step in the environment

        Args:
            actions: list of actions to take for each agent in the environment
        """
        self.ts = self._env.step(actions)
        return self.as_observation(self.ts)

    def render(self):
        """
        Render the environment
        """
        if self.ts is None:
            raise ValueError("Cannot render before environment is reset")

        return self.ts.observation[0]["WORLD.RGB"]
