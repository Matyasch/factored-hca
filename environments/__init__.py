from gym.envs.registration import register

register(
    id='Shortcut-v0',
    entry_point='environments.shortcut:Shortcut'
)

register(
    id='DelayedEffect-v0',
    entry_point='environments.delayed_effect:DelayedEffect'
)

register(
    id='TabularPong-v0',
    entry_point='environments.tabular_pong:TabularPong'
)
