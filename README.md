<div id="header" align="center">
  <img src="https://araffin.github.io/slides/rl-tuto-jnrr19/images/RL_illustration.png" width="300"/>
</div>



This ongoing project aims at training a deep RL cryptocurrencies trading agent from scratch. At the current stage it consists of a double DQN trained locally with small
models (simple MLP and simple convnet). The next goals are to analyze thoroughly the agent actions, scale up the model and data used, and try to adapt more complex Deep RL 
methods such as muZero.

The original idea is to see whether successfull trading strategies could be obtained using simple Deep RL methods on raw price chart data. To do this, only the N-last
price candles (of arbitrary timeframes) and corresponding volumes are given to the agent, which only has the choice between 3 actions: 0 -> do nothing, 1 -> go long
2 -> go short. This very simple method is set to evolve along the project. 
