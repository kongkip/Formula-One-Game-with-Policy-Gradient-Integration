
# Formula One Game with Policy Gradient Integration

## Description

This project is an implementation of a Formula One-style racing game enhanced with a reinforcement learning agent. The agent uses policy gradient methods to learn how to play the game, making decisions based on the game's state to perform actions like moving left, right, or staying in the same lane. The goal is to avoid obstacles and walls, with the game's difficulty increasing as the car's speed increases.

## Citation
1. The reference paper can be found [here](https://arxiv.org/pdf/1704.06440.pdf)
2. The initial game played manually can be found [here](https://www.sourcecodester.com/python/14694/f1-racer-game-using-pygame-source-code.html)



## Installation

To set up this game, you will need Python and PyTorch. Follow these steps to install the necessary dependencies:

1. **Clone the Repository:**
   ```sh
   git clone [URL to the repository]
   cd [repository name]
   ```

2. **Install PyTorch:**
   Visit [PyTorch's official website](https://pytorch.org/get-started/locally/) and follow the instructions to install PyTorch on your machine.

3. **Install Other Dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Running the Game

To run the game, execute the main script from the command line:

```sh
python main.py
```

## Reinforcement Learning

The game uses a policy gradient approach for the reinforcement learning (RL) agent. The RL agent observes the game state, decides on the best action to take, and learns from the results of those actions. The learning process involves:

- **Policy Network**: A neural network that takes the game state as input and outputs the probabilities of taking each action.
- **State Representation**: The current situation of the game is represented as a vector, including the player's car position, obstacle positions, and car speed.
- **Training Loop**: The agent learns through repeated gameplay, updating the policy network's weights based on the rewards received.

## Contributing

Contributions to this project are welcome. Please follow the standard fork-and-pull request workflow. Make sure to update tests as appropriate and keep style consistent with the existing codebase.

## License

[MIT License](LICENSE)
