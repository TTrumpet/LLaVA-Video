import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.func import functional_call
from collections import OrderedDict

# --- Helper to get parameter shapes from a model ---
def get_param_shapes(model: nn.Module):
    """Retrieves the shapes of all parameters in a model."""
    return [p.shape for p in model.parameters()]

def get_total_params(model: nn.Module):
    """Calculates the total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())

class CoordinateNet(nn.Module):
    """
    A simple MLP that maps a normalized timestamp t to a 4D bounding box vector.
    This is our f(t) -> bbox function.
    """
    def __init__(self, hidden_dim: int = 256, n_hidden_layers: int = 3):
        super().__init__()
        layers = [
            ('input', nn.Linear(1, hidden_dim)),
            ('input_act', nn.ReLU())
        ]
        for i in range(n_hidden_layers):
            layers.append((f'hidden_{i}', nn.Linear(hidden_dim, hidden_dim)))
            layers.append((f'hidden_{i}_act', nn.ReLU()))
        layers.append(('output', nn.Linear(hidden_dim, 4)))
        # Sigmoid activation to ensure bbox coordinates are between 0 and 1
        # [x_center, y_center, width, height]
        layers.append(('output_act', nn.Sigmoid()))

        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t (torch.Tensor): Tensor of shape (B, N, 1) or (N, 1) representing
                              N normalized timestamps for a batch of B videos.
        Returns:
            torch.Tensor: Predicted bounding boxes of shape (B, N, 4) or (N, 4).
        """
        return self.net(t)

# --- Bounding Box --- #
class HyperNetwork(nn.Module):
    """
    Generates the weights for the CoordinateNet from a single latent vector.
    """
    def __init__(self, latent_dim: int, target_net: nn.Module):
        super().__init__()
        self.latent_dim = latent_dim
        self.target_param_count = get_total_params(target_net)

        # A simple MLP to map the latent vector to the flattened parameters
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.target_param_count)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z (torch.Tensor): The latent vector from the LLM, shape (B, latent_dim).
        Returns:
            torch.Tensor: The flattened parameters for the CoordinateNet, shape (B, target_param_count).
        """
        return self.mlp(z)


class BBoxDecoder(nn.Module):
    """
    The main decoder module. It takes a latent vector representing a tubelet
    and can predict a bounding box for any given timestamp.
    """
    def __init__(self, latent_dim: int, coord_net_hidden_dim: int = 256, coord_net_layers: int = 3):
        super().__init__()
        # The CoordinateNet is a static "scaffold". Its parameters will be generated dynamically.
        self.target_net = CoordinateNet(hidden_dim=coord_net_hidden_dim, n_hidden_layers=coord_net_layers)
        self.target_net_param_shapes = get_param_shapes(self.target_net)

        # The HyperNetwork learns to generate the weights for the target_net
        self.hyper_net = HyperNetwork(latent_dim, self.target_net)

        # Buffer for parameter names, used in functional_call
        self.register_buffer('target_net_param_names',
                             torch.tensor([i for i, (name, _) in enumerate(self.target_net.named_parameters())]))

    def forward(self, z: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Generates bounding boxes for a batch of videos at given timestamps.

        Args:
            z (torch.Tensor): Latent vector from the LLM. Shape: (B, latent_dim).
            timestamps (torch.Tensor): Normalized timestamps. Shape: (B, N, 1).

        Returns:
            torch.Tensor: Predicted bounding boxes. Shape: (B, N, 4).
        """
        batch_size = z.shape[0]

        # 1. Generate the parameters for the CoordinateNet
        flat_params = self.hyper_net(z) # Shape: (B, param_count)

        # 2. Un-flatten the parameters to match the target network's layer shapes
        # We process the entire batch at once for efficiency.
        offset = 0
        params_dict_list = []
        for shape in self.target_net_param_shapes:
            numel = shape.numel()
            # Get parameters for the entire batch for this layer
            layer_params = flat_params[:, offset : offset + numel].view(batch_size, *shape)
            params_dict_list.append(layer_params)
            offset += numel

        # Create a tuple of parameter tensors, one for each layer
        # This is the format expected by `functional_call`
        batched_params = tuple(params_dict_list)

        # 3. Use `vmap` to apply the functional_call over the batch dimension
        # This is the most efficient way to run a different model (with different weights) for each item in a batch.
        # It vectorizes the operation, avoiding a Python for-loop.
        predict_fn = lambda params, t: functional_call(self.target_net, {name: p for name, p in zip(self.target_net.state_dict(), params)}, (t,))
        
        # `in_dims=(0, 0)` means map over the first dimension of both `batched_params` and `timestamps`
        # `out_dims=0` means the output should also be batched on the first dimension
        return torch.vmap(predict_fn, in_dims=(0, 0), out_dims=0)(batched_params, timestamps)

# --- Token Selection --- #
class RLBudgetPolicy(nn.Module):
    """
    A learnable policy that determines the optimal token budget (K).

    This module is a simple MLP that takes a state representation (e.g.,
    a text query embedding) and outputs a probability distribution over a
    pre-defined range of possible K values. It is trained using a
    policy gradient algorithm like REINFORCE.
    """
    def __init__(self, state_dim: int, min_k: int, max_k: int, hidden_dim: int = 128):
        """
        Initializes the policy network.

        Args:
            state_dim (int): The dimension of the input state vector.
            min_k (int): The minimum possible value for K.
            max_k (int): The maximum possible value for K.
            hidden_dim (int): The size of the hidden layer in the policy network.
        """
        super().__init__()
        self.min_k = min_k
        self.max_k = max_k
        self.action_space_size = max_k - min_k + 1

        # A simple two-layer MLP to act as the policy network
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_space_size),
            # We output raw logits; softmax will be applied to create a distribution
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to select an action (K) for each state in the batch.

        Args:
            state (torch.Tensor): The state representation for each item in the batch.
                                  Shape: (batch_size, state_dim).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - k_values (torch.Tensor): The selected K for each item. Shape: (batch_size,).
                - log_probs (torch.Tensor): The log probability of the selected actions,
                                            needed for the policy gradient loss.
                                            Shape: (batch_size,).
        """
        # 1. Get action logits from the policy network
        action_logits = self.network(state)

        # 2. Create a probability distribution over the possible actions
        # The Categorical distribution handles sampling and log_prob calculation
        prob_distribution = Categorical(logits=action_logits)

        # 3. Sample an action (an index in our range of K's) from the distribution
        # This is the "choice" the agent makes
        action_indices = prob_distribution.sample()

        # 4. Calculate the log probability of the chosen actions
        # This is crucial for calculating the REINFORCE loss
        log_probs = prob_distribution.log_prob(action_indices)

        # 5. Convert the action index back to the actual K value
        k_values = action_indices + self.min_k

        return k_values, log_probs


class DifferentiableGatingSelector(nn.Module):
    """
    Scores all visual tokens and selects the top K most relevant ones.
    The "differentiable" aspect comes from the fact that the scoring
    mechanism is a learnable neural network layer.
    """
    def __init__(self, token_dim: int):
        super().__init__()
        # A simple linear layer to map each token to a single scalar score
        self.scorer = nn.Linear(token_dim, 1)

    def forward(self, tokens: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Selects the top K tokens for each item in the batch.

        Args:
            tokens (torch.Tensor): Visual tokens from the MVT encoder.
                                   Shape: (B, num_tokens, token_dim).
            K (torch.Tensor): A 1D tensor of length B, where K[i] is the number
                              of tokens to select for the i-th item in the batch.

        Returns:
            torch.Tensor: The selected top K tokens for each item.
                          Shape: (B, max(K), token_dim). Padded with zeros if K varies.
        """
        batch_size, num_tokens, _ = tokens.shape
        max_k = int(torch.max(K))
        
        safe_k = min(max_k, num_tokens)
        if safe_k == 0:
             # Handle the edge case of zero tokens
             return torch.zeros(batch_size, 0, tokens.shape[-1], device=tokens.device, dtype=tokens.dtype)

        # 1. Calculate a score for each token
        scores = self.scorer(tokens).squeeze(-1)  # Shape: (B, num_tokens)

        # 2. Get the indices of the top K tokens
        # `topk` returns values and indices
        _, topk_indices = torch.topk(scores, k=safe_k, dim=1) # Shape: (B, max_k)

        # 3. Create a mask to handle variable K across the batch
        # This is important because `torch.gather` requires consistent indexing sizes.
        # We will select up to max_k for all, then mask out the extras.
        arange = torch.arange(safe_k, device=tokens.device).expand(batch_size, -1)
        mask = arange < K.unsqueeze(1) # Shape: (B, max_k)

        # 4. Gather the top tokens using the indices
        # We need to expand the indices to match the token dimension
        expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        selected_tokens = torch.gather(tokens, 1, expanded_indices) # Shape: (B, max_k, token_dim)

        # 5. Apply the mask to zero out tokens beyond the specific K for each batch item
        selected_tokens = selected_tokens * mask.unsqueeze(-1)

        return selected_tokens
