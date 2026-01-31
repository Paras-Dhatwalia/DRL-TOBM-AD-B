"""
GNN Models for Dynamic Billboard Allocation

Multi-modal action support (NA/EA/MH), graph encoding with skip connections,
multi-head attention for ad-billboard matching, mode-specific policy heads,
and shared critic network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.nn import GINConv, GATConv
from torch.nn import Sequential, Linear, ReLU, LayerNorm, Dropout
import numpy as np
from typing import Dict, Tuple, Optional, Any, Union, List
import logging

logger = logging.getLogger(__name__)

def validate_observations(observations: Dict[str, torch.Tensor], mode: str, 
                         n_billboards: int, max_ads: int, 
                         node_feat_dim: int, ad_feat_dim: int) -> None:
    """
    Comprehensive validation of observation dictionary to ensure it conforms 
    to expected specifications for the given action mode.
    
    Args:
        observations: Dictionary containing batched observations
        mode: Action mode ('na', 'ea', or 'mh')
        n_billboards: Number of billboards in the environment
        max_ads: Maximum number of ads per batch
        node_feat_dim: Expected node feature dimension
        ad_feat_dim: Expected ad feature dimension
        
    Raises:
        ValueError: If any validation check fails with detailed error message
        
    This function is critical for catching data format issues early that would
    otherwise cause silent failures or cryptic tensor dimension errors during
    training. Each validation provides specific error messages to aid debugging.
    """
    
    required_keys = ['graph_nodes', 'graph_edge_links', 'mask']
    missing_keys = [key for key in required_keys if key not in observations]
    if missing_keys:
        raise ValueError(f"Missing required keys {missing_keys} in observations. "
                        f"Available keys: {list(observations.keys())}")
    
    batch_size = observations['graph_nodes'].shape[0]
    
    expected_node_shape = (batch_size, n_billboards, node_feat_dim)
    actual_node_shape = observations['graph_nodes'].shape
    if actual_node_shape != expected_node_shape:
        raise ValueError(f"Invalid graph_nodes shape. Expected {expected_node_shape}, "
                        f"got {actual_node_shape}. This typically indicates a mismatch "
                        f"between environment configuration and model configuration.")
    
    edge_shape = observations['graph_edge_links'].shape
    if len(edge_shape) != 3:
        raise ValueError(f"graph_edge_links must have 3 dimensions (batch_size, 2, num_edges), "
                        f"got {len(edge_shape)} dimensions with shape {edge_shape}")
    if edge_shape[0] != batch_size:
        raise ValueError(f"graph_edge_links batch dimension {edge_shape[0]} != "
                        f"graph_nodes batch dimension {batch_size}")
    if edge_shape[1] != 2:
        raise ValueError(f"graph_edge_links must have 2 node indices per edge, "
                        f"got {edge_shape[1]}")
    
    if mode == 'na':
        # Node Action mode validation
        if 'current_ad' not in observations:
            raise ValueError(f"NA mode requires 'current_ad' key in observations. "
                           f"Available keys: {list(observations.keys())}")
        
        expected_ad_shape = (batch_size, ad_feat_dim)
        actual_ad_shape = observations['current_ad'].shape
        if actual_ad_shape != expected_ad_shape:
            raise ValueError(f"NA mode current_ad shape mismatch. Expected {expected_ad_shape}, "
                           f"got {actual_ad_shape}")
        
        expected_mask_shape = (batch_size, n_billboards)
        actual_mask_shape = observations['mask'].shape
        if actual_mask_shape != expected_mask_shape:
            raise ValueError(f"NA mode mask shape mismatch. Expected {expected_mask_shape}, "
                           f"got {actual_mask_shape}")
            
    elif mode in ['ea', 'mh']:
        # Edge Action and Multi-Head mode validation
        if 'ad_features' not in observations:
            raise ValueError(f"{mode.upper()} mode requires 'ad_features' key in observations. "
                           f"Available keys: {list(observations.keys())}")
        
        expected_ad_shape = (batch_size, max_ads, ad_feat_dim)
        actual_ad_shape = observations['ad_features'].shape
        if actual_ad_shape != expected_ad_shape:
            raise ValueError(f"{mode.upper()} mode ad_features shape mismatch. "
                           f"Expected {expected_ad_shape}, got {actual_ad_shape}")
        
        if mode == 'ea':
            # Edge Action specific mask validation
            expected_mask_shape = (batch_size, max_ads * n_billboards)
            actual_mask_shape = observations['mask'].shape
            if actual_mask_shape != expected_mask_shape:
                raise ValueError(f"EA mode mask shape mismatch. Expected {expected_mask_shape}, "
                               f"got {actual_mask_shape}. EA mode requires flattened "
                               f"(ad, billboard) pair mask.")
        elif mode == 'mh':
            # Multi-Head specific mask validation
            expected_mask_shape = (batch_size, max_ads, n_billboards)
            actual_mask_shape = observations['mask'].shape
            if actual_mask_shape != expected_mask_shape:
                raise ValueError(f"MH mode mask shape mismatch. Expected {expected_mask_shape}, "
                               f"got {actual_mask_shape}. MH mode requires structured "
                               f"(batch, ads, billboards) mask.")
    else:
        raise ValueError(f"Unknown mode '{mode}'. Supported modes: 'na', 'ea', 'mh'")

def log_input_statistics(observations: Dict[str, torch.Tensor], mode: str) -> None:
    """
    Log detailed statistics about input observations for debugging and monitoring.
    
    This function provides comprehensive logging of input tensor statistics which
    is crucial for:
    1. Detecting data distribution issues
    2. Monitoring for NaN/Inf values that would break training
    3. Understanding mask sparsity which affects action space coverage
    4. Debugging gradient flow issues
    
    Args:
        observations: Dictionary containing batched observations
        mode: Action mode for mode-specific analysis
    """
    
    logger.debug("=== Input Statistics ===")
    
    for key, tensor in observations.items():
        if isinstance(tensor, torch.Tensor):
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            mean_val = tensor.mean().item()
            std_val = tensor.std().item()
            
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()
            
            logger.debug(f"{key}: shape={tensor.shape}, dtype={tensor.dtype}")
            logger.debug(f"  Stats: min={min_val:.4f}, max={max_val:.4f}, "
                        f"mean={mean_val:.4f}, std={std_val:.4f}")
            
            if has_nan or has_inf:
                logger.warning(f"  WARNING: {key} contains NaN={has_nan}, Inf={has_inf}")
    
    mask = observations['mask']
    
    if mode == 'na':
        coverage = mask.float().mean().item()
        per_batch_coverage = mask.float().mean(dim=1)
        min_coverage = per_batch_coverage.min().item()
        max_coverage = per_batch_coverage.max().item()
        
        logger.debug(f"NA mask coverage: {coverage:.2%} of billboards available")
        logger.debug(f"  Per-batch coverage range: {min_coverage:.2%} - {max_coverage:.2%}")
        
        if coverage < 0.1:
            logger.warning("Very low billboard availability may limit learning")
            
    elif mode == 'ea':
        coverage = mask.float().mean().item()
        logger.debug(f"EA mask coverage: {coverage:.2%} of ad-billboard pairs available")
        
        if coverage < 0.01:
            logger.warning("Very low pair availability may cause training instability")
            
    elif mode == 'mh':
        ad_coverage = mask[:, :, 0].float().mean().item()
        bb_coverage = mask[:, 0, :].float().mean().item()
        
        logger.debug(f"MH mask coverage: {ad_coverage:.2%} ads, {bb_coverage:.2%} billboards available")
        
        if ad_coverage < 0.1 or bb_coverage < 0.1:
            logger.warning("Low availability in MH mode may cause sequential selection issues")

def preprocess_observations(observations: Dict[str, torch.Tensor],
                           device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    """Convert observations to tensors. Features are pre-normalized using fixed scaling constants."""
    processed = {}

    for key, value in observations.items():
        if isinstance(value, np.ndarray):
            if key == 'graph_edge_links':
                tensor = torch.from_numpy(value).long()
            elif key == 'mask':
                tensor = torch.from_numpy(value).bool()
            else:
                tensor = torch.from_numpy(value).float()
            processed[key] = tensor
        elif isinstance(value, torch.Tensor):
            processed[key] = value
        else:
            processed[key] = value

        if device is not None and isinstance(processed[key], torch.Tensor):
            processed[key] = processed[key].to(device)

    return processed

class AttentionModule(nn.Module):
    """
    Multi-head attention module for ad-billboard matching with proper parameter management.
    
    This module is critical for learning complex relationships between ads and billboards.
    Key design decisions:
    1. Uses PyTorch's native MultiheadAttention for efficiency and stability
    2. Includes residual connection to prevent gradient vanishing
    3. Layer normalization for training stability
    4. Configurable number of heads for different complexity needs
    
    The attention mechanism allows the model to focus on relevant billboard features
    when making allocation decisions for specific ads, which is crucial for the
    billboard allocation problem where context matters significantly.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True,
            dropout=0.1  # Slight regularization
        )
        
        self.norm = LayerNorm(embed_dim)
        
        logger.debug(f"Initialized AttentionModule: embed_dim={embed_dim}, num_heads={num_heads}")
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multi-head attention with residual connection.
        
        Args:
            query: Query tensor (batch_size, query_len, embed_dim)
            key: Key tensor (batch_size, key_len, embed_dim)  
            value: Value tensor (batch_size, value_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Attended features with residual connection and normalization
        """
        
        attn_out, attn_weights = self.attention(query, key, value, key_padding_mask=mask)
        
        # Apply residual connection and layer normalization
        output = self.norm(query + attn_out)
        
        return output

class GraphEncoder(nn.Module):
    """
    Graph encoder using GIN/GAT layers for billboard network representation.
    
    This encoder is responsible for learning spatial relationships between billboards
    which is crucial for the allocation problem. Key features:
    1. Skip connections preserve information across layers
    2. Supports both GIN and GAT convolution types
    3. Layer normalization and dropout for training stability
    4. Configurable depth for different problem complexities
    
    The graph structure captures important spatial relationships like:
    - Physical proximity between billboards
    - Traffic flow patterns
    - Demographic similarities of billboard locations
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, 
                 conv_type: str = 'gin', dropout: float = 0.1):
        super().__init__()
        
        if n_layers < 1:
            raise ValueError(f"n_layers must be at least 1, got {n_layers}")
        if conv_type not in ['gin', 'gat']:
            raise ValueError(f"conv_type must be 'gin' or 'gat', got '{conv_type}'")
        
        self.n_layers = n_layers
        self.conv_type = conv_type
        self.convs = nn.ModuleList()
        
        curr_dim = input_dim
        for i in range(n_layers):
            if conv_type == 'gin':
                mlp = Sequential(
                    Linear(curr_dim, hidden_dim),
                    LayerNorm(hidden_dim),
                    ReLU(),
                    Dropout(dropout)
                )
                self.convs.append(GINConv(mlp, eps=0.0))
                
            elif conv_type == 'gat':
                self.convs.append(GATConv(
                    in_channels=curr_dim, 
                    out_channels=hidden_dim, 
                    dropout=dropout
                ))
                
            curr_dim = hidden_dim
            
        self.output_dim = input_dim + n_layers * hidden_dim
        
        logger.info(f"Initialized GraphEncoder: {conv_type.upper()}, {n_layers} layers, "
                   f"output_dim={self.output_dim}")
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connections for information preservation.
        
        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Edge connectivity (2, num_edges)
            
        Returns:
            Enhanced node features with skip connections
        """
        
        outputs = [x]
        
        current_x = x
        for i, conv in enumerate(self.convs):
            current_x = conv(current_x, edge_index)
            current_x = F.relu(current_x)
            outputs.append(current_x)
            
        final_output = torch.cat(outputs, dim=1)
        
        return final_output

class BillboardAllocatorGNN(nn.Module):
    """
    Unified Billboard Allocation GNN supporting multiple action modes.

    This model addresses the multi-agent billboard allocation problem with three
    different action modes, each requiring different input/output structures and
    attention mechanisms.

    Architecture features:
    - Shared graph encoder for spatial billboard relationships
    - Mode-specific attention and projection layers
    - Proper parameter management for all learnable components
    - Comprehensive validation and debugging capabilities
    """
    
    def __init__(self, 
                 node_feat_dim: int,
                 ad_feat_dim: int,
                 hidden_dim: int,
                 n_graph_layers: int,
                 mode: str = 'na',
                 n_billboards: int = 100,
                 max_ads: int = 20,
                 conv_type: str = 'gin',
                 use_attention: bool = True,
                 dropout: float = 0.1,
                 min_val: float = -1e8):
        
        super().__init__()
        
        self.mode = mode.lower()
        if self.mode not in ['na', 'ea', 'mh']:
            raise ValueError(f"Unsupported mode: {self.mode}. Must be 'na', 'ea', or 'mh'")
            
        self.n_billboards = n_billboards
        self.max_ads = max_ads
        self.min_val = min_val
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim

        self._validation_done = False

        logger.info(f"Initializing BillboardAllocatorGNN: mode={self.mode}, "
                   f"n_billboards={n_billboards}, max_ads={max_ads}")
        
        self.graph_encoder = GraphEncoder(
            input_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            n_layers=n_graph_layers,
            conv_type=conv_type,
            dropout=dropout
        )
        self.billboard_embed_dim = self.graph_encoder.output_dim

        # INFERENCE-STABLE NORMALIZATION: LayerNorm on latent embeddings (not raw features)
        # This is applied AFTER graph encoding, ensuring sample-independent normalization
        # that works identically for batch_size=1 (inference) and batch_size=64 (training)
        self.billboard_norm = LayerNorm(self.billboard_embed_dim)

        self.ad_encoder = nn.Sequential(
            Linear(ad_feat_dim, hidden_dim),
            ReLU(),
            LayerNorm(hidden_dim),  # LayerNorm AFTER projection to latent space
            Dropout(dropout),
            Linear(hidden_dim, hidden_dim)
        )
        
        # Initialize all attention and projection layers in __init__
        if use_attention:
            self.attention = AttentionModule(hidden_dim, num_heads=4)
            
            # Pre-initialize projection layers for each mode
            if self.mode == 'na':
                # Project billboard features to ad embedding dimension for attention
                self.na_billboard_proj = Linear(self.billboard_embed_dim, hidden_dim)
                
            elif self.mode == 'ea':  
                # Project billboard features to ad embedding dimension for attention
                self.ea_billboard_proj = Linear(self.billboard_embed_dim, hidden_dim)
                
        self._build_mode_specific_layers(hidden_dim, dropout)
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Model initialized: {total_params} total parameters, "
                   f"{trainable_params} trainable")
        
    def _build_mode_specific_layers(self, hidden_dim: int, dropout: float) -> None:
        """
        Build layers specific to each action mode with proper parameter registration.
        
        This method ensures all learnable parameters are created during initialization
        and properly registered with PyTorch's parameter system for optimization.
        """
        
        if self.mode == 'na':
            # Node Action: Score individual billboards for a given ad

            input_dim = self.billboard_embed_dim + hidden_dim
            self.actor_head = nn.Sequential(
                Linear(input_dim, 2 * hidden_dim),
                LayerNorm(2 * hidden_dim),
                ReLU(),
                Dropout(dropout),
                Linear(2 * hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, 1)
            )

        elif self.mode == 'ea':
            # Edge Action: Score ad-billboard pairs directly
            # SEMANTIC FEATURES: Edge features (budget_ratio, influence, is_free)

            # SCALE BALANCE: Project 3d edge features to 16d to prevent scale drowning
            # Raw edge features are 3-dim (0-1 range), while embeddings are 64d+
            # Without projection, the 3 scalars become "noise" in the concatenation
            raw_edge_feat_dim = 3
            projected_edge_feat_dim = 16  # Large enough to carry gradient signal
            self.edge_feat_proj = nn.Sequential(
                Linear(raw_edge_feat_dim, projected_edge_feat_dim),
                ReLU(),
            )

            if self.use_attention:
                pair_dim = hidden_dim + projected_edge_feat_dim  # After attention projection + edge features
            else:
                pair_dim = self.billboard_embed_dim + hidden_dim + projected_edge_feat_dim


            self.pair_scorer = nn.Sequential(
                Linear(pair_dim, hidden_dim),
                LayerNorm(hidden_dim),
                ReLU(),
                Dropout(dropout),
                Linear(hidden_dim, hidden_dim // 2),
                ReLU(),
                Linear(hidden_dim // 2, 1)
            )
            with torch.no_grad():
                self.pair_scorer[-1].bias.fill_(0.0)

        elif self.mode == 'mh':
            # Multi-Head: Sequential ad selection then billboard selection
            
            # Head 1: Ad selection network
            self.ad_head = nn.Sequential(
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                LayerNorm(hidden_dim),
                Dropout(dropout),
                Linear(hidden_dim, hidden_dim // 2),
                ReLU(),
                Linear(hidden_dim // 2, 1)
            )
            
            # Head 2: Billboard selection network (conditioned on chosen ad)

            billboard_input_dim = self.billboard_embed_dim + hidden_dim
            self.billboard_head = nn.Sequential(
                Linear(billboard_input_dim, 2 * hidden_dim),
                LayerNorm(2 * hidden_dim),
                ReLU(),
                Dropout(dropout),
                Linear(2 * hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, 1)
            )

        # Critic network (shared across all modes)
        self.critic = nn.Sequential(
            Linear(self.billboard_embed_dim, 2 * hidden_dim),
            LayerNorm(2 * hidden_dim),
            ReLU(),
            Dropout(dropout),
            Linear(2 * hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, 1)
        )

        logger.debug(f"Built {self.mode.upper()} mode-specific layers")

    # Max samples per super-graph chunk (prevents OOM on dense graphs)
    _GNN_CHUNK_SIZE = 32

    def _batch_gnn_encode(self, nodes: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Batched GNN encoding via super-graph construction.

        Instead of looping over batch samples, constructs a single disconnected
        super-graph where each sample's nodes are offset so they don't interact.
        Mathematically identical to per-sample encoding but eliminates the Python loop.

        For large batches (>_GNN_CHUNK_SIZE), processes in chunks to avoid OOM
        on dense graphs (e.g. 444 NYC billboards with ~100K edges).

        Args:
            nodes: (B, N, F) batched node features
            edge_index: (2, E) shared edge index (same topology for all samples)

        Returns:
            (B, N, output_dim) normalized batched node embeddings
        """
        batch_size, n_nodes, _ = nodes.shape

        # Single-sample path (no super-graph needed)
        if batch_size == 1:
            flat_out = self.graph_encoder(nodes.squeeze(0), edge_index)
            flat_out = self.billboard_norm(flat_out)
            return flat_out.unsqueeze(0)

        # Chunk large batches to prevent OOM on dense graphs
        if batch_size > self._GNN_CHUNK_SIZE:
            chunks = []
            for i in range(0, batch_size, self._GNN_CHUNK_SIZE):
                chunk = self._batch_gnn_encode(nodes[i:i + self._GNN_CHUNK_SIZE], edge_index)
                chunks.append(chunk)
            return torch.cat(chunks, dim=0)

        # 1. Flatten nodes: (B, N, F) -> (B*N, F)
        x_flat = nodes.reshape(batch_size * n_nodes, -1)

        # 2. Build batched edge_index with offsets
        n_edges = edge_index.shape[1]
        edge_batch = edge_index.repeat(1, batch_size)  # (2, E*B)
        offsets = torch.arange(batch_size, device=nodes.device).repeat_interleave(n_edges) * n_nodes
        edge_batch = edge_batch + offsets.unsqueeze(0)

        # 3. Single GNN forward pass on super-graph
        out_flat = self.graph_encoder(x_flat, edge_batch)  # (B*N, D)

        out_flat = self.billboard_norm(out_flat)

        # 4. Reshape back: (B*N, D) -> (B, N, D)
        return out_flat.reshape(batch_size, n_nodes, -1)

    def forward(self, observations: Dict[str, torch.Tensor], 
                state: Optional[torch.Tensor] = None, 
                info: Dict[str, Any] = {}) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with comprehensive validation and proper parameter usage.
        
        This method now uses only pre-initialized parameters and layers, ensuring
        proper gradient flow and optimization.
        """
        device = next(self.parameters()).device

        # Input validation with proper dimension inference
        # PERFORMANCE: Only validate on first forward pass to catch dimension bugs early
        # After first successful validation, skip to avoid CPU overhead on every call
        if not self._validation_done:
            node_feat_dim = observations['graph_nodes'].shape[-1]
            ad_feat_dim = self.ad_encoder[0].in_features
            validate_observations(observations, self.mode, self.n_billboards,
                                self.max_ads, node_feat_dim, ad_feat_dim)
            self._validation_done = True
            logger.info("Input validation passed - skipping future validations for performance")

        # Optional debugging statistics
        if logger.isEnabledFor(logging.DEBUG):
            log_input_statistics(observations, self.mode)
        
        batch_size = observations['graph_nodes'].shape[0]
        
        if info.get('preprocess', True):
            observations = preprocess_observations(observations)

        observations = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k,v in observations.items()}

        # All samples share the same graph topology
        edge_index = observations['graph_edge_links'][0].to(device).long()

        billboard_embeds = self._batch_gnn_encode(
            observations['graph_nodes'].float(), edge_index
        )

        # Clamp extreme values from GNN to prevent NaN propagation downstream.
        # GIN convolutions sum neighbor features across high-degree nodes, which
        # can produce inf/NaN after 3 layers on dense graphs (444 nodes, ~100K edges).
        if torch.isnan(billboard_embeds).any() or torch.isinf(billboard_embeds).any():
            billboard_embeds = torch.nan_to_num(billboard_embeds, nan=0.0, posinf=1e6, neginf=-1e6)

        mask = observations['mask'].bool()
        
        if self.mode == 'na':
            probs, new_state = self._forward_na_fixed(billboard_embeds, observations, mask, batch_size, state, info)
        elif self.mode == 'ea':
            probs, new_state = self._forward_ea_fixed(billboard_embeds, observations, mask, batch_size, state, info)
        elif self.mode == 'mh':
            probs, new_state = self._forward_mh_fixed(billboard_embeds, observations, mask, batch_size, state, info)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        
        # Optional output statistics logging
        if logger.isEnabledFor(logging.DEBUG):
            self._log_output_statistics(probs, mask)
        
        return probs, new_state
    
    def _forward_na_fixed(self, billboard_embeds: torch.Tensor, observations: Dict[str, torch.Tensor],
                         mask: torch.Tensor, batch_size: int, state: Optional[torch.Tensor],
                         info: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Node Action forward pass using pre-initialized parameters.
        """
        
        current_ad = observations['current_ad']  # (batch_size, ad_feat_dim)
        ad_embed = self.ad_encoder(current_ad)  # (batch_size, hidden_dim)

        ad_embed_expanded = ad_embed.unsqueeze(1).expand(-1, self.n_billboards, -1)

        combined_features = torch.cat([billboard_embeds, ad_embed_expanded], dim=-1)
        
        if self.use_attention:
            # Use ad as query, projected billboards as key/value
            ad_query = ad_embed.unsqueeze(1)  # (batch_size, 1, hidden_dim)
            
            billboard_key_value = self.na_billboard_proj(billboard_embeds)
            
            attended_features = self.attention(ad_query, billboard_key_value, billboard_key_value)
            combined_features = torch.cat([billboard_embeds, attended_features.expand(-1, self.n_billboards, -1)], dim=-1)
        
        combined_flat = combined_features.view(-1, combined_features.shape[-1])
        scores = self.actor_head(combined_flat).view(batch_size, self.n_billboards)

        scores[~mask] = self.min_val
        probs = F.softmax(scores, dim=1)
        
        return probs, state
        
    def _forward_ea_fixed(self, billboard_embeds: torch.Tensor, observations: Dict[str, torch.Tensor],
                         mask: torch.Tensor, batch_size: int, state: Optional[torch.Tensor],
                         info: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        SEMANTIC LEARNING: Edge Action forward with edge features.

        Edge features provide explicit semantic signals for ad-billboard matching:
        - budget_ratio: Can the ad afford this billboard?
        - influence_score: Billboard's reach potential
        - is_free: Is the billboard available?

        Uses CHUNKED PROCESSING to prevent OOM during GAE when Tianshou passes
        the entire buffer (2880+ samples) through the model at once.
        """
        # CHUNKED PROCESSING: Process in chunks of 32 to prevent OOM during GAE
        # During GAE, batch_size can be entire buffer (2880+), causing 13GB+ allocation
        # for pair_features tensor: batch × 8880 pairs × hidden_dim
        CHUNK_SIZE = 32

        if batch_size > CHUNK_SIZE:
            # Large batch (GAE phase) - process in chunks and concatenate
            all_scores = []
            for start_idx in range(0, batch_size, CHUNK_SIZE):
                end_idx = min(start_idx + CHUNK_SIZE, batch_size)
                chunk_size = end_idx - start_idx

                # Slice all inputs for this chunk
                chunk_billboard = billboard_embeds[start_idx:end_idx]
                chunk_obs = {
                    'ad_features': observations['ad_features'][start_idx:end_idx],
                    'edge_features': observations.get('edge_features')
                }
                if chunk_obs['edge_features'] is not None:
                    chunk_obs['edge_features'] = chunk_obs['edge_features'][start_idx:end_idx]
                chunk_mask = mask[start_idx:end_idx]

                # Process this chunk
                chunk_scores = self._forward_ea_single_chunk(
                    chunk_billboard, chunk_obs, chunk_mask, chunk_size
                )
                all_scores.append(chunk_scores)

            scores = torch.cat(all_scores, dim=0)
            return scores, state
        else:
            # Small batch (normal inference) - process directly
            scores = self._forward_ea_single_chunk(billboard_embeds, observations, mask, batch_size)
            return scores, state

    def _forward_ea_single_chunk(self, billboard_embeds: torch.Tensor, observations: Dict[str, torch.Tensor],
                                  mask: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Process a single chunk of EA forward pass.

        Extracted from _forward_ea_fixed to enable chunked processing for large batches.
        """
        ad_features = observations['ad_features']  # (batch_size, max_ads, ad_feat_dim)
        ad_embeds = self.ad_encoder(ad_features.view(-1, ad_features.shape[-1]))
        ad_embeds = ad_embeds.view(batch_size, self.max_ads, -1)

        # Edge features for matching
        edge_features = observations.get('edge_features')  # (batch_size, max_ads, n_billboards, 3)
        if edge_features is None:
            # Fallback for backwards compatibility
            edge_features = torch.zeros(
                batch_size, self.max_ads, self.n_billboards, 3,
                device=ad_embeds.device, dtype=ad_embeds.dtype
            )

        ad_expanded = ad_embeds.unsqueeze(2).expand(-1, -1, self.n_billboards, -1)
        billboard_expanded = billboard_embeds.unsqueeze(1).expand(-1, self.max_ads, -1, -1)

        if self.use_attention:
            ad_query = ad_expanded.reshape(-1, self.n_billboards, ad_expanded.shape[-1])
            billboard_kv = billboard_expanded.reshape(-1, self.n_billboards, billboard_expanded.shape[-1])

            billboard_kv_proj = self.ea_billboard_proj(billboard_kv)

            pair_features = self.attention(ad_query, billboard_kv_proj, billboard_kv_proj)
            pair_features = pair_features.reshape(batch_size, self.max_ads, self.n_billboards, -1)
        else:
            # Simple concatenation fallback
            pair_features = torch.cat([ad_expanded, billboard_expanded], dim=-1)

        # Project edge features
        edge_features_proj = self.edge_feat_proj(edge_features)
        pair_features = torch.cat([pair_features, edge_features_proj], dim=-1)

        pair_flat = pair_features.reshape(-1, pair_features.shape[-1])
        scores = self.pair_scorer(pair_flat).reshape(batch_size, self.max_ads * self.n_billboards)

        # Apply mask to logits (invalid actions get very negative logits -> ~0 prob after sigmoid)
        # EA mode uses IndependentBernoulli which applies sigmoid internally
        mask_flat = mask.reshape(batch_size, -1)
        scores[~mask_flat] = self.min_val

        # Return raw logits for IndependentBernoulli distribution
        return scores
        
    def _forward_mh_fixed(self, billboard_embeds: torch.Tensor, observations: Dict[str, torch.Tensor],
                         mask: torch.Tensor, batch_size: int, state: Optional[torch.Tensor],
                         info: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-Head forward pass.

        Returns concatenated logits: [ad_logits, billboard_logits] with shape (batch, max_ads + n_billboards)
        This allows Tianshou to batch them properly. The distribution splits them back.
        """

        ad_features = observations['ad_features']  # (batch_size, max_ads, ad_feat_dim)
        ad_embeds = self.ad_encoder(ad_features.view(-1, ad_features.shape[-1]))
        ad_embeds = ad_embeds.view(batch_size, self.max_ads, -1)

        # Head 1: Select ad - return LOGITS (scores), not probs
        ad_logits = self.ad_head(ad_embeds.view(-1, ad_embeds.shape[-1])).view(batch_size, self.max_ads)

        # Create ad mask: ad is valid if it has ANY available billboard
        ad_mask = mask.any(dim=-1).bool()  # (batch_size, max_ads)
        ad_logits[~ad_mask] = self.min_val

        # When ALL ads are masked, all logits are -1e8 → softmax gives NaN.
        # Replace with 0 (uniform) for both internal sampling AND output logits.
        no_valid_ads = ~ad_mask.any(dim=-1)  # (batch_size,)
        if no_valid_ads.any():
            ad_logits = torch.where(
                no_valid_ads.unsqueeze(-1).expand_as(ad_logits),
                torch.zeros_like(ad_logits),
                ad_logits
            )

        # Sample or get ad selection for conditioning Head 2
        ad_probs = F.softmax(ad_logits, dim=1)

        # Replace NaN probs from any source (all-masked, weight divergence, etc.)
        nan_rows = torch.isnan(ad_probs).any(dim=-1)
        if nan_rows.any():
            uniform = torch.ones(self.max_ads, device=ad_probs.device, dtype=ad_probs.dtype) / self.max_ads
            ad_probs = torch.where(
                nan_rows.unsqueeze(-1).expand_as(ad_probs),
                uniform.unsqueeze(0).expand_as(ad_probs),
                ad_probs
            )

        if self.training and state is not None and 'learn' in info:
            chosen_ads = state[:, 0].long()
        else:
            if self.training:
                chosen_ads = Categorical(ad_probs).sample()
            else:
                chosen_ads = ad_probs.argmax(dim=1)

        # Head 2: Select billboard conditioned on chosen ad
        chosen_ad_embeds = ad_embeds[torch.arange(batch_size), chosen_ads]  # (batch_size, hidden_dim)
        chosen_ad_expanded = chosen_ad_embeds.unsqueeze(1).expand(-1, self.n_billboards, -1)

        combined_features = torch.cat([billboard_embeds, chosen_ad_expanded], dim=-1)

        billboard_logits = self.billboard_head(combined_features.view(-1, combined_features.shape[-1]))
        billboard_logits = billboard_logits.view(batch_size, self.n_billboards)

        billboard_mask = mask[torch.arange(batch_size), chosen_ads].bool()  # (batch_size, n_billboards)
        billboard_logits[~billboard_mask] = self.min_val

        # Replace all-masked or NaN billboard logits with uniform
        no_valid_bbs = ~billboard_mask.any(dim=-1)
        bb_nan = torch.isnan(billboard_logits).any(dim=-1)
        fix_bb = no_valid_bbs | bb_nan
        if fix_bb.any():
            billboard_logits = torch.where(
                fix_bb.unsqueeze(-1).expand_as(billboard_logits),
                torch.zeros_like(billboard_logits),
                billboard_logits
            )

        # Replace any remaining NaN in logits before output
        ad_logits = torch.nan_to_num(ad_logits, nan=0.0)
        billboard_logits = torch.nan_to_num(billboard_logits, nan=0.0)

        # Concatenate logits: (batch, max_ads + n_billboards)
        concatenated_logits = torch.cat([ad_logits, billboard_logits], dim=-1)

        return concatenated_logits, chosen_ads
        
    def critic_forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for critic network.

        Processes samples individually to avoid OOM from super-graph construction
        on dense graphs (e.g. 444 NYC billboards with ~100K edges). The critic
        runs under torch.no_grad() during _compute_returns so per-sample looping
        has negligible overhead compared to the memory cost of batched super-graphs.
        """
        device = next(self.parameters()).device
        observations = preprocess_observations(observations)

        observations = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in observations.items()}

        batch_size = observations['graph_nodes'].shape[0]
        edge_index = observations['graph_edge_links'][0].to(device).long()

        all_pooled = []
        for b in range(batch_size):
            sample_embeds = self.graph_encoder(
                observations['graph_nodes'][b].float(), edge_index
            )
            sample_embeds = self.billboard_norm(sample_embeds)
            all_pooled.append(sample_embeds.mean(dim=0, keepdim=True))

        pooled = torch.cat(all_pooled, dim=0)
        values = self.critic(pooled).squeeze(-1)

        return values
        
    def _log_output_statistics(self, probs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], 
                              mask: torch.Tensor) -> None:
        """
        Log detailed statistics about output probabilities for debugging.
        
        This is crucial for monitoring training progress and detecting issues like:
        - Probability collapse (all mass on single action)
        - Uniform distributions (no learning)
        - Mask violations (probability on invalid actions)
        """
        
        logger.debug("=== Output Statistics ===")
        
        if isinstance(probs, tuple):  # MH mode
            ad_probs, bb_probs = probs
            
            ad_entropy = (-ad_probs * torch.log(ad_probs + 1e-8)).sum(dim=1).mean().item()
            bb_entropy = (-bb_probs * torch.log(bb_probs + 1e-8)).sum(dim=1).mean().item()
            
            logger.debug(f"Ad probabilities: shape={ad_probs.shape}, entropy={ad_entropy:.4f}")
            logger.debug(f"Billboard probabilities: shape={bb_probs.shape}, entropy={bb_entropy:.4f}")
            
            ad_max_prob = ad_probs.max(dim=1)[0].mean().item()
            bb_max_prob = bb_probs.max(dim=1)[0].mean().item()
            
            if ad_max_prob > 0.95 or bb_max_prob > 0.95:
                logger.warning("High probability concentration detected - possible overconfidence")
                
        else:
            # Single output modes (NA, EA)
            entropy = (-probs * torch.log(probs + 1e-8)).sum(dim=1).mean().item()
            max_probs, _ = probs.max(dim=1)
            avg_max_prob = max_probs.mean().item()
            
            logger.debug(f"Action probabilities: shape={probs.shape}, entropy={entropy:.4f}")
            logger.debug(f"Max probability: mean={avg_max_prob:.4f}")
            
            # Check mask adherence
            if isinstance(mask, torch.Tensor):
                mask_usage = (probs * mask.float()).sum(dim=1) / mask.float().sum(dim=1)
                logger.debug(f"Probability mass on valid actions: {mask_usage.mean().item():.4f}")
                
                if mask_usage.min().item() < 0.99:
                    logger.warning("Probability leakage to invalid actions detected!")
            
            # Check for degenerate distributions
            if entropy < 0.1:
                logger.warning("Very low entropy - model may be overconfident")
            elif avg_max_prob < 0.1:
                logger.warning("Very uniform distribution - model may not be learning")
                
    def get_parameter_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive parameter summary for debugging and analysis.
        
        This is crucial for:
        1. Verifying all parameters are properly registered
        2. Monitoring parameter magnitudes for gradient issues
        3. Understanding model complexity
        4. Debugging training problems
        """
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Parameter breakdown by component
        component_params = {}
        for name, module in self.named_modules():
            if len(list(module.parameters())) > 0:
                component_params[name] = sum(p.numel() for p in module.parameters())
        
        # Parameter statistics
        param_stats = {}
        for name, param in self.named_parameters():
            param_stats[name] = {
                'shape': tuple(param.shape),
                'numel': param.numel(),
                'requires_grad': param.requires_grad,
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'abs_max': param.data.abs().max().item()
            }
        
        summary = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 ** 2),  # Assuming float32
            'component_parameters': component_params,
            'parameter_statistics': param_stats,
            'mode': self.mode,
            'architecture': {
                'n_billboards': self.n_billboards,
                'max_ads': self.max_ads,
                'hidden_dim': self.hidden_dim,
                'use_attention': self.use_attention,
                'billboard_embed_dim': self.billboard_embed_dim
            }
        }
        
        return summary
    
    def check_gradient_flow(self) -> Dict[str, Any]:
        """
        Check gradient magnitudes for debugging training issues.
        
        This function is essential for identifying:
        1. Vanishing gradients (gradients too small)
        2. Exploding gradients (gradients too large)
        3. Dead neurons (no gradients)
        4. Parameter/gradient ratio issues
        """
        
        grad_info = {
            'has_gradients': {},
            'gradient_norms': {},
            'parameter_norms': {},
            'grad_param_ratios': {},
            'grad_statistics': {}
        }
        
        total_grad_norm = 0.0
        param_count = 0
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                param_norm = param.norm().item()
                grad_param_ratio = grad_norm / (param_norm + 1e-8)
                
                grad_info['has_gradients'][name] = True
                grad_info['gradient_norms'][name] = grad_norm
                grad_info['parameter_norms'][name] = param_norm
                grad_info['grad_param_ratios'][name] = grad_param_ratio
                
                total_grad_norm += grad_norm ** 2
                param_count += 1
                
                # Flag potential issues
                if grad_norm < 1e-7:
                    logger.warning(f"Very small gradient for {name}: {grad_norm}")
                elif grad_norm > 10.0:
                    logger.warning(f"Large gradient for {name}: {grad_norm}")
                    
            else:
                grad_info['has_gradients'][name] = False
        
        # Overall gradient statistics
        total_grad_norm = (total_grad_norm ** 0.5) if param_count > 0 else 0.0
        grad_info['grad_statistics'] = {
            'total_gradient_norm': total_grad_norm,
            'parameters_with_gradients': param_count,
            'parameters_without_gradients': len(list(self.parameters())) - param_count
        }
        
        return grad_info
