"""
GNN Models for Dynamic Billboard Allocation

Multi-modal action support (NA/EA/MH), graph encoding with skip connections,
multi-head attention for ad-billboard matching, mode-specific policy heads,
and shared critic network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
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
    
    if mode in ['na', 'ea', 'mh']:
        if 'ad_features' not in observations:
            raise ValueError(f"{mode.upper()} mode requires 'ad_features' key in observations. "
                           f"Available keys: {list(observations.keys())}")

        expected_ad_shape = (batch_size, max_ads, ad_feat_dim)
        actual_ad_shape = observations['ad_features'].shape
        if actual_ad_shape != expected_ad_shape:
            raise ValueError(f"{mode.upper()} mode ad_features shape mismatch. "
                           f"Expected {expected_ad_shape}, got {actual_ad_shape}")

        expected_mask_shape = (batch_size, max_ads, n_billboards)
        actual_mask_shape = observations['mask'].shape
        if actual_mask_shape != expected_mask_shape:
            raise ValueError(f"{mode.upper()} mode mask shape mismatch. Expected {expected_mask_shape}, "
                           f"got {actual_mask_shape}")
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
        
        # need_weights=False enables Flash Attention kernel (O(1) memory instead of O(n²))
        # attn_weights was already discarded — pure memory saving, no behavioral change
        attn_out, _ = self.attention(query, key, value, key_padding_mask=mask, need_weights=False)
        
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
                self.convs.append(GINConv(mlp, eps=0.0, aggr='mean'))
                
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
        if self.mode not in ['na', 'ea', 'mh', 'sequential']:
            raise ValueError(f"Unsupported mode: {self.mode}. Must be 'na', 'ea', 'mh', or 'sequential'")
            
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
        self.raw_feat_dim = node_feat_dim  # Store for raw-feature bypass
        self.billboard_embed_dim = self.graph_encoder.output_dim + node_feat_dim  # 394 + 10 = 404

        # INFERENCE-STABLE NORMALIZATION: LayerNorm on GNN output only (before raw bypass concat)
        # Raw features bypass normalization so the agent has direct access to influence/cost/size
        self.billboard_norm = LayerNorm(self.graph_encoder.output_dim)  # 394, NOT 404

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

        elif self.mode == 'sequential':
            # Sequential: Score individual billboards for a SINGLE ad
            # Input: billboard_embed + ad_embed per billboard
            input_dim = self.billboard_embed_dim + hidden_dim
            self.sequential_scorer = nn.Sequential(
                Linear(input_dim, 2 * hidden_dim),
                LayerNorm(2 * hidden_dim),
                ReLU(),
                Dropout(dropout),
                Linear(2 * hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, 1)
            )

        # Critic network (shared across all modes)
        # Input: billboard embeddings (statistics-pooled) + ad embeddings (statistics-pooled)
        # Statistics pooling: [mean, max, std] provides 3x the information vs mean-only,
        # enabling the critic to distinguish states (e.g., high vs low occupancy) (RC3 fix)
        bb_pooled_dim = self.billboard_embed_dim * 3  # 404 * 3 = 1212
        if self.mode == 'sequential':
            ad_pooled_dim = hidden_dim  # single ad, no pooling needed
        else:
            ad_pooled_dim = hidden_dim * 3  # 128 * 3 = 384
        critic_input_dim = bb_pooled_dim + ad_pooled_dim
        self.critic = nn.Sequential(
            Linear(critic_input_dim, 2 * hidden_dim),
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

    def _get_forward_chunk_size(self) -> int:
        """Compute chunk size for the top-level forward pass based on n_billboards and mode.

        Keeps peak GPU memory under ~2GB per chunk:
        - EA, NYC (444): 32 — identical to previous hardcoded CHUNK_SIZE
        - EA, LA (1483): 9
        - NA/MH, NYC: 64 (clamped)
        - NA/MH, LA: 19

        EA is heaviest: creates (chunk × max_ads × n_billboards × dim) all-pairs tensor.
        NA/MH loop per-ad: only (chunk × n_billboards × dim) per iteration — 2× budget.
        """
        # Base calibration: 32 chunks × 444 billboards worked for EA on 16GB A4000
        base_budget = 32 * 444  # ~14K "billboard-samples" fits in ~2GB for EA

        if self.mode == 'ea':
            chunk = base_budget // self.n_billboards
        else:  # na, mh — lighter per-step, can afford 2× more
            chunk = (base_budget * 2) // self.n_billboards

        return max(4, min(chunk, 64))  # clamp to [4, 64]

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
            raw_features = nodes.squeeze(0)  # (N, 10)
            flat_out = self.graph_encoder(raw_features, edge_index)
            flat_out = self.billboard_norm(flat_out)
            # Raw-feature bypass: concat original features after normalization
            flat_out = torch.cat([flat_out, raw_features], dim=-1)  # (N, 394+10)
            return flat_out.unsqueeze(0)

        # Chunk large batches to prevent OOM on dense graphs.
        # Dynamic chunk size: scales inversely with n_billboards (safety net —
        # forward() already feeds small chunks, so this rarely triggers).
        gnn_chunk = max(4, min(32, 32 * 444 // max(n_nodes, 1)))
        if batch_size > gnn_chunk:
            chunks = []
            for i in range(0, batch_size, gnn_chunk):
                chunk = self._batch_gnn_encode(nodes[i:i + gnn_chunk], edge_index)
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

        # Raw-feature bypass: concat original features after normalization
        out_flat = torch.cat([out_flat, x_flat], dim=-1)  # (B*N, 394+10)

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
            if self.mode != 'sequential':
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

        # TOP-LEVEL CHUNKING: Process GNN + actor together per chunk to prevent OOM.
        # billboard_embeds is (batch, n_billboards, 404) — for LA (1483 billboards) with
        # full buffer (11520+ samples), this would be 27.7 GB without chunking.
        chunk_size = self._get_forward_chunk_size()

        if batch_size <= chunk_size:
            # Small batch (normal inference, single env step) — process directly
            probs, new_state = self._forward_unchunked(
                observations, edge_index, batch_size, state, info
            )
        elif self.mode == 'sequential':
            # Sequential mode is unused — fallback to unchunked (no LA support needed)
            probs, new_state = self._forward_unchunked(
                observations, edge_index, batch_size, state, info
            )
        else:
            # Large batch (GAE / PPO update) — chunk entire pipeline
            all_outputs = []
            nodes = observations['graph_nodes'].float()
            mask = observations['mask'].bool()
            ad_features = observations['ad_features']
            edge_features = observations.get('edge_features') if self.mode == 'ea' else None

            use_checkpointing = torch.is_grad_enabled()

            for start in range(0, batch_size, chunk_size):
                end = min(start + chunk_size, batch_size)
                chunk_nodes = nodes[start:end]
                chunk_mask = mask[start:end]
                chunk_ad = ad_features[start:end]
                chunk_edge = edge_features[start:end] if edge_features is not None else None

                if use_checkpointing:
                    # GRADIENT CHECKPOINTING: During PPO learn(), all chunk graphs
                    # accumulate before loss.backward(). Without checkpointing, 14 chunks
                    # × ~640 MB = 8.96 GB of saved activations. With checkpointing,
                    # only the final output per chunk is stored; intermediates are
                    # recomputed during backward. Peak memory ≈ 1 chunk's activations.
                    out = gradient_checkpoint(
                        self._process_single_chunk,
                        chunk_nodes, edge_index, chunk_mask, chunk_ad, chunk_edge,
                        use_reentrant=False
                    )
                else:
                    # No-grad path (GAE): no graph accumulation, chunks freed immediately
                    out = self._process_single_chunk(
                        chunk_nodes, edge_index, chunk_mask, chunk_ad, chunk_edge
                    )

                all_outputs.append(out)

            probs = torch.cat(all_outputs, dim=0)
            new_state = state  # PPO is non-recurrent

        # Optional output statistics logging
        if logger.isEnabledFor(logging.DEBUG):
            self._log_output_statistics(probs, observations['mask'].bool())

        return probs, new_state
        
    def _forward_unchunked(self, observations: Dict[str, torch.Tensor],
                           edge_index: torch.Tensor, batch_size: int,
                           state: Optional[torch.Tensor],
                           info: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Direct (unchunked) forward pass for small batches (inference, single env step)."""
        billboard_embeds = self._batch_gnn_encode(
            observations['graph_nodes'].float(), edge_index
        )

        # Clamp extreme values from GNN to prevent NaN propagation downstream.
        if torch.isnan(billboard_embeds).any() or torch.isinf(billboard_embeds).any():
            billboard_embeds = torch.nan_to_num(billboard_embeds, nan=0.0, posinf=1e6, neginf=-1e6)

        mask = observations['mask'].bool()

        if self.mode == 'na':
            return self._forward_na_fixed(billboard_embeds, observations, mask, batch_size, state, info)
        elif self.mode == 'ea':
            return self._forward_ea_fixed(billboard_embeds, observations, mask, batch_size, state, info)
        elif self.mode == 'mh':
            return self._forward_mh_fixed(billboard_embeds, observations, mask, batch_size, state, info)
        elif self.mode == 'sequential':
            return self._forward_sequential(billboard_embeds, observations, mask, batch_size, state, info)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _process_single_chunk(self, chunk_nodes: torch.Tensor, edge_index: torch.Tensor,
                               chunk_mask: torch.Tensor, chunk_ad: torch.Tensor,
                               chunk_edge: Optional[torch.Tensor]) -> torch.Tensor:
        """Process a single chunk through GNN + actor. Used by gradient_checkpoint().

        All arguments are tensors (required by torch.utils.checkpoint).
        Returns flat logits/scores tensor for this chunk.
        """
        chunk_bs = chunk_nodes.shape[0]

        # GNN encode
        chunk_bb_embeds = self._batch_gnn_encode(chunk_nodes, edge_index)

        # Per-chunk NaN/inf clamp
        if torch.isnan(chunk_bb_embeds).any() or torch.isinf(chunk_bb_embeds).any():
            chunk_bb_embeds = torch.nan_to_num(chunk_bb_embeds, nan=0.0, posinf=1e6, neginf=-1e6)

        # Build chunk observations dict
        chunk_obs = {'ad_features': chunk_ad}
        if chunk_edge is not None:
            chunk_obs['edge_features'] = chunk_edge

        # Actor forward
        if self.mode == 'na':
            out, _ = self._forward_na_fixed(
                chunk_bb_embeds, chunk_obs, chunk_mask, chunk_bs, None, {}
            )
        elif self.mode == 'ea':
            out, _ = self._forward_ea_fixed(
                chunk_bb_embeds, chunk_obs, chunk_mask, chunk_bs, None, {}
            )
        elif self.mode == 'mh':
            out, _ = self._forward_mh_fixed(
                chunk_bb_embeds, chunk_obs, chunk_mask, chunk_bs, None, {}
            )
        else:
            raise ValueError(f"Unsupported mode in chunk: {self.mode}")

        return out

    def _forward_na_fixed(self, billboard_embeds: torch.Tensor, observations: Dict[str, torch.Tensor],
                         mask: torch.Tensor, batch_size: int, state: Optional[torch.Tensor],
                         info: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Full-step Node Action forward pass.

        Loops over all ads, scoring each ad against all billboards independently.
        Returns (batch, max_ads * n_billboards) flat scores for PerAdCategorical.
        """
        ad_features = observations['ad_features']  # (batch, max_ads, ad_feat_dim)

        all_scores = []
        for ad_idx in range(self.max_ads):
            ad = ad_features[:, ad_idx]  # (batch, ad_feat_dim)
            ad_embed = self.ad_encoder(ad)  # (batch, hidden_dim)

            ad_expanded = ad_embed.unsqueeze(1).expand(-1, self.n_billboards, -1)
            combined = torch.cat([billboard_embeds, ad_expanded], dim=-1)

            if self.use_attention:
                ad_query = ad_embed.unsqueeze(1)  # (batch, 1, hidden_dim)
                billboard_kv = self.na_billboard_proj(billboard_embeds)
                attended = self.attention(ad_query, billboard_kv, billboard_kv)
                combined = torch.cat([billboard_embeds, attended.expand(-1, self.n_billboards, -1)], dim=-1)

            scores = self.actor_head(combined.view(-1, combined.shape[-1])).view(batch_size, self.n_billboards)

            # Per-ad billboard mask
            ad_mask = mask[:, ad_idx].bool()
            scores[~ad_mask] = self.min_val

            # Handle ghost ad slots (all-masked rows): deterministic distribution on billboard 0
            # This ensures log_prob=0 and entropy=0 for inactive slots, preventing phantom
            # entropy/log_prob noise that drowns the learning signal (RC6 fix)
            fix_rows = ~ad_mask.any(dim=-1) | torch.isnan(scores).any(dim=-1)
            if fix_rows.any():
                deterministic_logits = torch.full_like(scores, self.min_val)  # -1e8 everywhere
                deterministic_logits[:, 0] = 0.0  # all mass on billboard 0
                scores = torch.where(
                    fix_rows.unsqueeze(-1).expand_as(scores),
                    deterministic_logits,
                    scores
                )

            all_scores.append(scores)

        all_scores = torch.cat(all_scores, dim=-1)  # (batch, max_ads * n_billboards)
        all_scores = torch.nan_to_num(all_scores, nan=0.0)

        return all_scores, state

    def _forward_ea_fixed(self, billboard_embeds: torch.Tensor, observations: Dict[str, torch.Tensor],
                         mask: torch.Tensor, batch_size: int, state: Optional[torch.Tensor],
                         info: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        SEMANTIC LEARNING: Edge Action forward with edge features.

        Edge features provide explicit semantic signals for ad-billboard matching:
        - budget_ratio: Can the ad afford this billboard?
        - influence_score: Billboard's reach potential
        - is_free: Is the billboard available?

        Chunking is now handled by the top-level forward() method, which processes
        GNN + actor together per chunk to prevent OOM on large billboard counts.
        """
        scores = self._forward_ea_single_chunk(billboard_embeds, observations, mask, batch_size)
        return scores, state

    def _forward_ea_single_chunk(self, billboard_embeds: torch.Tensor, observations: Dict[str, torch.Tensor],
                                  mask: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Process a single chunk of EA forward pass.

        PER-AD LOOP: Instead of expanding all (max_ads × n_billboards) pairs at once
        (which creates a massive tensor that OOMs on large billboard counts like LA=1483),
        loop over ads individually — matching NA/MH's memory profile.

        Memory per ad iteration: (batch, n_billboards, pair_dim) ≈ 3 MB for chunk=9, LA.
        vs all-at-once: (batch, 20, n_billboards, pair_dim) ≈ 63 MB — 20× larger.
        """
        ad_features = observations['ad_features']  # (batch_size, max_ads, ad_feat_dim)
        ad_embeds = self.ad_encoder(ad_features.view(-1, ad_features.shape[-1]))
        ad_embeds = ad_embeds.view(batch_size, self.max_ads, -1)

        # Edge features for matching
        edge_features = observations.get('edge_features')  # (batch_size, max_ads, n_billboards, 3)
        if edge_features is None:
            edge_features = torch.zeros(
                batch_size, self.max_ads, self.n_billboards, 3,
                device=ad_embeds.device, dtype=ad_embeds.dtype
            )

        all_scores = []
        for ad_idx in range(self.max_ads):
            ad_embed = ad_embeds[:, ad_idx]  # (batch, hidden_dim)
            ad_expanded = ad_embed.unsqueeze(1).expand(-1, self.n_billboards, -1)  # (batch, N, hidden)

            if self.use_attention:
                ad_query = ad_embed.unsqueeze(1)  # (batch, 1, hidden_dim)
                billboard_kv = self.ea_billboard_proj(billboard_embeds)  # (batch, N, hidden_dim)
                attended = self.attention(ad_query, billboard_kv, billboard_kv)  # (batch, 1, hidden_dim)
                pair_features = attended.expand(-1, self.n_billboards, -1)  # (batch, N, hidden_dim)
            else:
                # Concatenate ad embedding with billboard embeddings per-ad
                pair_features = torch.cat([billboard_embeds, ad_expanded], dim=-1)  # (batch, N, bb_dim+hidden)

            # Project edge features for this ad
            ad_edge_features = edge_features[:, ad_idx]  # (batch, N, 3)
            edge_proj = self.edge_feat_proj(ad_edge_features)  # (batch, N, 16)
            pair_features = torch.cat([pair_features, edge_proj], dim=-1)  # (batch, N, pair_dim+16)

            # Score each billboard for this ad
            scores = self.pair_scorer(pair_features.reshape(-1, pair_features.shape[-1]))
            scores = scores.view(batch_size, self.n_billboards)  # (batch, N)

            # Per-ad billboard mask
            ad_mask = mask[:, ad_idx].bool()  # (batch, N)
            scores[~ad_mask] = self.min_val

            # Handle ghost ad slots: deterministic distribution on billboard 0
            # Ensures log_prob=0, entropy=0 for inactive billboard selections (RC6 fix)
            fix_rows = ~ad_mask.any(dim=-1) | torch.isnan(scores).any(dim=-1)
            if fix_rows.any():
                deterministic_logits = torch.full_like(scores, self.min_val)
                deterministic_logits[:, 0] = 0.0
                scores = torch.where(
                    fix_rows.unsqueeze(-1).expand_as(scores),
                    deterministic_logits,
                    scores
                )

            all_scores.append(scores)

        all_scores = torch.cat(all_scores, dim=-1)  # (batch, max_ads * n_billboards)
        all_scores = torch.nan_to_num(all_scores, nan=0.0)

        return all_scores

    def _forward_mh_fixed(self, billboard_embeds: torch.Tensor, observations: Dict[str, torch.Tensor],
                         mask: torch.Tensor, batch_size: int, state: Optional[torch.Tensor],
                         info: Dict[str, Any]) -> Tuple[torch.Tensor, None]:
        """
        Autoregressive Multi-Head forward pass.

        Chunking is now handled by the top-level forward() method, which processes
        GNN + actor together per chunk to prevent OOM on large billboard counts.

        Returns concatenated logits: [ad_logits, all_bb_logits_flat]
        Shape: (batch, max_ads + max_ads * n_billboards)
        """
        return self._forward_mh_single_chunk(
            billboard_embeds, observations, mask, batch_size
        ), None

    def _forward_mh_single_chunk(self, billboard_embeds: torch.Tensor,
                                  observations: Dict[str, torch.Tensor],
                                  mask: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Process a single chunk of MH forward pass.

        Extracted from _forward_mh_fixed to enable chunked processing for large batches.
        """
        ad_features = observations['ad_features']  # (batch_size, max_ads, ad_feat_dim)
        ad_embeds = self.ad_encoder(ad_features.view(-1, ad_features.shape[-1]))
        ad_embeds = ad_embeds.view(batch_size, self.max_ads, -1)

        # Head 1: Ad selection logits
        ad_logits = self.ad_head(ad_embeds.view(-1, ad_embeds.shape[-1])).view(batch_size, self.max_ads)

        ad_mask = mask.any(dim=-1).bool()  # ad is valid if ANY billboard is available
        ad_logits[~ad_mask] = self.min_val

        # When ALL ads are masked, use deterministic (all mass on ad 0) to prevent NaN
        # and ensure log_prob=0, entropy=0 for this degenerate case (RC6 fix)
        no_valid_ads = ~ad_mask.any(dim=-1)
        if no_valid_ads.any():
            deterministic_ad_logits = torch.full_like(ad_logits, self.min_val)
            deterministic_ad_logits[:, 0] = 0.0
            ad_logits = torch.where(
                no_valid_ads.unsqueeze(-1).expand_as(ad_logits),
                deterministic_ad_logits,
                ad_logits
            )

        # Head 2: Billboard logits for EVERY ad (autoregressive)
        all_bb_logits = []
        for ad_idx in range(self.max_ads):
            ad_embed = ad_embeds[:, ad_idx]  # (batch, hidden)
            ad_expanded = ad_embed.unsqueeze(1).expand(-1, self.n_billboards, -1)
            combined = torch.cat([billboard_embeds, ad_expanded], dim=-1)

            bb_logits = self.billboard_head(combined.view(-1, combined.shape[-1]))
            bb_logits = bb_logits.view(batch_size, self.n_billboards)

            # Per-ad billboard mask
            bb_mask = mask[:, ad_idx].bool()
            bb_logits[~bb_mask] = self.min_val

            # Handle ghost ad slots: deterministic distribution on billboard 0
            # Ensures log_prob=0, entropy=0 for inactive billboard selections (RC6 fix)
            fix_rows = ~bb_mask.any(dim=-1) | torch.isnan(bb_logits).any(dim=-1)
            if fix_rows.any():
                deterministic_logits = torch.full_like(bb_logits, self.min_val)
                deterministic_logits[:, 0] = 0.0
                bb_logits = torch.where(
                    fix_rows.unsqueeze(-1).expand_as(bb_logits),
                    deterministic_logits,
                    bb_logits
                )

            all_bb_logits.append(bb_logits)

        all_bb_logits = torch.stack(all_bb_logits, dim=1)  # (batch, max_ads, n_billboards)

        # NaN safety
        ad_logits = torch.nan_to_num(ad_logits, nan=0.0)
        all_bb_logits = torch.nan_to_num(all_bb_logits, nan=0.0)

        # Concatenate: [ad_logits, all_bb_logits_flat]
        concatenated = torch.cat([ad_logits, all_bb_logits.view(batch_size, -1)], dim=-1)

        return concatenated

    def _forward_sequential(self, billboard_embeds: torch.Tensor, observations: Dict[str, torch.Tensor],
                            mask: torch.Tensor, batch_size: int, state: Optional[torch.Tensor],
                            info: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sequential mode: Score all billboards for a SINGLE ad.

        Input observations have:
          - ad_features: (batch, ad_feat_dim) — single ad, NOT (batch, max_ads, ad_feat_dim)
          - mask: (batch, n_billboards) — 1D mask, NOT (batch, max_ads, n_billboards)

        Returns: (batch, n_billboards) logits for standard Categorical distribution.
        """
        ad_features = observations['ad_features']  # (batch, ad_feat_dim)

        # Handle case where ad_features might have an extra dim
        if ad_features.dim() == 3:
            ad_features = ad_features.squeeze(1)  # (batch, 1, feat) -> (batch, feat)

        ad_embed = self.ad_encoder(ad_features)  # (batch, hidden_dim)

        # Expand ad embedding to match billboard count
        ad_expanded = ad_embed.unsqueeze(1).expand(-1, self.n_billboards, -1)  # (batch, N, hidden)

        # Concatenate billboard embeddings with ad embedding
        combined = torch.cat([billboard_embeds, ad_expanded], dim=-1)  # (batch, N, embed+hidden)

        # Score each billboard
        scores = self.sequential_scorer(
            combined.view(-1, combined.shape[-1])
        ).view(batch_size, self.n_billboards)  # (batch, N)

        # Apply mask — mask is (batch, n_billboards) for sequential mode
        if mask.dim() == 2:
            scores[~mask] = self.min_val
        elif mask.dim() == 3:
            # Fallback: if mask is still 3D, use first ad's mask
            scores[~mask[:, 0]] = self.min_val

        # Handle all-masked rows
        no_valid = ~mask.any(dim=-1) if mask.dim() == 2 else ~mask[:, 0].any(dim=-1)
        if no_valid.any():
            scores = torch.where(
                no_valid.unsqueeze(-1).expand_as(scores),
                torch.zeros_like(scores),
                scores
            )

        scores = torch.nan_to_num(scores, nan=0.0)
        return scores, state

    def critic_forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for critic network.

        Encodes both billboard state (via GNN) and ad state (via ad_encoder)
        to produce value estimates. Both signals are critical:
        - Billboard state: occupancy, cost, influence, location
        - Ad state: TTL urgency, budget remaining, demand progress

        Processes GNN samples individually to avoid OOM from super-graph
        construction on dense graphs (444 NYC billboards, ~100K edges).
        """
        device = next(self.parameters()).device
        observations = preprocess_observations(observations)

        observations = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in observations.items()}

        batch_size = observations['graph_nodes'].shape[0]
        edge_index = observations['graph_edge_links'][0].to(device).long()

        # Billboard encoding (per-sample GNN to avoid OOM)
        # Statistics pooling: [mean, max, std] preserves spatial/occupancy information
        # that mean-only pooling destroys (RC3 fix)
        all_billboard_pooled = []
        for b in range(batch_size):
            raw_features_b = observations['graph_nodes'][b].float()
            sample_embeds = self.graph_encoder(raw_features_b, edge_index)
            sample_embeds = self.billboard_norm(sample_embeds)
            # Raw-feature bypass: concat original features after normalization
            sample_embeds = torch.cat([sample_embeds, raw_features_b], dim=-1)
            # Statistics pooling: [mean, max, std] — captures distribution of billboard states
            bb_mean = sample_embeds.mean(dim=0)        # (billboard_embed_dim,)
            bb_max = sample_embeds.max(dim=0).values   # (billboard_embed_dim,)
            bb_std = sample_embeds.std(dim=0)          # (billboard_embed_dim,)
            bb_pooled = torch.cat([bb_mean, bb_max, bb_std], dim=-1)  # (billboard_embed_dim * 3,)
            all_billboard_pooled.append(bb_pooled.unsqueeze(0))

        billboard_pooled = torch.cat(all_billboard_pooled, dim=0)  # (batch, billboard_embed_dim * 3)

        # Ad encoding (shared ad_encoder)
        ad_features = observations['ad_features'].float()

        if self.mode == 'sequential':
            # Sequential mode: ad_features is (batch, ad_feat_dim) — single ad
            if ad_features.dim() == 3:
                ad_features = ad_features.squeeze(1)
            ad_pooled = self.ad_encoder(ad_features)  # (batch, hidden_dim)
        else:
            # Other modes: ad_features is (batch, max_ads, ad_feat_dim)
            ad_flat = ad_features.view(-1, ad_features.shape[-1])  # (batch*max_ads, ad_feat_dim)
            ad_embeds = self.ad_encoder(ad_flat)  # (batch*max_ads, hidden_dim)
            ad_embeds = ad_embeds.view(batch_size, self.max_ads, -1)  # (batch, max_ads, hidden_dim)
            # Statistics pooling: [mean, max, std] preserves urgency/priority information (RC3 fix)
            ad_mean = ad_embeds.mean(dim=1)        # (batch, hidden_dim)
            ad_max = ad_embeds.max(dim=1).values   # (batch, hidden_dim)
            ad_std = ad_embeds.std(dim=1)          # (batch, hidden_dim)
            ad_pooled = torch.cat([ad_mean, ad_max, ad_std], dim=-1)  # (batch, hidden_dim * 3)

        # Concatenate billboard + ad state for value prediction
        state_repr = torch.cat([billboard_pooled, ad_pooled], dim=-1)

        values = self.critic(state_repr).squeeze(-1)

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
