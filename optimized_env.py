# OPTIMIZED DYNABILLBOARD ENVIRONMENT
from __future__ import annotations
import math
import random
import logging
import time
from functools import wraps
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Set
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#  CONFIGURATION 
@dataclass
class EnvConfig:
    """Environment configuration parameters"""
    influence_radius_meters: float = 100.0
    slot_duration_range: Tuple[int, int] = (3, 7)  # Increased from (1,5): longer assignments capture more influence
    new_ads_per_step_range: Tuple[int, int] = (0, 3)  # Increased from (0,2) for more ad flow
    tardiness_cost: float = 50.0
    max_events: int = 1440  # Full day (1 minute per step, 1440 minutes = 24 hours)
    max_active_ads: int = 8  # Reduced from 20 to allow concentration (50 allocs / 8 ads = 6+ billboards each)
    ad_ttl: int = 720  # Ad time-to-live in timesteps (increased from 600 for more accumulation time)
    graph_connection_distance: float = 5000.0
    cache_ttl: int = 1  # Cache TTL in steps
    enable_profiling: bool = False
    debug: bool = False

#  PERFORMANCE MONITORING 
class PerformanceMonitor:
    """Track performance metrics and timing"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.step_times = []
        self.influence_calc_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.step_count = 0
    
    def time_function(self, category='general'):
        """Decorator to time function execution"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000  # ms
                
                if category == 'step':
                    self.step_times.append(elapsed)
                elif category == 'influence':
                    self.influence_calc_times.append(elapsed)
                
                if self.step_count % 100 == 0 and self.step_count > 0:
                    self.print_stats()
                    
                return result
            return wrapper
        return decorator
    
    def record_cache_hit(self):
        self.cache_hits += 1
    
    def record_cache_miss(self):
        self.cache_misses += 1
    
    def print_stats(self):
        """Print performance statistics"""
        if self.step_times:
            avg_step = np.mean(self.step_times)
            logger.info(f"Avg step time: {avg_step:.2f}ms")
        
        if self.influence_calc_times:
            avg_influence = np.mean(self.influence_calc_times)
            logger.info(f"Avg influence calc time: {avg_influence:.2f}ms")
        
        if self.cache_hits + self.cache_misses > 0:
            hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses)
            logger.info(f"Cache hit rate: {hit_rate:.2%}")

#  HELPER FUNCTIONS 
def time_str_to_minutes(v: Any) -> int:
    """Convert time string to minutes since midnight."""
    if isinstance(v, str) and ":" in v:
        try:
            hh, mm = v.split(":")[:2]
            return int(hh) * 60 + int(mm)
        except Exception as e:
            logger.warning(f"Could not parse time string {v}: {e}")
            return 0
    try:
        return int(v)
    except Exception:
        return 0

def haversine_distance_vectorized(lat1: np.ndarray, lon1: np.ndarray, 
                                  lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized haversine distance calculation between points in meters."""
    R = 6371000.0  # Earth radius in meters
    
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

def validate_csv(df: pd.DataFrame, required_columns: List[str], csv_name: str):
    """Validate that CSV has required columns"""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_name} missing required columns: {missing}")

#  DATA CLASSES 
class Ad:
    """Represents an advertisement with demand and payment attributes."""

    def __init__(self, aid: int, demand: float, payment: float,
                 payment_demand_ratio: float, ttl: int = 15):
        self.aid = aid
        self.demand = float(demand)
        self.payment = float(payment)  # Total budget available
        self.remaining_budget = float(payment)  # BUDGET TRACKING: Money left to spend
        self.total_cost_spent = 0.0  # BUDGET TRACKING: Total spent on billboards
        self.payment_demand_ratio = float(payment_demand_ratio)
        self.ttl = ttl
        self.original_ttl = ttl
        self.state = 0  # 0: ongoing, 1: finished, 2: tardy/expired, 3: budget exhausted
        self.assigned_billboards: Set[int] = set()  # Use set for O(1) operations
        self.time_active = 0
        self.cumulative_influence = 0.0
        self.spawn_step: Optional[int] = None
        self._cached_influence: Optional[float] = None
        self._cache_step: Optional[int] = None

    def step_time(self):
        """Tick TTL and mark tardy if TTL expires while still ongoing."""
        if self.state == 0:
            self.time_active += 1
            self.ttl -= 1
            if self.ttl <= 0:
                self.state = 2  # tardy / failed

    def assign_billboard(self, b_id: int, billboard_cost: float) -> bool:
        """
        Assign a billboard to this ad if budget allows.

        Returns:
            True if assignment successful, False if can't afford
        """
        if self.remaining_budget >= billboard_cost:
            self.assigned_billboards.add(b_id)
            self.remaining_budget -= billboard_cost  # BUDGET TRACKING: Deduct cost
            self.total_cost_spent += billboard_cost  # BUDGET TRACKING: Track spending
            self._cached_influence = None  # Invalidate cache
            return True
        else:
            return False  # Can't afford this billboard

    def release_billboard(self, b_id: int):
        """Release a billboard from this ad."""
        self.assigned_billboards.discard(b_id)
        self._cached_influence = None  # Invalidate cache

    def norm_payment_ratio(self) -> float:
        """Normalized payment ratio using sigmoid function."""
        return 1.0 / (1.0 + math.exp(-(self.payment_demand_ratio - 1.0)))

    # FIXED SCALING CONSTANTS for inference-stable normalization
    # These ensure all features are in [0, 1] regardless of batch size
    # Values from Advertiser_100.csv: Demand=100-149, Payment=90k-161k, Ratio=902-1099
    MAX_DEMAND = 200.0         # Demand range: 100-149, buffer to 200
    MAX_PAYMENT = 200000.0     # Payment range: 90k-161k, buffer to 200k
    MAX_RATIO = 1200.0         # Ratio range: 902-1099, buffer to 1200
    MAX_BILLBOARDS = 25.0      # Max billboards per ad

    def get_feature_vector(self) -> np.ndarray:
        """Get feature vector for this ad (12 features, all normalized to [0, 1]).

        INFERENCE-STABLE: Uses fixed scaling constants instead of batch statistics.
        This ensures identical normalization for batch_size=1 (inference) and
        batch_size=64 (training), fixing the "inference blindness" bug.
        """
        return np.array([
            min(self.demand / self.MAX_DEMAND, 1.0),              # [0, 1]
            min(self.payment / self.MAX_PAYMENT, 1.0),            # [0, 1]
            min(self.payment_demand_ratio / self.MAX_RATIO, 1.0), # [0, 1]
            self.norm_payment_ratio(),                             # Already sigmoid → [0, 1]
            self.ttl / max(1, self.original_ttl),                  # [0, 1]
            min(self.cumulative_influence / max(self.demand, 1e-6), 1.0),  # [0, 1]
            min(len(self.assigned_billboards) / self.MAX_BILLBOARDS, 1.0), # [0, 1]
            1.0 if self.state == 0 else 0.0,                       # Boolean [0, 1]
            self.remaining_budget / max(self.payment, 1e-6),       # [0, 1]
            self.total_cost_spent / max(self.payment, 1e-6),       # [0, 1]
            (self.demand - self.cumulative_influence) / max(self.demand, 1e-6),  # [0, 1]
            min((self.spawn_step or 0) / 1000.0, 1.0),             # [0, 1]
        ], dtype=np.float32)


class Billboard:
    """Represents a billboard with location and properties."""
    
    def __init__(self, b_id: int, lat: float, lon: float, tags: str, 
                 b_size: float, b_cost: float, influence: float):
        self.b_id = b_id
        self.latitude = float(lat)
        self.longitude = float(lon)
        self.tags = tags if pd.notna(tags) else ""
        self.b_size = float(b_size)
        self.b_cost = float(b_cost)
        self.influence = float(influence)
        self.occupied_until = 0
        self.current_ad: Optional[int] = None
        self.p_size = 0.0  # normalized size
        self.total_usage = 0
        self.revenue_generated = 0.0

    def is_free(self) -> bool:
        """Check if billboard is available."""
        return self.occupied_until <= 0

    def assign(self, ad_id: int, duration: int):
        """Assign an ad to this billboard for a duration."""
        self.current_ad = ad_id
        self.occupied_until = max(1, int(duration))
        self.total_usage += 1

    def release(self) -> Optional[int]:
        """Release current ad from billboard."""
        ad_id = self.current_ad
        self.current_ad = None
        self.occupied_until = 0
        return ad_id

    # FIXED SCALING CONSTANTS for inference-stable normalization
    # These ensure all features are in [0, 1] regardless of batch size
    # Values from BB_NYC.csv: B_Size=80-670, B_Cost=0.25-104.5, Influence=0.4-149.8
    MAX_COST = 150.0         # Cost range: 0.25-104.5, buffer to 150
    MAX_SIZE = 800.0         # Size range: 80-670, buffer to 800
    MAX_INFLUENCE = 200.0    # Influence range: 0.4-149.8, buffer to 200
    MAX_DURATION = 10.0      # Max occupation duration
    MAX_USAGE = 100.0        # Max total usage count

    def get_feature_vector(self) -> np.ndarray:
        """Get feature vector for this billboard (10 features, all normalized to [0, 1]).

        INFERENCE-STABLE: Uses fixed scaling constants instead of batch statistics.
        This ensures identical normalization for batch_size=1 (inference) and
        batch_size=64 (training), fixing the "inference blindness" bug.
        """
        return np.array([
            1.0,  # node type (billboard)
            0.0 if self.is_free() else 1.0,  # is_occupied [0, 1]
            min(self.b_cost / self.MAX_COST, 1.0),      # [0, 1]
            min(self.b_size / self.MAX_SIZE, 1.0),      # [0, 1]
            min(self.influence / self.MAX_INFLUENCE, 1.0),  # [0, 1] - was raw value!
            self.p_size,                                 # Already normalized [0, 1]
            min(self.occupied_until / self.MAX_DURATION, 1.0),  # [0, 1]
            min(self.total_usage / self.MAX_USAGE, 1.0),        # [0, 1]
            (self.latitude + 90.0) / 180.0,              # [0, 1]
            (self.longitude + 180.0) / 360.0,            # [0, 1]
        ], dtype=np.float32)


#  OPTIMIZED ENVIRONMENT 
class OptimizedBillboardEnv(gym.Env):
    """
    FIXED: Changed from AECEnv to gym.Env (single-agent, synchronous).

    Optimized Dynamic Billboard Allocation Environment with vectorized operations.

    Key optimizations:
    - Vectorized influence calculations using NumPy broadcasting
    - Cached per-minute billboard probabilities
    - Precomputed billboard size ratios
    - Efficient trajectory storage as NumPy arrays
    - Performance monitoring and profiling
    """

    metadata = {"render_modes": ["human"], "name": "optimized_billboard_env"}
    
    def __init__(self, billboard_csv: str, advertiser_csv: str, trajectory_csv: str,
                 action_mode: str = "na", config: Optional[EnvConfig] = None,
                 start_time_min: Optional[int] = None, seed: Optional[int] = None):
        
        super().__init__()
        
        # Use provided config or default
        self.config = config or EnvConfig()
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.action_mode = action_mode.lower()
        
        if self.action_mode not in ['na', 'ea', 'mh']:
            raise ValueError(f"Unsupported action_mode: {action_mode}. Use 'na', 'ea', or 'mh'")
        
        logger.info(f"Initializing OptimizedBillboardEnv with action_mode={self.action_mode}")
        
        # Performance monitoring
        self.perf_monitor = PerformanceMonitor() if self.config.enable_profiling else None

        # Load and process data
        self._load_data(billboard_csv, advertiser_csv, trajectory_csv, start_time_min)
        
        # Precompute billboard properties
        self._precompute_billboard_properties()
        
        # Create graph structure
        self.edge_index = self._create_billboard_graph()
        logger.info(f"Created graph with {self.edge_index.shape[1]} edges")

        # Gym-style action/observation spaces
        self._setup_action_observation_spaces()

        # Runtime state
        self._initialize_state()
    
    def _load_data(self, billboard_csv: str, advertiser_csv: str, 
                   trajectory_csv: str, start_time_min: Optional[int]):
        """Load and preprocess all data files with validation."""
        
        # Load and validate billboard data
        bb_df = pd.read_csv(billboard_csv)
        validate_csv(bb_df, ['B_id', 'Latitude', 'Longitude', 'B_Size', 'B_Cost', 'Influence'], 
                    "Billboard CSV")
        logger.info(f"Loaded {len(bb_df)} billboard entries")
        
        # Get unique billboards
        uniq_df = bb_df.drop_duplicates(subset=['B_id'], keep='first')
        logger.info(f"Found {len(uniq_df)} unique billboards")
        
        # Create billboard objects
        self.billboards: List[Billboard] = []
        for _, r in uniq_df.iterrows():
            self.billboards.append(Billboard(
                int(r['B_id']), float(r['Latitude']), float(r['Longitude']),
                r.get('Tags', ''), float(r['B_Size']), float(r['B_Cost']),
                float(r['Influence'])
            ))
        
        self.n_nodes = len(self.billboards)
        self.billboard_map = {b.b_id: b for b in self.billboards}
        self.billboard_id_to_node_idx = {b.b_id: i for i, b in enumerate(self.billboards)}
        
        # Load and validate advertiser data
        adv_df = pd.read_csv(advertiser_csv)
        adv_df.columns = adv_df.columns.str.strip().str.replace('\ufeff', '')
        validate_csv(adv_df, ['Id', 'Demand', 'Payment', 'Payment_Demand_Ratio'],
                    "Advertiser CSV")
        logger.info(f"Loaded {len(adv_df)} advertiser templates")
        
        self.ads_db: List[Ad] = []
        for aid, demand, payment, ratio in zip(
            adv_df['Id'].values,
            adv_df['Demand'].values,
            adv_df['Payment'].values,
            adv_df['Payment_Demand_Ratio'].values):
            
            self.ads_db.append(
                Ad(aid=int(aid), demand=float(demand), payment=float(payment),
                   payment_demand_ratio=float(ratio), ttl=self.config.ad_ttl)
            )
        
        # Load and validate trajectory data
        traj_df = pd.read_csv(trajectory_csv)
        validate_csv(traj_df, ['Time', 'Latitude', 'Longitude'], "Trajectory CSV")
        
        traj_df['t_min'] = traj_df['Time'].apply(time_str_to_minutes)
        self.start_time_min = int(start_time_min if start_time_min is not None 
                                 else traj_df['t_min'].min())
        
        # Preprocess trajectories as NumPy arrays for efficient operations
        self.trajectory_map = self._preprocess_trajectories_optimized(traj_df)
        logger.info(f"Processed trajectories for {len(self.trajectory_map)} time points")
    
    def _preprocess_trajectories_optimized(self, df: pd.DataFrame) -> Dict[int, np.ndarray]:
        """Preprocess trajectory data as NumPy arrays for vectorized operations."""
        traj_map: Dict[int, np.ndarray] = {}
        
        for t_min, grp in df.groupby('t_min'):
            # Store as float32 NumPy array for efficiency
            coords = np.column_stack([
                grp['Latitude'].values.astype(np.float32),
                grp['Longitude'].values.astype(np.float32)
            ])
            traj_map[int(t_min)] = coords
        
        return traj_map
    
    def _precompute_billboard_properties(self):
        """Precompute billboard properties for efficiency."""
        # Find max billboard size
        self.max_billboard_size = max((b.b_size for b in self.billboards), default=1.0)
        
        # Precompute normalized sizes
        for b in self.billboards:
            b.p_size = (b.b_size / self.max_billboard_size) if self.max_billboard_size > 0 else 0.0
        
        # Store billboard coordinates as NumPy arrays for vectorized distance calculations
        self.billboard_coords = np.array([
            [b.latitude, b.longitude] for b in self.billboards
        ], dtype=np.float32)
        
        # Precompute size ratios
        self.billboard_size_ratios = np.array([
            b.b_size / self.max_billboard_size for b in self.billboards
        ], dtype=np.float32)

    def _precompute_slot_influence(self, start_step: int) -> np.ndarray:
        """Compute expected influence for each billboard slot starting at current step.

        For each billboard and each possible duration (1-5), compute how many
        users will pass within influence_radius during that slot.

        This uses pre-loaded trajectory data to predict future influence,
        similar to how real advertising companies use historical traffic data.

        Returns:
            slot_influence: (n_billboards, max_duration) array
            slot_influence[b, d] = expected users within influence_radius of billboard over the next timesteps
        """
        # PER-STEP CACHE: Avoid recomputing multiple times per step
        if hasattr(self, '_slot_influence_cache_step') and self._slot_influence_cache_step == start_step:
            return self._slot_influence_cache_data

        max_duration = self.config.slot_duration_range[1]  # 5
        slot_influence = np.zeros((self.n_nodes, max_duration), dtype=np.float32)

        for d in range(max_duration):
            future_step = start_step + d + 1
            minute_key = (self.start_time_min + future_step) % 1440
            user_locs = self.trajectory_map.get(minute_key, np.array([]))

            if len(user_locs) == 0:
                # No users at this minute - carry forward previous cumulative value
                if d > 0:
                    slot_influence[:, d] = slot_influence[:, d-1]
                continue

            # Vectorized distance: (n_users, n_billboards)
            distances = haversine_distance_vectorized(
                user_locs[:, 0:1], user_locs[:, 1:2],
                self.billboard_coords[:, 0:1].T, self.billboard_coords[:, 1:2].T
            )

            # Count users within influence radius for each billboard
            within_radius = distances <= self.config.influence_radius_meters
            users_per_billboard = within_radius.sum(axis=0).astype(np.float32)  # (n_billboards,)

            # Cumulative: slot_influence[b, d] = total users over steps 1..d+1
            if d == 0:
                slot_influence[:, d] = users_per_billboard
            else:
                slot_influence[:, d] = slot_influence[:, d-1] + users_per_billboard

        # Cache result for this step
        self._slot_influence_cache_step = start_step
        self._slot_influence_cache_data = slot_influence
        return slot_influence

    def get_expected_slot_influence(self) -> np.ndarray:
        """Get expected influence for max slot duration (used for masking).

        Returns:
            expected_influence: (n_billboards,) array normalized to [0, 1]
        """
        slot_influence = self._precompute_slot_influence(self.current_step)

        # Use MAX duration for masking (conservative - don't mask billboards with delayed traffic)
        # If a billboard has 0 influence over the full slot duration, it's truly dead
        max_duration = self.config.slot_duration_range[1]  # 7 for (3,7)
        max_duration_idx = max_duration - 1  # 0-indexed: duration 7 → index 6
        raw_influence = slot_influence[:, max_duration_idx]

        # Normalize: typical range is 0-50 users, cap at 100
        MAX_USERS = 100.0
        normalized = np.clip(raw_influence / MAX_USERS, 0.0, 1.0)

        return normalized.astype(np.float32)

    def _get_allocation_expected_influence(self, bb_idx: int, duration: int) -> float:
        """Get expected influence for a specific allocation.

        Args:
            bb_idx: Billboard index
            duration: Slot duration (1-5)

        Returns:
            Expected number of users that will see this billboard over the slot duration
        """
        slot_influence = self._precompute_slot_influence(self.current_step)
        return float(slot_influence[bb_idx, duration - 1])  

    def _create_billboard_graph(self) -> np.ndarray:
        """Create adjacency matrix for billboards using vectorized distance calculation."""
        n = len(self.billboards)
        
        if n == 0:
            return np.array([[0], [0]])
        
        # Vectorized distance calculation
        coords = self.billboard_coords
        lat1 = coords[:, 0:1]  # Shape (n, 1)
        lon1 = coords[:, 1:2]  # Shape (n, 1)
        lat2 = coords[:, 0].reshape(1, -1)  # Shape (1, n)
        lon2 = coords[:, 1].reshape(1, -1)  # Shape (1, n)
        
        # Calculate all pairwise distances at once
        distances = haversine_distance_vectorized(lat1, lon1, lat2, lon2)
        
        # Find edges within threshold
        valid_pairs = np.where((distances <= self.config.graph_connection_distance) & 
                              (distances > 0))
        
        if len(valid_pairs[0]) > 0:
            edges = np.column_stack(valid_pairs)
            # Add reverse edges for bidirectional graph
            edges_reverse = edges[:, [1, 0]]
            all_edges = np.vstack([edges, edges_reverse])
            return all_edges.T
        else:
            # If no edges, create self-loops
            edges = np.array([[i, i] for i in range(n)])
            return edges.T
    
    def _setup_action_observation_spaces(self):
        """Setup action and observation spaces based on action mode (Gym-style)."""
        # Node features: billboard properties
        self.n_node_features = 10
        # Ad features: advertisement properties (updated to 12 with budget tracking)
        self.n_ad_features = 12

        if self.action_mode == 'na':
            # Node Action: select billboard (ad chosen by environment)
            self.action_space = spaces.Discrete(self.n_nodes)
            self.observation_space = spaces.Dict({
                'graph_nodes': spaces.Box(low=-np.inf, high=np.inf,
                                         shape=(self.n_nodes, self.n_node_features), 
                                         dtype=np.float32),
                'graph_edge_links': spaces.Box(low=0, high=self.n_nodes-1,
                                              shape=(2, self.edge_index.shape[1]), 
                                              dtype=np.int64),
                'mask': spaces.MultiBinary(self.n_nodes),
                'current_ad': spaces.Box(low=-np.inf, high=np.inf,
                                        shape=(self.n_ad_features,), dtype=np.float32)
            })
        
        elif self.action_mode == 'ea':
            # Edge Action: select ad-billboard pairs
            max_pairs = self.config.max_active_ads * self.n_nodes
            self.action_space = spaces.MultiBinary(max_pairs)
            # Edge features: 3 semantic features per (ad, billboard) pair
            n_edge_features = 3
            self.observation_space = spaces.Dict({
                'graph_nodes': spaces.Box(low=-np.inf, high=np.inf,
                                         shape=(self.n_nodes, self.n_node_features),
                                         dtype=np.float32),
                'graph_edge_links': spaces.Box(low=0, high=self.n_nodes-1,
                                              shape=(2, self.edge_index.shape[1]),
                                              dtype=np.int64),
                'ad_features': spaces.Box(low=-np.inf, high=np.inf,
                                         shape=(self.config.max_active_ads, self.n_ad_features),
                                         dtype=np.float32),
                'edge_features': spaces.Box(low=0.0, high=1.0,
                                           shape=(self.config.max_active_ads, self.n_nodes, n_edge_features),
                                           dtype=np.float32),
                'mask': spaces.MultiBinary(max_pairs)
            })

        elif self.action_mode == 'mh':
            # Multi-Head: two sequential categorical decisions
            # Action[0] = ad index (0 to max_active_ads-1)
            # Action[1] = billboard index (0 to n_nodes-1)
            self.action_space = spaces.MultiDiscrete([self.config.max_active_ads, self.n_nodes])
            self.observation_space = spaces.Dict({
                'graph_nodes': spaces.Box(low=-np.inf, high=np.inf,
                                         shape=(self.n_nodes, self.n_node_features),
                                         dtype=np.float32),
                'graph_edge_links': spaces.Box(low=0, high=self.n_nodes-1,
                                              shape=(2, self.edge_index.shape[1]),
                                              dtype=np.int64),
                'ad_features': spaces.Box(low=-np.inf, high=np.inf,
                                         shape=(self.config.max_active_ads, self.n_ad_features),
                                         dtype=np.float32),
                # Mask shape stays same: (max_active_ads, n_nodes) for valid (ad, bb) pairs
                'mask': spaces.MultiBinary([self.config.max_active_ads, self.n_nodes])
            })
    
    def _initialize_state(self):
        """Initialize runtime state variables (Gym-style, single-agent)."""
        self.current_step = 0
        self.ads: List[Ad] = []
        self.placement_history: List[Dict[str, Any]] = []
        self.current_ad_for_na_mode: Optional[Ad] = None
        self.performance_metrics = {
            'total_ads_processed': 0,
            'total_ads_completed': 0,
            'total_ads_tardy': 0,
            'total_revenue': 0.0,
            'billboard_utilization': 0.0
        }

        # CANONICAL REWARD: Event-based tracking
        self.ads_completed_this_step: List[int] = []
        self.ads_failed_this_step: List[int] = []  # Track new failures
        self.allocations_this_step: int = 0  # Track successful allocations for reward shaping

        # ADVERTISER TRACKING: Track used advertiser IDs within current episode
        # This set acts as a "discard pile" - once an advertiser is used, they
        # cannot be used again until reset() is called (new episode/game)
        self.used_advertiser_ids: set = set()

        if self.perf_monitor:
            self.perf_monitor.reset()
    
    def distance_factor(self, dist_meters: np.ndarray) -> np.ndarray:
        """Vectorized distance effect on billboard influence.

        SIMPLIFIED: All users within influence radius are equally influenced.
        Previously used linear decay which reduced influence at distance.
        Distance threshold already handled by influence_radius_meters (100m).
        """
        # All users within radius contribute equally (no distance decay)
        return np.ones_like(dist_meters)
    
    def get_mask(self) -> np.ndarray:
        """
        Get action mask based on current action mode with budget validation.

        OPTIMIZED: Uses NumPy vectorization instead of nested Python loops.
        Now includes INFLUENCE MASKING: billboards with 0 expected influence are masked out.
        """
        # Precompute billboard properties once (used by all modes)
        free_mask = np.array([b.is_free() for b in self.billboards], dtype=bool)
        costs = np.array([b.b_cost for b in self.billboards], dtype=np.float32)

        # INFLUENCE MASKING: Mask out billboards with zero influence at current time
        # This prevents wasting allocations/budget on "dead" billboards with no traffic
        current_influence = self.get_expected_slot_influence()  # Shape: (n_nodes,), normalized [0,1]
        has_influence = current_influence > 0.001  # Small threshold for float precision

        # Combine: billboard must be free AND have current traffic
        free_mask = free_mask & has_influence

        # Account for per-timestep cost: use max duration for conservative estimate
        max_duration = self.config.slot_duration_range[1]
        total_costs = costs * max_duration

        if self.action_mode == 'na':
            # NA mode: mask free billboards that current ad can afford
            if self.current_ad_for_na_mode is not None:
                # Vectorized: free AND affordable (cost × max_duration)
                affordable = self.current_ad_for_na_mode.remaining_budget >= total_costs
                mask = (free_mask & affordable).astype(np.int8)
            else:
                mask = free_mask.astype(np.int8)

            if mask.sum() == 0:
                logger.warning("No affordable free billboards available for 'na' mode")
            return mask

        elif self.action_mode == 'ea':
            # EA mode: mask valid ad-billboard pairs (flattened)
            active_ads = [ad for ad in self.ads if ad.state == 0]
            n_active = min(len(active_ads), self.config.max_active_ads)

            if n_active == 0:
                logger.warning("No affordable ad-billboard pairs for 'ea' mode")
                return np.zeros(self.config.max_active_ads * self.n_nodes, dtype=np.int8)

            # Vectorized budget check: (n_active,) budgets vs (n_nodes,) total_costs
            budgets = np.array([active_ads[i].remaining_budget for i in range(n_active)], dtype=np.float32)
            # Broadcasting: (n_active, 1) >= (1, n_nodes) -> (n_active, n_nodes)
            # Use total_costs (cost × max_duration) for conservative estimate
            affordable = budgets[:, None] >= total_costs[None, :]
            # Combine: (n_active, n_nodes) & (n_nodes,) broadcast
            valid_pairs = affordable & free_mask

            # Create full mask with zero-padding for unused ad slots
            full_mask = np.zeros((self.config.max_active_ads, self.n_nodes), dtype=np.int8)
            full_mask[:n_active, :] = valid_pairs.astype(np.int8)

            mask = full_mask.flatten()
            if mask.sum() == 0:
                logger.warning("No affordable ad-billboard pairs for 'ea' mode")
            return mask

        elif self.action_mode == 'mh':
            # MH mode: 2D mask over (ad_idx, bb_idx) pairs
            active_ads = [ad for ad in self.ads if ad.state == 0]
            n_active = min(len(active_ads), self.config.max_active_ads)

            if n_active == 0:
                logger.warning("No affordable ad-billboard pairs for 'mh' mode")
                return np.zeros((self.config.max_active_ads, self.n_nodes), dtype=np.int8)

            # Vectorized budget check (cost × max_duration)
            budgets = np.array([active_ads[i].remaining_budget for i in range(n_active)], dtype=np.float32)
            affordable = budgets[:, None] >= total_costs[None, :]
            valid_pairs = affordable & free_mask

            # Create full mask with zero-padding
            pair_mask = np.zeros((self.config.max_active_ads, self.n_nodes), dtype=np.int8)
            pair_mask[:n_active, :] = valid_pairs.astype(np.int8)

            if pair_mask.sum() == 0:
                logger.warning("No affordable ad-billboard pairs for 'mh' mode")
            return pair_mask

        return np.array([1], dtype=np.int8)

    def get_edge_features(self) -> np.ndarray:
        """Compute semantic edge features for all (ad, billboard) pairs.

        Returns 3 features per pair to help the model learn matching rules:
        1. budget_ratio: Can the ad afford this billboard? (0-1)
        2. influence_score: Billboard's reach potential (normalized influence)
        3. is_free: Is the billboard available? (0 or 1)

        Returns:
            edge_features: (max_ads, n_billboards, 3) array
        """
        n_edge_features = 3
        edge_features = np.zeros(
            (self.config.max_active_ads, self.n_nodes, n_edge_features),
            dtype=np.float32
        )

        active_ads = [ad for ad in self.ads if ad.state == 0]
        n_active = min(len(active_ads), self.config.max_active_ads)

        if n_active == 0:
            return edge_features

        # Precompute billboard properties (vectorized)
        max_duration = self.config.slot_duration_range[1]
        billboard_costs = np.array([b.b_cost * max_duration for b in self.billboards],
                                   dtype=np.float32)
        # DYNAMIC: Expected influence based on trajectory data for current time
        # This replaces static billboard.influence with time-varying expected users
        # Real-world analogy: Advertising companies use historical traffic data
        billboard_influence = self.get_expected_slot_influence()  # Already [0, 1]
        billboard_free = np.array([1.0 if b.is_free() else 0.0 for b in self.billboards],
                                  dtype=np.float32)

        # Compute features for each active ad
        for i in range(n_active):
            ad = active_ads[i]
            # Feature 1: Budget ratio (can afford?)
            edge_features[i, :, 0] = np.minimum(
                1.0, ad.remaining_budget / np.maximum(billboard_costs, 1e-6)
            )
            # Feature 2: Billboard influence (reach potential)
            edge_features[i, :, 1] = billboard_influence
            # Feature 3: Is billboard free?
            edge_features[i, :, 2] = billboard_free

        return edge_features

    def _get_obs(self) -> Dict[str, Any]:
        """Get current observation."""
        # Get DYNAMIC influence based on current traffic (trajectory data)
        # This replaces static CSV influence with time-varying expected users
        dynamic_influence = self.get_expected_slot_influence()  # Shape: (n_billboards,)

        # Node features (billboards)
        nodes = np.zeros((self.n_nodes, self.n_node_features), dtype=np.float32)
        for i, b in enumerate(self.billboards):
            feat = b.get_feature_vector()
            # CRITICAL FIX: Overwrite static influence (index 4) with dynamic influence
            # This gives the agent "vision" of current traffic patterns
            feat[4] = dynamic_influence[i]
            nodes[i] = feat
        
        obs = {
            'graph_nodes': nodes,
            'graph_edge_links': self.edge_index.copy(),
            'mask': self.get_mask()
        }
        
        # Add ad features for modes that need them
        if self.action_mode in ['ea', 'mh']:
            ad_features = np.zeros((self.config.max_active_ads, self.n_ad_features),
                                  dtype=np.float32)
            active_ads = [ad for ad in self.ads if ad.state == 0]

            for i, ad in enumerate(active_ads[:self.config.max_active_ads]):
                ad_features[i] = ad.get_feature_vector()

            obs['ad_features'] = ad_features

            # Add semantic edge features for EA mode
            if self.action_mode == 'ea':
                obs['edge_features'] = self.get_edge_features()
        
        # Add current ad for NA mode
        elif self.action_mode == 'na':
            active_ads = [ad for ad in self.ads if ad.state == 0]
            if active_ads:
                # DETERMINISTIC: Select most urgent ad (lowest TTL ratio) for Markov property
                # This fixes the non-Markovian behavior from random selection
                self.current_ad_for_na_mode = min(active_ads,
                                                  key=lambda ad: ad.ttl / max(ad.original_ttl, 1))
                obs['current_ad'] = self.current_ad_for_na_mode.get_feature_vector()
            else:
                self.current_ad_for_na_mode = None
                obs['current_ad'] = np.zeros(self.n_ad_features, dtype=np.float32)
        
        return obs

    def _compute_reward(self) -> float:
        """
        SEMANTIC LEARNING REWARD FUNCTION (v3.0) - Expected Influence Edition

        Design principles:
        1. Expected influence bonus: Reward based on PREDICTED allocation value
        2. Strong progress signal: High coefficient for credit assignment
        3. High completion jackpot: Dominates any spam strategy
        4. Minimal failure penalty: Don't discourage exploration

        Key changes from v2:
        - Replaced unconditional allocation bonus with expected influence bonus
        - Completion jackpot: 20.0 → 100.0 (dominates spam)
        - Reward clip: 25.0 → 150.0 (let big wins show)
        """

        reward = 0.0

        # === 1. COMPLETION REWARDS (profit + efficiency bonus) ===
        for ad_id in self.ads_completed_this_step:
            ad = next((a for a in self.ads if a.aid == ad_id), None)
            if ad:
                # Base profit reward
                profit = ad.payment - ad.total_cost_spent
                reward += profit / 100.0  # Scale down for stability

                # Efficiency bonus: Reward high profit margins
                efficiency = profit / max(ad.payment, 1e-6)
                reward += efficiency * 2.0  # Bonus for cost-effective completion

                # COMPLETION JACKPOT - high enough to dominate spam strategy
                # With gamma=0.995, 30-step delay → 0.86 discount → 86 effective value
                reward += 100.0
            else:
                if self.config.debug:
                    logger.warning(f"Completion reward: Ad {ad_id} not found")

        # === 2. PROGRESS REWARDS (STRONGER shaping for credit assignment) ===
        for ad in self.ads:
            if ad.state == 0:  # Active ads only
                delta = getattr(ad, '_step_delta', 0.0)
                progress_ratio = delta / max(ad.demand, 1e-6)
                reward += progress_ratio * 2.0  # 4x stronger than before

        # === 3. SNIPER APPROACH: Reward quality, not quantity ===
        # Problem: -0.5 penalty was too harsh → agent paralyzed (-650/episode)
        # Problem: +0.05 baseline would bring back lazy spammer
        # Solution: Strong HIT bonus, tiny MISS penalty → 100x contrast
        # Math: Spam 1000 empty = -10, Pick 1 good (10 users) = +1.0
        if self.allocations_this_step > 0:
            if self.expected_influence_this_step > 0.001:  # HIT: Billboard has traffic
                # Strong positive reinforcement based on quality
                # Scale: 10 users = +1.0 reward, capped at +5.0
                reward += min(self.expected_influence_this_step * 0.1, 5.0)
            else:  # MISS: Billboard is empty
                # Tiny "transaction cost" - discourages spam without paralysis
                reward -= 0.01

        # === 4. FAILURE PENALTIES (minimal to encourage exploration) ===
        for ad_id in self.ads_failed_this_step:
            ad = next((a for a in self.ads if a.aid == ad_id), None)
            if ad:
                waste_ratio = ad.total_cost_spent / max(ad.payment, 1e-6)
                reward -= waste_ratio * 0.1  # Low penalty

        # === 5. WIDER CLIP (preserve completion jackpot signal) ===
        # Completion can now give 100+ reward, don't clip it away
        return np.clip(reward, -10.0, 150.0)   
    
    def _apply_influence_for_current_minute(self):
        """
        Apply influence for current minute using GLOBAL MATRIX vectorization.

        PERFORMANCE OPTIMIZATION: Single-pass computation for ALL billboards.
        Instead of computing distances per-ad (20 NumPy calls), we compute ONE
        global probability matrix and slice columns for each ad (O(1) lookup).

        This eliminates Python dispatch overhead by calling NumPy C-API once
        instead of N_ads times per timestep.

        CANONICAL REWARD: Track per-step delta for progress shaping.
        """
        minute_key = (self.start_time_min + self.current_step) % 1440

        if self.config.debug:
            logger.debug(f"Applying influence at step {self.current_step} (minute {minute_key})")

        # Get active ads that need influence calculation
        active_ads = [ad for ad in self.ads if ad.state == 0]
        if not active_ads:
            return

        # Get user locations for current time
        user_locs = self.trajectory_map.get(minute_key, np.array([]))
        if len(user_locs) == 0:
            # No users at this minute - no influence to apply
            for ad in active_ads:
                ad._step_delta = 0.0
            return

        # ========== GLOBAL MATRIX COMPUTATION (Single NumPy call) ==========
        # Compute distance from ALL users to ALL billboards in one operation
        n_users = len(user_locs)
        n_billboards = len(self.billboard_coords)

        # Extract coordinates for broadcasting
        user_lats = user_locs[:, 0:1]  # Shape: (n_users, 1)
        user_lons = user_locs[:, 1:2]  # Shape: (n_users, 1)
        bb_lats = self.billboard_coords[:, 0].reshape(1, -1)  # Shape: (1, n_billboards)
        bb_lons = self.billboard_coords[:, 1].reshape(1, -1)  # Shape: (1, n_billboards)

        # SINGLE Haversine call: (n_users, n_billboards) distance matrix
        global_distances = haversine_distance_vectorized(user_lats, user_lons, bb_lats, bb_lons)

        # Apply influence radius mask globally
        within_radius = global_distances <= self.config.influence_radius_meters

        # Compute global probability matrix
        global_probabilities = np.zeros_like(global_distances)

        if np.any(within_radius):
            # Base probability from size ratios: broadcast (1, n_billboards) across users
            global_probabilities[within_radius] = np.broadcast_to(
                self.billboard_size_ratios[None, :], (n_users, n_billboards)
            )[within_radius]

            # Apply distance decay factor
            global_probabilities[within_radius] *= self.distance_factor(global_distances[within_radius])

            # Numerical safety clamp
            global_probabilities = np.clip(global_probabilities, 0.0, 0.999999)

        # ========== PER-AD COLUMN SLICING (O(1) lookups) ==========
        for ad in active_ads:
            # Get billboard indices for this ad (fast dict lookups)
            bb_indices = [self.billboard_id_to_node_idx[b_id]
                         for b_id in ad.assigned_billboards
                         if b_id in self.billboard_id_to_node_idx]

            if not bb_indices:
                ad._step_delta = 0.0
                continue

            # COLUMN SLICE: Extract probabilities for this ad's billboards
            # This is O(1) memory reference, not recomputation
            ad_probabilities = global_probabilities[:, bb_indices]  # Shape: (n_users, n_ad_billboards)

            # Aggregate: 1 - prod(1 - p) for each user, then sum
            prob_no_influence = np.prod(1.0 - ad_probabilities, axis=1)
            total_influence = np.sum(1.0 - prob_no_influence)

            # Add this minute's influence directly (no delta tracking needed)
            # total_influence = expected users influenced THIS MINUTE
            ad._step_delta = total_influence
            ad.cumulative_influence += total_influence

            if self.config.debug and delta > 0:
                logger.debug(f"Ad {ad.aid} gained {delta:.4f} influence")

            # Complete ad if demand is satisfied
            if ad.cumulative_influence >= ad.demand:
                ad.state = 1  # completed
                self.performance_metrics['total_ads_completed'] += 1

                # CANONICAL REWARD: Track completion event
                self.ads_completed_this_step.append(ad.aid)

                # Release billboards and generate revenue
                for b_id in list(ad.assigned_billboards):
                    if b_id in self.billboard_map:
                        billboard = self.billboard_map[b_id]
                        billboard.revenue_generated += ad.payment / max(1, len(ad.assigned_billboards))
                        billboard.release()
                    ad.release_billboard(b_id)

                if self.config.debug:
                    logger.debug(f"Ad {ad.aid} completed with {ad.cumulative_influence:.2f}/{ad.demand} demand")
    
    def _tick_and_release_boards(self):
        """Tick billboard timers and release expired ones."""
        for b in self.billboards:
            if not b.is_free():
                b.occupied_until -= 1
                
                if b.occupied_until <= 0:
                    ad_id = b.release()
                    if ad_id is not None:
                        ad = next((a for a in self.ads if a.aid == ad_id), None)
                        if ad:
                            ad.release_billboard(b.b_id)
                            
                            # Update placement history
                            for rec in self.placement_history:
                                if (rec['ad_id'] == ad.aid and
                                    rec['billboard_id'] == b.b_id and
                                    'fulfilled_by_end' not in rec):
                                    rec['fulfilled_by_end'] = ad.cumulative_influence
                                    break
    
    def _spawn_ads(self):
        """Spawn new ads based on configuration with HYSTERESIS.

        Hysteresis logic:
        - Only spawn new ads when active count drops below LOW_THRESHOLD (15)
        - Then spawn until reaching HIGH_THRESHOLD (max_active_ads = 20)
        - This ensures continuous ad flow without blocking when at capacity

        Uses "deck of cards" logic:
        - Each advertiser can only be used ONCE per episode
        - Available pool = All advertisers - (Currently active ∪ Previously used)
        - Once used, advertiser goes to "discard pile" (used_advertiser_ids)
        - Discard pile is only cleared when reset() is called (new episode)
        """
        # Remove completed/tardy ads
        self.ads = [ad for ad in self.ads if ad.state == 0]

        # HYSTERESIS: Only spawn when below low threshold
        # LOW_THRESHOLD is 75% of max to ensure continuous ad flow
        LOW_THRESHOLD = max(1, int(self.config.max_active_ads * 0.75))  # 6 when max=8
        HIGH_THRESHOLD = self.config.max_active_ads  # 8

        active_count = len(self.ads)
        if active_count >= LOW_THRESHOLD:
            return  # Wait until ads complete/expire

        # Spawn new ads (up to HIGH_THRESHOLD)
        n_spawn = random.randint(*self.config.new_ads_per_step_range)

        # Get currently active advertiser IDs
        current_ad_ids = {ad.aid for ad in self.ads}

        # CRITICAL FIX: Exclude both active AND previously used advertiser IDs
        # This ensures each advertiser is used only once per episode (like drawing from a deck)
        excluded_ids = current_ad_ids | self.used_advertiser_ids  # Union of both sets
        available_templates = [a for a in self.ads_db if a.aid not in excluded_ids]

        # Handle edge case: Empty pool (all advertisers have been used)
        if len(available_templates) == 0:
            if self.config.debug:
                logger.debug(f"Advertiser pool exhausted: {len(self.used_advertiser_ids)} used, "
                           f"{len(current_ad_ids)} active, {len(self.ads_db)} total")
            return  # Cannot spawn new ads - deck is empty

        spawn_count = min(
            HIGH_THRESHOLD - len(self.ads),  # Up to HIGH_THRESHOLD
            n_spawn,
            len(available_templates)
        )

        if spawn_count > 0:
            selected_templates = random.sample(available_templates, spawn_count)
            for template in selected_templates:
                new_ad = Ad(
                    template.aid, template.demand, template.payment,
                    template.payment_demand_ratio, template.ttl
                )
                new_ad.spawn_step = self.current_step
                self.ads.append(new_ad)

                # CRITICAL: Add to "discard pile" - this advertiser cannot be used again
                # until reset() is called (new episode starts)
                self.used_advertiser_ids.add(template.aid)

                self.performance_metrics['total_ads_processed'] += 1

            if self.config.debug:
                logger.debug(f"Spawned {spawn_count} new ads. Pool status: "
                           f"{len(available_templates)-spawn_count} remaining, "
                           f"{len(self.used_advertiser_ids)} used")
    
    def _execute_action(self, action):
        """Execute the selected action with validation."""
        try:
            if self.action_mode == 'na':
                # Node Action mode - now accepts single integer
                ad_to_assign = self.current_ad_for_na_mode
                if ad_to_assign and isinstance(action, (int, np.integer)):
                    bb_idx = int(action)
                    
                    # Check if action is valid and billboard is free
                    if 0 <= bb_idx < self.n_nodes and self.billboards[bb_idx].is_free():
                        billboard = self.billboards[bb_idx]

                        # BUDGET TRACKING: Charge cost × duration (per-timestep cost)
                        duration = random.randint(*self.config.slot_duration_range)
                        total_cost = billboard.b_cost * duration
                        if ad_to_assign.assign_billboard(billboard.b_id, total_cost):
                            billboard.assign(ad_to_assign.aid, duration)
                            self.allocations_this_step += 1  # Track for reward shaping
                            # Track expected influence for immediate reward
                            self.expected_influence_this_step += self._get_allocation_expected_influence(bb_idx, duration)

                            self.placement_history.append({
                                'spawn_step': ad_to_assign.spawn_step,
                                'allocated_step': self.current_step,
                                'ad_id': ad_to_assign.aid,
                                'billboard_id': billboard.b_id,
                                'duration': duration,
                                'demand': ad_to_assign.demand,
                                'cost': total_cost  # Total cost (per-timestep cost × duration)
                            })

                            if self.config.debug:
                                logger.debug(f"Assigned ad {ad_to_assign.aid} to billboard {billboard.b_id} "
                                           f"(total cost: ${total_cost:.2f} = ${billboard.b_cost:.2f}/step × {duration} steps, "
                                           f"remaining budget: ${ad_to_assign.remaining_budget:.2f})")
                        elif self.config.debug:
                            logger.warning(f"Ad {ad_to_assign.aid} can't afford billboard {billboard.b_id} "
                                         f"(total cost: ${total_cost:.2f} = ${billboard.b_cost:.2f}/step × {duration} steps, "
                                         f"budget: ${ad_to_assign.remaining_budget:.2f})")
                    
                    elif self.config.debug:
                        if not (0 <= bb_idx < self.n_nodes):
                            logger.warning(f"Invalid action (billboard index) {bb_idx}")
                        elif not self.billboards[bb_idx].is_free():
                            logger.warning(f"Action failed: Billboard {bb_idx} is not free")
                
                elif self.config.debug:
                    if not ad_to_assign:
                        logger.warning("NA action skipped: No ad to assign")
                    else:
                        logger.warning(f"Invalid action type for NA mode: {type(action)}")
            
            elif self.action_mode == 'ea':
                # Edge Action mode
                if isinstance(action, (list, np.ndarray)):
                    action = np.asarray(action).flatten()
                    expected_shape = self.config.max_active_ads * self.n_nodes
                    if action.shape[0] != expected_shape:
                        logger.warning(f"Invalid action shape for 'ea' mode: {action.shape}")
                        return

                    active_ads = [ad for ad in self.ads if ad.state == 0]

                    # CRITICAL FIX: Track used billboards to prevent multi-assign in same step
                    used_billboards = set()

                    # ALLOCATION: Process all selected pairs (no limit since influence masking
                    # already filters out zero-influence billboards - all allocations are productive)
                    # Prioritize by expected influence for best results

                    # Get all selected pair indices
                    selected_indices = np.where(action == 1)[0]

                    if len(selected_indices) > 0:
                        # PERFORMANCE: Compute once and reuse
                        cached_slot_influence = self._precompute_slot_influence(self.current_step)
                        # Normalized scores for sorting (use avg duration col 2)
                        influence_scores = np.clip(cached_slot_influence[:, 2] / 100.0, 0.0, 1.0)

                        # Score each selected pair by billboard's expected influence
                        pair_scores = []
                        for pair_idx in selected_indices:
                            bb_idx = pair_idx % self.n_nodes
                            score = influence_scores[bb_idx] if bb_idx < len(influence_scores) else 0
                            pair_scores.append((pair_idx, score))

                        # Sort by influence score descending (best billboards first)
                        pair_scores.sort(key=lambda x: x[1], reverse=True)

                        # Process ALL valid pairs (no MAX limit - influence mask ensures quality)
                        for pair_idx, _ in pair_scores:
                            ad_idx = pair_idx // self.n_nodes
                            bb_idx = pair_idx % self.n_nodes

                            if (ad_idx < min(len(active_ads), self.config.max_active_ads) and
                                self.billboards[bb_idx].is_free() and
                                bb_idx not in used_billboards):

                                ad_to_assign = active_ads[ad_idx]
                                billboard = self.billboards[bb_idx]

                                # BUDGET TRACKING: Charge cost × duration (per-timestep cost)
                                duration = random.randint(*self.config.slot_duration_range)
                                total_cost = billboard.b_cost * duration
                                if ad_to_assign.assign_billboard(billboard.b_id, total_cost):
                                    billboard.assign(ad_to_assign.aid, duration)
                                    self.allocations_this_step += 1
                                    # Use cached value instead of recomputing
                                    self.expected_influence_this_step += float(cached_slot_influence[bb_idx, duration - 1])

                                    used_billboards.add(bb_idx)

                                    self.placement_history.append({
                                        'spawn_step': ad_to_assign.spawn_step,
                                        'allocated_step': self.current_step,
                                        'ad_id': ad_to_assign.aid,
                                        'billboard_id': billboard.b_id,
                                        'duration': duration,
                                        'demand': ad_to_assign.demand,
                                        'cost': total_cost
                                    })

            elif self.action_mode == 'mh':
                # Multi-Head mode: action is (ad_idx, bb_idx) - two integers
                if isinstance(action, (list, np.ndarray, torch.Tensor)):
                    action = np.asarray(action).flatten()

                    if len(action) != 2:
                        if self.config.debug:
                            logger.warning(f"Invalid MH action length: {len(action)}, expected 2")
                        return

                    ad_idx = int(action[0])
                    bb_idx = int(action[1])

                    # Validate indices
                    active_ads = [ad for ad in self.ads if ad.state == 0]
                    if ad_idx >= len(active_ads) or ad_idx < 0:
                        if self.config.debug:
                            logger.debug(f"Invalid ad index: {ad_idx}, active ads: {len(active_ads)}")
                        return

                    if bb_idx >= self.n_nodes or bb_idx < 0:
                        if self.config.debug:
                            logger.debug(f"Invalid billboard index: {bb_idx}")
                        return

                    ad_to_assign = active_ads[ad_idx]
                    billboard = self.billboards[bb_idx]

                    # Check billboard is free
                    if not billboard.is_free():
                        if self.config.debug:
                            logger.debug(f"Billboard {bb_idx} is occupied")
                        return

                    # BUDGET TRACKING: Charge cost × duration
                    duration = random.randint(*self.config.slot_duration_range)
                    total_cost = billboard.b_cost * duration

                    if ad_to_assign.assign_billboard(billboard.b_id, total_cost):
                        billboard.assign(ad_to_assign.aid, duration)
                        self.allocations_this_step += 1

                        # Use cached slot influence
                        cached_slot_influence = self._precompute_slot_influence(self.current_step)
                        self.expected_influence_this_step += float(cached_slot_influence[bb_idx, duration - 1])

                        self.placement_history.append({
                            'spawn_step': ad_to_assign.spawn_step,
                            'allocated_step': self.current_step,
                            'ad_id': ad_to_assign.aid,
                            'billboard_id': billboard.b_id,
                            'duration': duration,
                            'demand': ad_to_assign.demand,
                            'cost': total_cost
                        })

        except Exception as e:
            logger.error(f"Error executing action: {e}")
            if self.config.debug:
                import traceback
                traceback.print_exc()

    # --- Gym required methods ---

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state (Gym-style).

        This is called at the start of each NEW EPISODE (new game).
        The "discard pile" of used advertisers is shuffled back into the deck,
        making all advertisers available again.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.current_step = 0
        self.ads.clear()

        # Reset billboards
        for b in self.billboards:
            b.release()
            b.total_usage = 0
            b.revenue_generated = 0.0

        # Reset tracking
        self.placement_history.clear()
        self.performance_metrics = {
            'total_ads_processed': 0,
            'total_ads_completed': 0,
            'total_ads_tardy': 0,
            'total_revenue': 0.0,
            'billboard_utilization': 0.0
        }

        # CANONICAL REWARD: Clear event tracking
        self.ads_completed_this_step.clear()
        self.ads_failed_this_step.clear()
        self.allocations_this_step = 0
        self.expected_influence_this_step = 0.0  # Track expected influence from allocations

        # ADVERTISER TRACKING: Shuffle deck - all advertisers become available again
        # This is the "new game" reset - the discard pile is cleared
        self.used_advertiser_ids.clear()

        if self.perf_monitor:
            self.perf_monitor.reset()

        # Spawn initial ads
        self._spawn_ads()

        logger.info(f"Environment reset with {len(self.ads)} initial ads")

        return self._get_obs(), {}
    
    def step(self, action):
        """Execute one environment step (Gym-style)."""
        return self._step_internal(action)
    
    
    def _step_internal(self, action):
        """Internal step implementation (Gym-style)."""
        # CANONICAL REWARD: Clear events from previous step
        self.ads_completed_this_step.clear()
        self.ads_failed_this_step.clear()
        self.allocations_this_step = 0
        self.expected_influence_this_step = 0.0  # Track expected influence from allocations

        # 1. Apply influence for current minute
        self._apply_influence_for_current_minute()

        # 2. Tick and release expired billboards
        self._tick_and_release_boards()

        # 3. Tick ad TTLs
        for ad in self.ads:
            prev_state = ad.state
            ad.step_time()
            if ad.state == 2 and prev_state != 2:  # became tardy
                self.performance_metrics['total_ads_tardy'] += 1
                # CANONICAL REWARD: Track new failures for event-based penalty
                self.ads_failed_this_step.append(ad.aid)

        # 4. Execute agent action
        self._execute_action(action)

        # 5. Compute reward
        reward = self._compute_reward()

        # 6. Update performance metrics
        self.performance_metrics['total_revenue'] = sum(b.revenue_generated for b in self.billboards)
        occupied_count = sum(1 for b in self.billboards if not b.is_free())
        self.performance_metrics['billboard_utilization'] = occupied_count / max(1, self.n_nodes) * 100

        # 7. Spawn new ads
        self._spawn_ads()

        # 8. Update termination conditions
        self.current_step += 1
        terminated = (self.current_step >= self.config.max_events)
        truncated = False

        # Build info dict
        info = {
            'total_revenue': self.performance_metrics['total_revenue'],
            'utilization': self.performance_metrics['billboard_utilization'],
            'ads_completed': self.performance_metrics['total_ads_completed'],
            'ads_processed': self.performance_metrics['total_ads_processed'],
            'ads_tardy': self.performance_metrics['total_ads_tardy'],
            'current_minute': (self.start_time_min + self.current_step) % 1440
        }

        # Episode-end summary (always logged, not controlled by debug flag)
        if terminated:
            c = self.performance_metrics['total_ads_completed']
            t = self.performance_metrics['total_ads_tardy']
            p = self.performance_metrics['total_ads_processed']
            r = self.performance_metrics['total_revenue']
            logger.info(f"[{self.action_mode.upper()}] {c}/{p} completed, {t} tardy, ${r:.0f}")

        return self._get_obs(), reward, terminated, truncated, info
    
    def render(self, mode="human"):
        """Render current environment state."""
        minute = (self.start_time_min + self.current_step) % 1440
        print(f"\n--- Step {self.current_step} | Time: {minute//60:02d}:{minute%60:02d} ---")
        
        # Show occupied billboards
        occupied = [b for b in self.billboards if not b.is_free()]
        print(f"\nOccupied Billboards ({len(occupied)}/{self.n_nodes}):")
        
        if not occupied:
            print("  None")
        else:
            for b in occupied[:10]:  # Show first 10
                idx = self.billboard_id_to_node_idx[b.b_id]
                print(f"  Node {idx} (ID: {b.b_id}): Ad {b.current_ad}, "
                      f"Time Left: {b.occupied_until}, Cost: {b.b_cost:.2f}")
            if len(occupied) > 10:
                print(f"  ... and {len(occupied) - 10} more")
        
        # Show active ads
        active_with_assignments = [ad for ad in self.ads if ad.assigned_billboards]
        print(f"\nActive Ads with Assignments ({len(active_with_assignments)}):")
        
        if not active_with_assignments:
            print("  None")
        else:
            for ad in active_with_assignments[:10]:
                state_str = ('Ongoing', 'Finished', 'Tardy')[ad.state]
                progress = f"{ad.cumulative_influence:.2f}/{ad.demand:.2f}"
                print(f"  Ad {ad.aid}: Progress={progress}, TTL={ad.ttl}, "
                      f"State={state_str}, Billboards={len(ad.assigned_billboards)}")
            if len(active_with_assignments) > 10:
                print(f"  ... and {len(active_with_assignments) - 10} more")
        
        # Show performance metrics
        metrics = self.performance_metrics
        print(f"\nPerformance Metrics:")
        print(f"  Processed: {metrics['total_ads_processed']}")
        print(f"  Completed: {metrics['total_ads_completed']}")
        print(f"  Tardy: {metrics['total_ads_tardy']}")
        print(f"  Revenue: ${metrics['total_revenue']:.2f}")
        print(f"  Utilization: {metrics['billboard_utilization']:.1f}%")
        if self.current_step >= self.config.max_events:
            self.render_summary()
    
    def render_summary(self):
        """Render final performance summary."""
        print(f"\n{'='*60}")
        print(f"SIMULATION COMPLETE - Final Results")
        print(f"{'='*60}")
        
        metrics = self.performance_metrics
        
        print(f"Total Ads Processed: {metrics['total_ads_processed']}")
        print(f"Successfully Completed: {metrics['total_ads_completed']}")
        print(f"Failed (Tardy): {metrics['total_ads_tardy']}")
        success_rate = (metrics['total_ads_completed'] / max(1, metrics['total_ads_processed'])) * 100.0
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Total Revenue Generated: ${metrics['total_revenue']:.2f}")
        print(f"Average Billboard Utilization: {metrics['billboard_utilization']:.1f}%")
        print(f"Total Placements: {len(self.placement_history)}")
        
        # Performance stats if profiling enabled
        if self.perf_monitor:
            self.perf_monitor.print_stats()
    
    def close(self):
        """Clean up environment."""
        pass