"""Model card registry — stores and serves model cards programmatically."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelCard:
    name: str
    description: str
    version: str = "1.0.0"
    model_type: str = ""
    input_features: list[str] = field(default_factory=list)
    output_description: str = ""
    training_data: str = ""
    known_limitations: list[str] = field(default_factory=list)
    tested_range: str = ""
    failure_modes: list[str] = field(default_factory=list)
    test_count: int = 0
    last_validated: str = ""


_registry: dict[str, ModelCard] = {}


def register_model_card(card: ModelCard):
    _registry[card.name] = card


def get_model_card(name: str) -> ModelCard | None:
    return _registry.get(name)


def list_model_cards() -> list[ModelCard]:
    return list(_registry.values())


def clear_registry():
    _registry.clear()


# ── Register built-in model cards ──

register_model_card(
    ModelCard(
        name="xG",
        description="Expected Goals — logistic regression model that estimates the probability a shot results in a goal",
        version="2.0.0",
        model_type="Logistic Regression (heuristic coefficients)",
        input_features=[
            "distance_to_goal",
            "angle_to_goal",
            "body_part",
            "one_on_one",
            "under_pressure",
            "gk_distance",
            "is_rebound",
            "big_chance",
        ],
        output_description="Probability (0.0–1.0) that the shot results in a goal",
        training_data="StatsBomb-like distribution — coefficients hand-tuned to approximate published xG models",
        known_limitations=[
            "Not fitted via maximum likelihood on own data",
            "No shot placement context",
            "No defender positioning beyond pressure flag",
            "Assumes all shots are independent",
        ],
        tested_range="Distance: 0–40m, Angle: 0–90°, 29 tests across shot scenarios",
        failure_modes=[
            "Very long shots (>40m) may be overestimated",
            "Deflections not modeled",
            "One-on-one flag may be stale by shot time",
        ],
        test_count=29,
    )
)

register_model_card(
    ModelCard(
        name="xT",
        description="Expected Threat — grid-based valuation of possession actions based on zone transitions",
        version="2.0.0",
        model_type="Grid valuation (20×32 zones with calibrated values)",
        input_features=["start_zone_x", "start_zone_y", "end_zone_x", "end_zone_y", "action_type"],
        output_description="Expected threat value (goals added) for the possession action",
        training_data="StatsBomb event data — calibrated zone values from published methodology",
        known_limitations=[
            "No context of surrounding players",
            "Assumes average team skill",
            "Grid resolution may miss fine spatial detail",
        ],
        tested_range="All 640 zones, all action types, 165 tests",
        failure_modes=[
            "Actions near touchline may have inflated threat",
            "Carry xT less validated than pass xT",
        ],
        test_count=165,
    )
)

register_model_card(
    ModelCard(
        name="VAEP",
        description="Valuing Actions by Estimating Probabilities — spatiotemporal action valuation framework",
        version="2.0.0",
        model_type="Spatiotemporal probability estimation with player-relative features",
        input_features=[
            "player_distance_to_event",
            "player_velocity_to_event",
            "teammate_density",
            "opponent_density",
            "event_type",
            "location_x",
            "location_y",
        ],
        output_description="VAEP value — change in scoring probability attributed to the action",
        training_data="Self-generated from tracking data — post_score_prob recalculated from next event",
        known_limitations=[
            "Requires accurate tracking (positions of all 22 players)",
            "Zero-value bug fixed in Sprint 11",
            "Does not model off-ball contributions",
        ],
        tested_range="All event types, 77 tests including correctness and property-based",
        failure_modes=[
            "With fragmented tracking (<15 players tracked), VAEP degrades significantly",
            "Set pieces not fully modeled",
        ],
        test_count=77,
    )
)

register_model_card(
    ModelCard(
        name="Pitch Control",
        description="Voronoi-based pitch control with ball-physics trajectory integration",
        version="2.0.0",
        model_type="Voronoi tessellation + RK4 ball trajectory",
        input_features=[
            "player_x",
            "player_y",
            "ball_x",
            "ball_y",
            "ball_z",
            "player_max_speed",
            "ball_velocity",
            "player_reaction_time",
        ],
        output_description="Per-player pitch control value (proportion of pitch controlled at each timestep)",
        training_data="Theoretical model — no training data required",
        known_limitations=[
            "Does not model tactical intent",
            "Players treated as having equal acceleration profiles",
            "Ball physics simplified (no spin/wind)",
        ],
        tested_range="5-frame sequences, 41 tests including numpy broadcasting optimization verification",
        failure_modes=[
            "With <18 players tracked, control map becomes unreliable",
            "Goalkeeper not modeled differently from outfield players",
        ],
        test_count=41,
    )
)

register_model_card(
    ModelCard(
        name="Win Probability",
        description="Monte Carlo simulation of match outcome based on xG rates",
        version="1.0.0",
        model_type="Monte Carlo (10,000 simulations, Poisson goal generation)",
        input_features=[
            "home_xg_rate",
            "away_xg_rate",
            "score_home",
            "score_away",
            "minutes_remaining",
            "home_advantage",
        ],
        output_description="Win/draw/loss probabilities at a given match state",
        training_data="No external training — Poisson assumption with team xG rates",
        known_limitations=[
            "Assumes goals are i.i.d. Poisson events",
            "No momentum or tactical context",
            "Does not model red cards, injuries, or substitutions",
        ],
        tested_range="All scorelines 0-0 to 5-5, all time states 0-90 min, 6 tests",
        failure_modes=[
            "Early in match (<15 min) probabilities are near 50/50 and uninformative",
            "Very one-sided matches may show overconfidence",
        ],
        test_count=6,
    )
)

register_model_card(
    ModelCard(
        name="Injury Risk",
        description="ACWR-based injury risk prediction using Hulin/Gabbett reference models",
        version="1.0.0",
        model_type="Heuristic with ACWR threshold bands",
        input_features=[
            "acute_load_7d",
            "chronic_load_28d",
            "sprint_volume",
            "total_distance",
            "fatigue_index",
            "position",
            "rest_days",
            "hrv_rmssd",
        ],
        output_description="Risk level (low/moderate/high/critical) and recovery recommendation",
        training_data="Hulin & Gabbett published reference models — not fitted on own data",
        known_limitations=[
            "Heuristic thresholds, not ML-based",
            "Does not incorporate subjective wellness",
            "No historical injury pattern matching",
        ],
        tested_range="All positions (GK/DEF/MID/FWD), ACWR 0.5-2.0, 12 tests",
        failure_modes=[
            "With <7 days of load data, ACWR is unreliable",
            "Acute spikes (single high-load session) may overstate risk",
        ],
        test_count=12,
    )
)

register_model_card(
    ModelCard(
        name="Formation Analysis",
        description="K-means clustering of player positions to detect tactical formations",
        version="1.0.0",
        model_type="K-means (k=10) with silhouette scoring for optimal k",
        input_features=["player_x_meters", "player_y_meters", "team_id", "timestamp"],
        output_description="Detected formation (e.g., '4-3-3'), per-player role assignments, width/depth metrics",
        training_data="No training data — unsupervised clustering on match positions",
        known_limitations=[
            "Requires homography calibration (real-world meters)",
            "Assumes players maintain position within phase",
            "May misclassify fluid systems",
        ],
        tested_range="5 detected formations (4-3-3, 4-4-2, 4-2-3-1, 3-5-2, 5-3-2), 11 tests",
        failure_modes=[
            "Without homography, formation detection is in pixel space and unreliable",
            "During transitions, detected formation may not reflect intended shape",
            "Substitutions mid-phase can distort cluster centers",
        ],
        test_count=11,
    )
)

# ── Phase 2 (elite-readiness): cards for the remaining headline models ──

register_model_card(
    ModelCard(
        name="PSxG",
        description="Post-Shot Expected Goals — probability an on-target shot becomes a goal, given placement",
        version="2.0.0",
        model_type="Logistic regression (trained) with hand-tuned fallback",
        input_features=[
            "distance_m",
            "angle_deg",
            "placement_x",
            "placement_y",
            "shot_speed",
            "body_part",
        ],
        output_description="PSxG probability (0-1) plus save probability and shot quality",
        training_data="Fitted from 2,768 StatsBomb open-data on-target shots (2026-09-07); heuristic fallback otherwise",
        known_limitations=[
            "Placement is approximated from freeze-frame data when absent",
            "No keeper-positioning context",
            "Shot speed default applied when unmeasured",
        ],
        tested_range="Distance 1-40m, placements across the goal mouth, trained vs heuristic parity tests",
        failure_modes=[
            "Very close-range headers can be overrated",
            "Deflected shots treated as clean strikes",
        ],
        test_count=0,
    )
)

register_model_card(
    ModelCard(
        name="xA",
        description="Expected Assists — xG of the shot a pass creates, credited to the passer",
        version="1.0.0",
        model_type="xG application to shot-assisting passes",
        input_features=[
            "pass_end_x",
            "pass_end_y",
            "shot_distance_m",
            "shot_angle_deg",
            "body_part",
        ],
        output_description="Expected-assist value per key pass and per-player totals",
        training_data="Reuses the StatsBomb-trained xG model on the shot the pass created",
        known_limitations=[
            "No receiver-positioning context",
            "Does not separate created-chance quality from finisher skill",
        ],
        tested_range="Match-level totals and per-event xA, wired into the Results view",
        failure_modes=["Crosses to crowded boxes may be undervalued (no defender feature)"],
        test_count=0,
    )
)

register_model_card(
    ModelCard(
        name="PPDA",
        description="Passes Per Defensive Action — pressing-intensity proxy",
        version="1.0.0",
        model_type="Ratio computation over opponent passes vs own defensive actions",
        input_features=["opponent_passes", "defensive_actions", "pitch_zone_filter"],
        output_description="PPDA value (lower = more intense pressing), optionally by pitch third",
        training_data="No training data — definitional metric",
        known_limitations=[
            "Counts opponent passes only while ball is in play in the defined zone",
            "Does not distinguish pressing quality from quantity",
        ],
        tested_range="Season and match level, home/away splits",
        failure_modes=["Matches with unusual defensive-action definitions shift the ratio"],
        test_count=0,
    )
)

register_model_card(
    ModelCard(
        name="Pass Network",
        description="Passing network with centrality metrics (betweenness, eigenvector)",
        version="1.0.0",
        model_type="Weighted directed graph + centrality algorithms",
        input_features=["passer_track_id", "receiver_track_id", "pass_count", "xT_gained"],
        output_description="Adjacency edges, node centrality, connection importance ranking",
        training_data="No training data — graph construction from completed passes",
        known_limitations=[
            "Needs reliable player identity (fragmented tracking merges nodes)",
            "Static aggregate — no temporal evolution within the match",
        ],
        tested_range="Centrality correctness on synthetic networks; per-match construction",
        failure_modes=["High track fragmentation inflates the number of pseudo-players"],
        test_count=0,
    )
)

register_model_card(
    ModelCard(
        name="Goals Added",
        description="Goals Added (g+) — per-action-type contribution above average",
        version="1.0.0",
        model_type="Zone-transition value tables per action type",
        input_features=["action_type", "start_zone", "end_zone", "player_track_id"],
        output_description="Goals-added value per action, per-player season totals",
        training_data="Zone values derived from league-average transition outcomes (StatsBomb-corpus-calibrated)",
        known_limitations=[
            "Coarse zones lose within-zone context",
            "No game-state leverage adjustment",
        ],
        tested_range="All wired action types, per-player aggregation",
        failure_modes=[
            "Players with unusual positional roles compare poorly to league-average zones"
        ],
        test_count=0,
    )
)

register_model_card(
    ModelCard(
        name="OBV",
        description="On-Ball Value — expected change in scoring probability per on-ball action",
        version="1.0.0",
        model_type="Pre/post-action possession-value differential (VAEP family)",
        input_features=["event_type", "location", "next_event_location", "score_state"],
        output_description="OBV value per action, per-player totals",
        training_data="Self-generated from tracking/event streams — post-action value from the following state",
        known_limitations=[
            "Requires complete event chains; gaps break the differential",
            "No off-ball contribution",
        ],
        tested_range="Synthetic chains with known differentials",
        failure_modes=["Event-detection errors propagate directly into OBV swings"],
        test_count=0,
    )
)

register_model_card(
    ModelCard(
        name="EPV",
        description="Expected Possession Value — continuous possession-value surface over the pitch",
        version="1.0.0",
        model_type="Pitch-grid value surface driven by ball position and control",
        input_features=["ball_x", "ball_y", "pitch_control", "possession_team"],
        output_description="EPV per frame, possession-level EPV delta, player share of value added",
        training_data="Reference surface calibrated from StatsBomb-corpus possession outcomes",
        known_limitations=[
            "Frame-rate dependent (works best on 25Hz vendor tracking)",
            "Control model errors flow into EPV",
        ],
        tested_range="Synthetic possessions; Metrica sample-game integration",
        failure_modes=["Without homography/vendor meters, EPV surface is meaningless pixel noise"],
        test_count=0,
    )
)

register_model_card(
    ModelCard(
        name="Momentum",
        description="Momentum index — rolling share of match control between teams",
        version="1.0.0",
        model_type="Time-weighted action aggregation",
        input_features=["event_type", "timestamp", "team", "xg", "location"],
        output_description="Per-team momentum percentages over time windows",
        training_data="No training data — weighted aggregation of match actions",
        known_limitations=[
            "Weights are hand-tuned, not learned",
            "No crowd/game-state psychological term",
        ],
        tested_range="Synthetic matches with dominant spells",
        failure_modes=["Choppy event timing (low fps detection) makes momentum noisy"],
        test_count=0,
    )
)

register_model_card(
    ModelCard(
        name="Progressive Actions",
        description="Progressive passes and carries — actions moving the ball significantly toward goal",
        version="1.0.0",
        model_type="Ratio-of-remaining-distance + attacking-third zone rule (StatsBomb-style)",
        input_features=["start_x", "start_y", "end_x", "end_y", "action_type"],
        output_description="Per-action progression flags, per-player and per-team totals",
        training_data="No training data — definitional rule from game_constants.py",
        known_limitations=[
            "Direction assumption: attacks toward x=105 (needs single attacking frame)",
            "Set-piece restarts can inflate counts",
        ],
        tested_range="Zone-aware distance thresholds vs flat-threshold baseline",
        failure_modes=["Pixel-space inputs (no homography) make the ratio rule meaningless"],
        test_count=0,
    )
)

register_model_card(
    ModelCard(
        name="Set Piece xT",
        description="Set-piece delivery threat — xT contribution of corners, free kicks, and throws",
        version="1.0.0",
        model_type="xT grid applied to set-piece end locations with delivery-type priors",
        input_features=["delivery_type", "start_zone", "end_zone", "outcome"],
        output_description="Per-delivery threat value, per-taker and per-match totals",
        training_data="xT reference grid trained on StatsBomb corpus (297 matches)",
        known_limitations=[
            "Delivery flight quality (inswing/outswing) not modeled",
            "First-contact outcomes only — second phases excluded",
        ],
        tested_range="Corner/free-kick/throw-in conversion across zone grid",
        failure_modes=["Misclassified delivery types apply the wrong prior"],
        test_count=0,
    )
)
