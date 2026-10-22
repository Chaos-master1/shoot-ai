"""Squad Value Estimation — heuristic market value estimation."""

from __future__ import annotations

from dataclasses import dataclass

POSITION_BASELINES: dict[str, float] = {
    "gk": 5.0,
    "def": 8.0,
    "mid": 12.0,
    "fwd": 15.0,
}

LEAGUE_MULTIPLIERS: dict[str, float] = {
    "premier_league": 1.0,
    "la_liga": 0.9,
    "bundesliga": 0.85,
    "serie_a": 0.85,
    "ligue1": 0.75,
    "championship": 0.4,
    "eredivisie": 0.35,
    "primeira_liga": 0.35,
    "mls": 0.25,
    "j_league": 0.20,
    "saudi_pro": 0.30,
}

TRANSFER_POSITION_BASELINES: dict[str, float] = {
    "gk": 2.0,
    "def": 4.0,
    "mid": 6.0,
    "fwd": 8.0,
}

LEAGUE_TRANSFER_MULTIPLIERS: dict[str, float] = {
    "premier_league": 1.0,
    "la_liga": 0.85,
    "bundesliga": 0.80,
    "serie_a": 0.80,
    "ligue1": 0.70,
    "championship": 0.35,
    "eredivisie": 0.30,
    "primeira_liga": 0.30,
    "mls": 0.20,
    "j_league": 0.15,
    "saudi_pro": 0.25,
}

PRIME_AGE_MIN = 22
PRIME_AGE_MAX = 28
PEAK_AGE = 25


@dataclass
class PlayerValuation:
    player_id: str
    estimated_value: float
    age_multiplier: float
    performance_score: float
    position_baseline: float
    contract_multiplier: float
    confidence: str


@dataclass
class SquadValuationReport:
    team_id: str
    total_squad_value: float
    players: list[PlayerValuation]
    avg_value: float
    most_valuable: str
    age_distribution: dict
    value_rating: str


def _age_multiplier(age: int) -> float:
    if age <= 21:
        return 1.2
    if age <= 28:
        peak = PEAK_AGE
        dist = abs(age - peak)
        return round(1.1 - dist * 0.025, 2)
    if age <= 32:
        return 0.7
    return 0.4


def _performance_score(stats: dict) -> float:
    if not stats:
        return 0.0
    minutes = stats.get("minutes_played", 0)
    if minutes < 90:
        return 5.0
    components = []
    for key in ("xg_per_90", "xa_per_90", "goals_per_90", "assists_per_90"):
        val = stats.get(key, 0.0)
        components.append(val)
    rating = stats.get("rating_per_90", 0.0)
    if rating:
        components.append(rating / 10.0)
    if not components:
        return 5.0
    avg = sum(components) / len(components)
    score = min(100.0, avg * 50.0 + 10.0)
    return round(score, 2)


def _contract_multiplier(years_remaining: int) -> float:
    if years_remaining >= 4:
        return 1.2
    if years_remaining >= 2:
        return 1.0
    if years_remaining == 1:
        return 0.8
    return 0.6


def _league_multiplier(league_tier: str) -> float:
    return LEAGUE_MULTIPLIERS.get(league_tier, 0.3)


def _confidence_label(
    performance_score: float,
    minutes_played: int,
    contract_years: int,
) -> str:
    if minutes_played >= 1500 and contract_years >= 2:
        return "high"
    if minutes_played >= 500:
        return "medium"
    return "low"


def _transfer_age_factor(age: int) -> float:
    if age <= 17:
        return 1.3
    if age <= 20:
        return 1.2
    if age <= 24:
        return 1.1
    if age == 25:
        return 1.0
    if age <= 28:
        return 0.9
    if age <= 30:
        return 0.7
    if age <= 32:
        return 0.5
    if age <= 34:
        return 0.35
    return 0.2


def _transfer_performance_score(stats: dict) -> float:
    if not stats:
        return 0.0
    components = []
    for key in ("xg_per_90", "xa_per_90", "goals_per_90", "assists_per_90"):
        val = stats.get(key, 0.0)
        components.append(val)
    rating = stats.get("rating_per_90", 0.0)
    if rating:
        components.append(rating / 10.0)
    minutes = stats.get("minutes_played", 0)
    if minutes:
        components.append(min(minutes / 1800.0, 1.0) * 0.5)
    if not components:
        return 0.0
    avg = sum(components) / len(components)
    return round(min(100.0, avg * 60.0 + 5.0), 2)


def _transfer_contract_factor(years_remaining: int) -> float:
    if years_remaining >= 4:
        return 1.2
    if years_remaining >= 2:
        return 1.0
    if years_remaining == 1:
        return 0.7
    return 0.5


def estimate_player_transfer_fee(
    age: int,
    position: str,
    performance_stats: dict,
    contract_years_remaining: int = 2,
    league_tier: str = "premier_league",
    market_trend: str = "stable",
    is_international: bool = False,
    injury_history: str = "low",
) -> dict:
    pos_key = position.lower()
    baseline = TRANSFER_POSITION_BASELINES.get(pos_key, 3.0)
    age_factor = _transfer_age_factor(age)
    perf = _transfer_performance_score(performance_stats)
    contract_factor = _transfer_contract_factor(contract_years_remaining)
    league_factor = LEAGUE_TRANSFER_MULTIPLIERS.get(league_tier, 0.15)
    trend_mult = {"rising": 1.15, "stable": 1.0, "declining": 0.85}.get(market_trend, 1.0)
    intl_premium = 1.2 if is_international else 1.0
    injury_discount = {"low": 1.0, "moderate": 0.8, "high": 0.5}.get(injury_history, 1.0)

    raw = (
        baseline
        * age_factor
        * (1.0 + perf / 150.0)
        * contract_factor
        * league_factor
        * trend_mult
        * intl_premium
        * injury_discount
    )
    estimated = round(raw, 2)

    minutes = performance_stats.get("minutes_played", 0)
    if (
        minutes >= 1500
        and contract_years_remaining >= 2
        and league_tier in ("premier_league", "la_liga", "bundesliga")
    ):
        confidence = "high"
    elif minutes >= 500:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "estimated_fee_millions": estimated,
        "age_factor": age_factor,
        "performance_score": perf,
        "position_baseline": baseline,
        "contract_factor": contract_factor,
        "league_factor": league_factor,
        "market_trend_adjustment": trend_mult,
        "international_premium": intl_premium,
        "injury_discount": injury_discount,
        "confidence": confidence,
        "fee_range": {
            "low": round(estimated * 0.6, 2),
            "mid": round(estimated * 0.85, 2),
            "high": round(estimated * 1.2, 2),
        },
    }


def estimate_player_value(
    player_id: str,
    age: int,
    position: str,
    performance_stats: dict,
    contract_years_remaining: int = 2,
    league_tier: str = "premier_league",
) -> PlayerValuation:
    pos_key = position.lower()
    baseline = POSITION_BASELINES.get(pos_key, 5.0)
    age_mult = _age_multiplier(age)
    perf = _performance_score(performance_stats)
    contract_mult = _contract_multiplier(contract_years_remaining)
    league_mult = _league_multiplier(league_tier)

    raw = baseline * age_mult * (1.0 + perf / 200.0) * contract_mult * league_mult
    estimated = round(raw, 2)
    conf = _confidence_label(
        perf, performance_stats.get("minutes_played", 0), contract_years_remaining
    )

    return PlayerValuation(
        player_id=player_id,
        estimated_value=estimated,
        age_multiplier=age_mult,
        performance_score=perf,
        position_baseline=baseline,
        contract_multiplier=contract_mult,
        confidence=conf,
    )


def estimate_squad_value(
    team_id: str,
    players: list[dict],
    league_tier: str = "premier_league",
) -> SquadValuationReport:
    valuations: list[PlayerValuation] = []
    for p in players:
        val = estimate_player_value(
            player_id=p["player_id"],
            age=p["age"],
            position=p["position"],
            performance_stats=p.get("performance_stats", {}),
            contract_years_remaining=p.get("contract_years_remaining", 2),
            league_tier=league_tier,
        )
        valuations.append(val)

    if not valuations:
        return SquadValuationReport(
            team_id=team_id,
            total_squad_value=0.0,
            players=[],
            avg_value=0.0,
            most_valuable="",
            age_distribution={"u21": 0, "prime": 0, "veteran": 0},
            value_rating="fair",
        )

    total = round(sum(v.estimated_value for v in valuations), 2)
    avg = round(total / len(valuations), 2)
    most_valuable = max(valuations, key=lambda v: v.estimated_value).player_id

    age_dist: dict[str, int] = {"u21": 0, "prime": 0, "veteran": 0}
    for p in players:
        a = p.get("age", 25)
        if a <= 21:
            age_dist["u21"] += 1
        elif a <= 28:
            age_dist["prime"] += 1
        else:
            age_dist["veteran"] += 1

    _ = sum(v.performance_score for v in valuations) / len(valuations) if valuations else 0
    total_baseline = sum(
        v.position_baseline * v.age_multiplier * _league_multiplier(league_tier) for v in valuations
    )
    value_ratio = total / total_baseline if total_baseline > 0 else 1.0

    if value_ratio < 0.8:
        value_rating = "underpriced"
    elif value_ratio > 1.2:
        value_rating = "overpriced"
    else:
        value_rating = "fair"

    return SquadValuationReport(
        team_id=team_id,
        total_squad_value=total,
        players=valuations,
        avg_value=avg,
        most_valuable=most_valuable,
        age_distribution=age_dist,
        value_rating=value_rating,
    )
