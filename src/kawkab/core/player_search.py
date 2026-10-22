"""Player Search — multi-criteria player search with scoring."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchCriteria:
    age_min: int = 16
    age_max: int = 40
    positions: list[str] = field(default_factory=lambda: [])
    leagues: list[str] = field(default_factory=lambda: [])
    stat_thresholds: dict[str, float] = field(default_factory=dict)
    stat_maximums: dict[str, float] = field(default_factory=dict)
    nationality: str | None = None
    height_min_cm: int | None = None
    height_max_cm: int | None = None
    preferred_foot: str | None = None
    sort_by: str = "rating_per_90"
    sort_dir: str = "DESC"
    limit: int = 50
    offset: int = 0


@dataclass
class SearchResult:
    player_id: str
    player_name: str
    age: int
    position: str
    league: str
    team: str
    nationality: str
    preferred_foot: str | None
    height_cm: int | None
    stats: dict[str, float]
    scout_report_summary: str | None
    match_score: float


def _matches_position(player_positions: str, criteria_positions: list[str]) -> bool:
    if not criteria_positions:
        return True
    pos_upper = player_positions.upper().replace("/", " ").split()
    for cp in criteria_positions:
        cp_upper = cp.upper()
        for pp in pos_upper:
            if cp_upper == pp:
                return True
    return False


def _compute_match_score(criteria: SearchCriteria, player: dict) -> float:
    score = 0.0
    total_weight = 0.0
    weights = {
        "age": 1.0,
        "position": 2.0,
        "league": 1.5,
        "nationality": 1.0,
        "height": 1.0,
        "foot": 1.0,
        "stat": 2.0,
    }
    if criteria.age_min > 16 or criteria.age_max < 40:
        age = player.get("age", 25)
        if criteria.age_min <= age <= criteria.age_max:
            score += weights["age"]
        total_weight += weights["age"]
    if criteria.positions:
        if _matches_position(player.get("position", ""), criteria.positions):
            score += weights["position"]
        total_weight += weights["position"]
    if criteria.leagues:
        if player.get("league", "") in criteria.leagues:
            score += weights["league"]
        total_weight += weights["league"]
    if criteria.nationality:
        if player.get("nationality", "").lower() == criteria.nationality.lower():
            score += weights["nationality"]
        total_weight += weights["nationality"]
    if criteria.height_min_cm is not None or criteria.height_max_cm is not None:
        h = player.get("height_cm")
        if h is not None:
            lo = criteria.height_min_cm if criteria.height_min_cm is not None else 0
            hi = criteria.height_max_cm if criteria.height_max_cm is not None else 999
            if lo <= h <= hi:
                score += weights["height"]
        total_weight += weights["height"]
    if criteria.preferred_foot:
        if player.get("preferred_foot", "").lower() == criteria.preferred_foot.lower():
            score += weights["foot"]
        total_weight += weights["foot"]
    player_stats = player.get("stats", {})
    for key, threshold in criteria.stat_thresholds.items():
        val = player_stats.get(key, 0.0)
        if val >= threshold:
            score += weights["stat"]
        total_weight += weights["stat"]
    for key, max_val in criteria.stat_maximums.items():
        val = player_stats.get(key, 0.0)
        if val <= max_val:
            score += weights["stat"]
        total_weight += weights["stat"]
    if total_weight == 0:
        return 100.0
    return round((score / total_weight) * 100.0, 1)


def search_players(
    criteria: SearchCriteria,
    player_database: list[dict],
) -> list[SearchResult]:
    results: list[SearchResult] = []
    for player in player_database:
        age = player.get("age", 25)
        if age < criteria.age_min or age > criteria.age_max:
            continue
        if criteria.positions and not _matches_position(
            player.get("position", ""), criteria.positions
        ):
            continue
        if criteria.leagues and player.get("league", "") not in criteria.leagues:
            continue
        if (
            criteria.nationality
            and player.get("nationality", "").lower() != criteria.nationality.lower()
        ):
            continue
        if (
            criteria.preferred_foot
            and player.get("preferred_foot", "").lower() != criteria.preferred_foot.lower()
        ):
            continue
        if criteria.height_min_cm is not None:
            h = player.get("height_cm")
            if h is None or h < criteria.height_min_cm:
                continue
        if criteria.height_max_cm is not None:
            h = player.get("height_cm")
            if h is None or h > criteria.height_max_cm:
                continue
        player_stats = player.get("stats", {})
        stat_pass = True
        for key, threshold in criteria.stat_thresholds.items():
            if player_stats.get(key, 0.0) < threshold:
                stat_pass = False
                break
        if not stat_pass:
            continue
        for key, max_val in criteria.stat_maximums.items():
            if player_stats.get(key, 0.0) > max_val:
                stat_pass = False
                break
        if not stat_pass:
            continue
        match_score = _compute_match_score(criteria, player)
        results.append(
            SearchResult(
                player_id=str(player.get("player_id", "")),
                player_name=player.get("name", ""),
                age=age,
                position=player.get("position", ""),
                league=player.get("league", ""),
                team=player.get("team", ""),
                nationality=player.get("nationality", ""),
                preferred_foot=player.get("preferred_foot"),
                height_cm=player.get("height_cm"),
                stats=player_stats,
                scout_report_summary=player.get("scout_report_summary"),
                match_score=match_score,
            )
        )
    sort_desc = criteria.sort_dir.upper() != "ASC"
    if criteria.sort_by == "match_score":
        results.sort(key=lambda r: r.match_score, reverse=sort_desc)
    elif criteria.sort_by == "age":
        results.sort(key=lambda r: r.age, reverse=sort_desc)
    elif criteria.sort_by == "player_name":
        results.sort(key=lambda r: r.player_name, reverse=sort_desc)
    else:

        def _sort_key(r: SearchResult) -> float:
            return r.stats.get(criteria.sort_by, 0.0)

        results.sort(key=_sort_key, reverse=sort_desc)
    return results[criteria.offset : criteria.offset + criteria.limit]
