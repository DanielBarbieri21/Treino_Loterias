from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LotteryConfig:
    name: str
    slug: str
    file_path: str
    number_min: int
    number_max: int
    pick_count: int

    @property
    def number_range(self) -> np.ndarray:
        return np.arange(self.number_min, self.number_max + 1)

    @property
    def total_numbers(self) -> int:
        return self.number_max - self.number_min + 1

    @property
    def expected_random_hits(self) -> float:
        return (self.pick_count / self.total_numbers) * self.pick_count


DEFAULT_STRATEGIES: List[Dict[str, object]] = [
    {"name": "freq_curta", "weights": {"freq_5": 0.55, "freq_10": 0.45}},
    {"name": "freq_media", "weights": {"freq_10": 0.5, "freq_20": 0.35, "freq_50": 0.15}},
    {"name": "decay", "weights": {"decay": 0.75, "freq_20": 0.25}},
    {"name": "equilibrada", "weights": {"freq_10": 0.25, "freq_20": 0.3, "freq_50": 0.15, "decay": 0.2, "repeat_rate": 0.1}},
    {"name": "coocorrencia", "weights": {"freq_20": 0.3, "decay": 0.25, "cooc_last": 0.3, "repeat_rate": 0.15}},
    {"name": "contrarian", "weights": {"freq_50": 0.35, "decay": 0.15, "overdue": 0.35, "gap_inverse": 0.15}},
]


def _safe_int(value: object) -> int | None:
    text = str(value).strip()
    if not text.isdigit():
        return None
    return int(text)


def _extract_draw(row: Sequence[object], config: LotteryConfig) -> Tuple[int, ...] | None:
    values: List[int] = []
    for raw in row:
        parsed = _safe_int(raw)
        if parsed is None:
            continue
        if config.number_min <= parsed <= config.number_max:
            values.append(parsed)

    if len(values) < config.pick_count:
        return None

    trimmed = values[-config.pick_count:]
    unique_sorted = tuple(sorted(dict.fromkeys(trimmed)))
    if len(unique_sorted) != config.pick_count:
        return None

    return unique_sorted


def load_history(config: LotteryConfig) -> pd.DataFrame:
    raw_df = pd.read_excel(config.file_path)
    draws: List[Tuple[int, ...]] = []
    for row in raw_df.itertuples(index=False, name=None):
        draw = _extract_draw(row, config)
        if draw is not None:
            draws.append(draw)

    columns = [f"n{i}" for i in range(1, config.pick_count + 1)]
    return pd.DataFrame(draws, columns=columns)


def draws_from_frame(df: pd.DataFrame) -> List[Tuple[int, ...]]:
    return [tuple(sorted(int(value) for value in row if pd.notna(value))) for row in df.to_numpy()]


def history_to_binary(df: pd.DataFrame, config: LotteryConfig) -> pd.DataFrame:
    binary = pd.DataFrame(0, index=np.arange(len(df)), columns=config.number_range)
    for idx, row in enumerate(df.to_numpy()):
        for number in row:
            binary.at[idx, int(number)] = 1
    return binary


def _minmax(values: np.ndarray) -> np.ndarray:
    arr = values.astype(float)
    low = arr.min()
    high = arr.max()
    if np.isclose(low, high):
        return np.zeros_like(arr, dtype=float)
    return (arr - low) / (high - low)


def _number_metrics(draws: Sequence[Tuple[int, ...]], config: LotteryConfig) -> pd.DataFrame:
    numbers = config.number_range
    total_draws = len(draws)
    history_sets = [set(draw) for draw in draws]
    last_draw = history_sets[-1] if history_sets else set()
    previous_draw = history_sets[-2] if len(history_sets) > 1 else set()

    pair_counts = np.zeros((config.number_max + 1, config.number_max + 1), dtype=float)
    for draw in draws[-120:]:
        for i, first in enumerate(draw):
            for second in draw[i + 1:]:
                pair_counts[first, second] += 1
                pair_counts[second, first] += 1

    rows: List[Dict[str, float]] = []
    alpha = 0.08

    for number in numbers:
        appearances = np.array([1 if number in draw else 0 for draw in history_sets], dtype=float)
        positions = np.flatnonzero(appearances)

        frequencies = {}
        for window in (5, 10, 20, 50, 100):
            recent = appearances[-window:]
            frequencies[f"freq_{window}"] = float(recent.mean()) if len(recent) else 0.0

        gap = float(total_draws - 1 - positions[-1]) if len(positions) else float(total_draws)
        mean_gap = float(np.diff(positions).mean()) if len(positions) > 1 else float(total_draws)

        decay_value = 0.0
        for hit in appearances:
            decay_value = ((1 - alpha) * decay_value) + (alpha * hit)

        repeats = 0
        repeat_opportunities = 0
        for index in range(len(appearances) - 1):
            if appearances[index] == 1:
                repeat_opportunities += 1
                if appearances[index + 1] == 1:
                    repeats += 1
        repeat_rate = repeats / repeat_opportunities if repeat_opportunities else 0.0

        cooc_last = float(np.mean([pair_counts[number, other] for other in last_draw])) if last_draw else 0.0

        rows.append(
            {
                "number": number,
                "freq_all": float(appearances.mean()) if len(appearances) else 0.0,
                "gap": gap,
                "mean_gap": mean_gap,
                "overdue": gap / max(mean_gap, 1.0),
                "gap_inverse": 1.0 / (gap + 1.0),
                "decay": decay_value,
                "repeat_rate": repeat_rate,
                "cooc_last": cooc_last,
                "in_last_draw": float(number in last_draw),
                "in_previous_draw": float(number in previous_draw),
                "number_norm": number / config.number_max,
                "parity": float(number % 2 == 0),
                **frequencies,
            }
        )

    metrics = pd.DataFrame(rows)
    for column in [
        "freq_all",
        "freq_5",
        "freq_10",
        "freq_20",
        "freq_50",
        "freq_100",
        "gap",
        "overdue",
        "gap_inverse",
        "decay",
        "repeat_rate",
        "cooc_last",
    ]:
        metrics[f"{column}_norm"] = _minmax(metrics[column].to_numpy())

    return metrics


def score_numbers(draws: Sequence[Tuple[int, ...]], config: LotteryConfig, strategy: Dict[str, object]) -> pd.DataFrame:
    metrics = _number_metrics(draws, config)
    weights: Dict[str, float] = strategy["weights"]  # type: ignore[assignment]

    score = np.zeros(len(metrics), dtype=float)
    for column, weight in weights.items():
        norm_column = f"{column}_norm"
        if norm_column in metrics:
            score += weight * metrics[norm_column].to_numpy()
        elif column in metrics:
            score += weight * metrics[column].to_numpy()

    metrics["score"] = score
    metrics["rank"] = metrics["score"].rank(method="dense", ascending=False).astype(int)
    return metrics.sort_values(["score", "number"], ascending=[False, True]).reset_index(drop=True)


def top_numbers(draws: Sequence[Tuple[int, ...]], config: LotteryConfig, strategy: Dict[str, object]) -> List[int]:
    ranked = score_numbers(draws, config, strategy)
    return sorted(ranked.head(config.pick_count)["number"].astype(int).tolist())


def _historical_profile(draws: Sequence[Tuple[int, ...]], config: LotteryConfig) -> Dict[str, float]:
    even_counts = []
    sums = []
    for draw in draws:
        even_counts.append(sum(number % 2 == 0 for number in draw))
        sums.append(sum(draw))
    return {
        "avg_even": float(np.mean(even_counts)) if even_counts else config.pick_count / 2,
        "avg_sum": float(np.mean(sums)) if sums else 0.0,
        "std_sum": float(np.std(sums)) if sums else 1.0,
    }


def _game_score(game: Sequence[int], ranked_scores: Dict[int, float], profile: Dict[str, float], config: LotteryConfig) -> float:
    numbers = sorted(game)
    base_score = float(sum(ranked_scores[number] for number in numbers))

    even_count = sum(number % 2 == 0 for number in numbers)
    parity_penalty = abs(even_count - profile["avg_even"])

    total_sum = sum(numbers)
    sum_penalty = abs(total_sum - profile["avg_sum"]) / max(profile["std_sum"], 1.0)

    buckets = np.array_split(config.number_range, min(config.pick_count, 6))
    occupied_buckets = 0
    for bucket in buckets:
        bucket_set = set(int(value) for value in bucket)
        if any(number in bucket_set for number in numbers):
            occupied_buckets += 1

    consecutive_count = sum(1 for i in range(len(numbers) - 1) if numbers[i + 1] - numbers[i] == 1)

    return base_score + (0.15 * occupied_buckets) - (0.08 * parity_penalty) - (0.05 * sum_penalty) - (0.04 * consecutive_count)


def generate_candidate_games(
    draws: Sequence[Tuple[int, ...]],
    config: LotteryConfig,
    strategy: Dict[str, object],
    total_games: int = 10,
) -> List[Dict[str, object]]:
    ranked = score_numbers(draws, config, strategy)
    score_map = dict(zip(ranked["number"].astype(int), ranked["score"].astype(float)))
    base = ranked["number"].astype(int).tolist()
    pool = base[: min(len(base), config.pick_count + max(5, config.pick_count // 2))]
    profile = _historical_profile(draws, config)

    candidates = {tuple(sorted(base[: config.pick_count]))}
    rng = np.random.default_rng(42)

    while len(candidates) < total_games * 6 and len(pool) > config.pick_count:
        chosen = rng.choice(pool, size=config.pick_count, replace=False)
        candidates.add(tuple(sorted(int(value) for value in chosen)))

        for replace_count in (1, 2):
            fixed = base[: config.pick_count - replace_count]
            replacements = pool[config.pick_count - replace_count : config.pick_count + 3]
            if len(replacements) >= replace_count:
                chosen_replacements = rng.choice(replacements, size=replace_count, replace=False)
                candidates.add(tuple(sorted(int(value) for value in [*fixed, *chosen_replacements])))

    scored_games = []
    for candidate in candidates:
        scored_games.append(
            {
                "numbers": list(candidate),
                "score": _game_score(candidate, score_map, profile, config),
            }
        )

    scored_games.sort(key=lambda item: item["score"], reverse=True)
    unique_games = []
    seen = set()
    for game in scored_games:
        key = tuple(game["numbers"])
        if key in seen:
            continue
        unique_games.append(game)
        seen.add(key)
        if len(unique_games) >= total_games:
            break

    return unique_games


def backtest_strategy(
    draws: Sequence[Tuple[int, ...]],
    config: LotteryConfig,
    strategy: Dict[str, object],
    start_index: int,
    end_index: int,
) -> Dict[str, object]:
    hits: List[int] = []
    details: List[Dict[str, object]] = []

    for index in range(start_index, end_index):
        prediction = set(top_numbers(draws[:index], config, strategy))
        actual = set(draws[index])
        hit_count = len(prediction & actual)
        hits.append(hit_count)
        details.append(
            {
                "draw_index": index,
                "predicted": sorted(prediction),
                "actual": sorted(actual),
                "hits": hit_count,
            }
        )

    mean_hits = float(np.mean(hits)) if hits else 0.0
    return {
        "strategy": strategy["name"],
        "mean_hits": mean_hits,
        "median_hits": float(np.median(hits)) if hits else 0.0,
        "max_hits": int(max(hits)) if hits else 0,
        "min_hits": int(min(hits)) if hits else 0,
        "std_hits": float(np.std(hits)) if hits else 0.0,
        "hits": hits,
        "details": details,
    }


def evaluate_strategies(
    draws: Sequence[Tuple[int, ...]],
    config: LotteryConfig,
    strategies: Sequence[Dict[str, object]] | None = None,
    validation_size: int = 120,
    test_size: int = 60,
) -> Dict[str, object]:
    strategies = list(strategies or DEFAULT_STRATEGIES)
    total_draws = len(draws)
    min_train = max(80, config.pick_count * 8)
    validation_start = max(min_train, total_draws - (validation_size + test_size))
    validation_end = max(validation_start + 1, total_draws - test_size)
    test_start = validation_end

    validation_results = [
        backtest_strategy(draws, config, strategy, validation_start, validation_end)
        for strategy in strategies
    ]
    validation_results.sort(key=lambda item: (item["mean_hits"], item["median_hits"]), reverse=True)

    best_strategy_name = validation_results[0]["strategy"]
    best_strategy = next(strategy for strategy in strategies if strategy["name"] == best_strategy_name)
    test_result = backtest_strategy(draws, config, best_strategy, test_start, total_draws)

    return {
        "validation_start": validation_start,
        "validation_end": validation_end,
        "test_start": test_start,
        "total_draws": total_draws,
        "validation_results": validation_results,
        "selected_strategy": best_strategy,
        "test_result": test_result,
    }


def pattern_snapshot(df: pd.DataFrame, config: LotteryConfig) -> Dict[str, object]:
    draws = draws_from_frame(df)
    metrics = _number_metrics(draws, config)

    temporal = {
        "top_freq_10": metrics.sort_values("freq_10", ascending=False)[["number", "freq_10"]].head(10).to_dict("records"),
        "top_freq_20": metrics.sort_values("freq_20", ascending=False)[["number", "freq_20"]].head(10).to_dict("records"),
        "mais_atrasados": metrics.sort_values("gap", ascending=False)[["number", "gap", "overdue"]].head(10).to_dict("records"),
    }
    groups = {
        "top_coocorrencia": metrics.sort_values("cooc_last", ascending=False)[["number", "cooc_last"]].head(10).to_dict("records"),
    }
    repetition = {
        "top_repeat_rate": metrics.sort_values("repeat_rate", ascending=False)[["number", "repeat_rate"]].head(10).to_dict("records"),
    }
    sequence = {
        "ultimo_sorteio": list(draws[-1]) if draws else [],
        "sobreposicao_ultimos_2": len(set(draws[-1]) & set(draws[-2])) if len(draws) > 1 else 0,
    }

    return {
        "temporais": temporal,
        "grupos": groups,
        "repeticao": repetition,
        "sequenciais": sequence,
        "metrics": metrics,
    }


def run_full_analysis(
    config: LotteryConfig,
    strategies: Sequence[Dict[str, object]] | None = None,
    validation_size: int = 120,
    test_size: int = 60,
    total_games: int = 10,
) -> Dict[str, object]:
    df = load_history(config)
    draws = draws_from_frame(df)
    evaluation = evaluate_strategies(draws, config, strategies=strategies, validation_size=validation_size, test_size=test_size)
    selected_strategy = evaluation["selected_strategy"]
    ranking = score_numbers(draws, config, selected_strategy)
    candidates = generate_candidate_games(draws, config, selected_strategy, total_games=total_games)

    return {
        "config": config,
        "history": df,
        "draws": draws,
        "binary_history": history_to_binary(df, config),
        "evaluation": evaluation,
        "ranking": ranking,
        "candidates": candidates,
        "patterns": pattern_snapshot(df, config),
        "main_game": candidates[0]["numbers"] if candidates else [],
    }


def save_text_report(result: Dict[str, object], destination: str | Path) -> Path:
    config: LotteryConfig = result["config"]  # type: ignore[assignment]
    evaluation: Dict[str, object] = result["evaluation"]  # type: ignore[assignment]
    ranking: pd.DataFrame = result["ranking"]  # type: ignore[assignment]
    candidates: List[Dict[str, object]] = result["candidates"]  # type: ignore[assignment]
    test_result: Dict[str, object] = evaluation["test_result"]  # type: ignore[assignment]

    lines = [
        f"Projeto: {config.name}",
        f"Base histórica: {len(result['draws'])} sorteios",
        f"Estratégia selecionada: {evaluation['selected_strategy']['name']}",
        f"Backtest teste: média {test_result['mean_hits']:.2f} | mediana {test_result['median_hits']:.2f} | máximo {test_result['max_hits']}",
        f"Referência aleatória: {config.expected_random_hits:.2f}",
        "",
        f"Jogo principal: {candidates[0]['numbers'] if candidates else []}",
        "",
        "Top 10 números por score:",
    ]

    for row in ranking.head(10).itertuples(index=False):
        lines.append(f"- {int(row.number):02d} | score={row.score:.4f} | freq_20={row.freq_20:.3f} | decay={row.decay:.3f}")

    lines.append("")
    lines.append("Top combinações:")
    for index, candidate in enumerate(candidates, start=1):
        lines.append(f"- {index:02d}: {candidate['numbers']} | score={candidate['score']:.4f}")

    target = Path(destination)
    target.write_text("\n".join(lines), encoding="utf-8")
    return target
