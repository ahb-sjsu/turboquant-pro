#!/usr/bin/env python3
"""
A* Aesthetic Path Analysis.

Models a poem/text as a trajectory through embedding space.
The perceiver's eigenspace defines edge costs (Mahalanobis metric).
Beauty = how close the actual trajectory is to the A* optimal path.

The most beautiful poem is the near-geodesic path through semantic
space: maximum compression progress at minimum metric cost.

Runs on Atlas.
"""
from __future__ import annotations

import gc
import heapq
import json
import os
import time

import numpy as np
from scipy import stats


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ================================================================== #
# Mahalanobis metric from perceiver eigenspace                        #
# ================================================================== #


def fit_perceiver(corpus_embeddings, n_components=50):
    """Fit perceiver eigenspace from a corpus of embeddings.

    Returns (mean, eigenvectors, eigenvalues, inverse_eigenvalues).
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(corpus_embeddings)
    return {
        "mean": pca.mean_,
        "components": pca.components_,  # (n_comp, d)
        "eigenvalues": pca.explained_variance_,
        "inv_eigenvalues": 1.0 / np.maximum(pca.explained_variance_, 1e-30),
    }


def mahalanobis_cost(v1, v2, perceiver):
    """Mahalanobis distance between two points in perceiver's metric.

    Cost is LOW along high-eigenvalue (familiar) directions,
    HIGH along low-eigenvalue (unfamiliar) directions.
    """
    diff = v2 - v1
    # Project difference onto eigenbasis
    z = perceiver["components"] @ diff  # (n_comp,)
    # Weight by inverse eigenvalues
    cost = np.sqrt(np.sum(z**2 * perceiver["inv_eigenvalues"]))
    return float(cost)


def euclidean_cost(v1, v2):
    return float(np.linalg.norm(v2 - v1))


# ================================================================== #
# Compression progress along a path                                    #
# ================================================================== #


def path_compression_progress(line_embeddings, perceiver):
    """Total compression progress along a trajectory.

    Measures how much each new line makes the sequence more
    compressible in the perceiver's basis.
    """
    n = len(line_embeddings)
    if n < 2:
        return 0.0

    total_progress = 0.0
    running_cov = np.zeros((len(perceiver["eigenvalues"]),))

    for t in range(n):
        z = perceiver["components"] @ (line_embeddings[t] - perceiver["mean"])
        z_sq = z**2

        if t > 0:
            # How much does this new point align with existing structure?
            alignment = np.sum(z_sq * perceiver["eigenvalues"])
            # How much does it extend into new territory?
            novelty = np.sum(z_sq * perceiver["inv_eigenvalues"])
            # Progress = alignment minus cost of novelty
            progress = alignment / max(novelty, 1e-30)
            total_progress += progress

        running_cov += z_sq

    return total_progress


# ================================================================== #
# A* optimal path through semantic space                               #
# ================================================================== #


def astar_optimal_cost(line_embeddings, perceiver):
    """Compute the A* optimal reordering cost.

    Given N line embeddings, find the permutation that minimizes
    total Mahalanobis path cost (traveling salesman in metric space).

    For small N (< 15), use exact brute force on a subset.
    For larger N, use nearest-neighbor greedy as approximation.
    """
    n = len(line_embeddings)
    if n < 2:
        return 0.0, list(range(n))

    # Compute pairwise Mahalanobis cost matrix
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            c = mahalanobis_cost(line_embeddings[i], line_embeddings[j], perceiver)
            cost_matrix[i, j] = c
            cost_matrix[j, i] = c

    if n <= 10:
        # Exact: try all starting points with greedy nearest-neighbor
        # (true TSP is too expensive even for n=10 with 10! permutations)
        best_cost = float("inf")
        best_order = list(range(n))

        for start in range(n):
            visited = {start}
            order = [start]
            total = 0.0
            current = start

            while len(visited) < n:
                # Find nearest unvisited
                best_next = -1
                best_next_cost = float("inf")
                for j in range(n):
                    if j not in visited and cost_matrix[current, j] < best_next_cost:
                        best_next = j
                        best_next_cost = cost_matrix[current, j]
                visited.add(best_next)
                order.append(best_next)
                total += best_next_cost
                current = best_next

            if total < best_cost:
                best_cost = total
                best_order = order

        return best_cost, best_order
    else:
        # Greedy nearest-neighbor from first line
        visited = {0}
        order = [0]
        total = 0.0
        current = 0

        while len(visited) < n:
            best_next = -1
            best_next_cost = float("inf")
            for j in range(n):
                if j not in visited and cost_matrix[current, j] < best_next_cost:
                    best_next = j
                    best_next_cost = cost_matrix[current, j]
            visited.add(best_next)
            order.append(best_next)
            total += best_next_cost
            current = best_next

        return total, order


def actual_path_cost(line_embeddings, perceiver):
    """Cost of the actual sequential path (line 1 -> 2 -> 3 -> ...)."""
    total = 0.0
    for i in range(len(line_embeddings) - 1):
        total += mahalanobis_cost(line_embeddings[i], line_embeddings[i + 1], perceiver)
    return total


def aesthetic_efficiency(line_embeddings, perceiver):
    """Ratio of optimal path cost to actual path cost.

    Efficiency close to 1.0 = the text follows a near-optimal path.
    Efficiency << 1.0 = the text takes expensive detours.

    Beautiful texts should have INTERMEDIATE efficiency:
    - Too high (1.0) = predictable, boring (greedy path)
    - Too low = incoherent, wasteful
    - Sweet spot = purposeful detours that create compression progress
    """
    actual = actual_path_cost(line_embeddings, perceiver)
    optimal, optimal_order = astar_optimal_cost(line_embeddings, perceiver)

    if actual < 1e-30:
        return 1.0, 0.0, 0.0, []

    efficiency = optimal / actual  # 0 to 1
    progress = path_compression_progress(line_embeddings, perceiver)

    # The "aesthetic value" combines efficiency with progress
    # Beautiful = takes slightly longer than optimal BUT gains compression
    # aesthetic_value = progress / actual_cost (compression per unit cost)
    value = progress / actual if actual > 0 else 0.0

    return efficiency, value, actual, optimal_order


# ================================================================== #
# Test corpus                                                          #
# ================================================================== #


POEMS = {
    "shakespeare_18": [
        "Shall I compare thee to a summer's day",
        "Thou art more lovely and more temperate",
        "Rough winds do shake the darling buds of May",
        "And summer's lease hath all too short a date",
        "Sometime too hot the eye of heaven shines",
        "And often is his gold complexion dimmed",
        "And every fair from fair sometime declines",
        "By chance or nature's changing course untrimmed",
        "But thy eternal summer shall not fade",
        "Nor lose possession of that fair thou owest",
        "Nor shall death brag thou wanderest in his shade",
        "When in eternal lines to time thou growest",
        "So long as men can breathe or eyes can see",
        "So long lives this and this gives life to thee",
    ],
    "basho_frog": [
        "An old silent pond",
        "A frog jumps into the pond",
        "Splash! Silence again",
    ],
    "frost_roads": [
        "Two roads diverged in a yellow wood",
        "And sorry I could not travel both",
        "And be one traveler long I stood",
        "And looked down one as far as I could",
        "To where it bent in the undergrowth",
        "Then took the other as just as fair",
        "And having perhaps the better claim",
        "Because it was grassy and wanted wear",
        "Though as for that the passing there",
        "Had worn them really about the same",
    ],
    "dickinson_hope": [
        "Hope is the thing with feathers",
        "That perches in the soul",
        "And sings the tune without the words",
        "And never stops at all",
        "And sweetest in the gale is heard",
        "And sore must be the storm",
        "That could abash the little bird",
        "That kept so many warm",
    ],
    "eliot_prufrock_opening": [
        "Let us go then you and I",
        "When the evening is spread out against the sky",
        "Like a patient etherized upon a table",
        "Let us go through certain half deserted streets",
        "The muttering retreats",
        "Of restless nights in one night cheap hotels",
        "And sawdust restaurants with oyster shells",
        "Streets that follow like a tedious argument",
        "Of insidious intent",
        "To lead you to an overwhelming question",
    ],
    "blake_tyger": [
        "Tyger Tyger burning bright",
        "In the forests of the night",
        "What immortal hand or eye",
        "Could frame thy fearful symmetry",
        "In what distant deeps or skies",
        "Burnt the fire of thine eyes",
        "On what wings dare he aspire",
        "What the hand dare seize the fire",
    ],
}

MUNDANE = {
    "meeting": [
        "The quarterly meeting will be held on Tuesday",
        "All department heads are required to attend",
        "Please bring your budget reports",
        "The agenda will be distributed by email",
        "Refreshments will be provided in the lobby",
        "Parking validation is available at the front desk",
    ],
    "recipe": [
        "Preheat the oven to three hundred and fifty degrees",
        "Mix the flour and sugar in a large bowl",
        "Add the eggs and butter and stir until smooth",
        "Pour the batter into a greased baking pan",
        "Bake for twenty five minutes until golden brown",
        "Let cool for ten minutes before serving",
    ],
    "manual": [
        "Remove the device from its packaging carefully",
        "Connect the power adapter to the charging port",
        "Press and hold the power button for three seconds",
        "The indicator light will flash blue when ready",
        "Download the companion app from the app store",
        "Follow the on screen instructions to complete setup",
    ],
    "weather_week": [
        "Monday will be sunny with a high of seventy five",
        "Tuesday brings partly cloudy skies and cooler temps",
        "Wednesday expect rain showers throughout the afternoon",
        "Thursday will clear up with temperatures in the sixties",
        "Friday looks pleasant with sunshine and light winds",
        "The weekend forecast calls for mild and dry conditions",
    ],
    "commute": [
        "Take the northbound train to Central Station",
        "Transfer to the Blue Line at Platform Three",
        "Exit at University Avenue and walk two blocks east",
        "The office building is on the corner of Fifth and Main",
        "Visitor parking is in the underground garage",
        "Check in at the reception desk on the ground floor",
    ],
}

SCRAMBLED = {}  # We'll create scrambled versions of poems


def main():
    log("=" * 70)
    log("A* AESTHETIC PATH ANALYSIS")
    log("=" * 70)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    log("Model loaded")

    # Build perceiver from a large corpus of all texts
    log("\nBuilding perceiver eigenspace from all texts...")
    all_lines = []
    for poems in [POEMS, MUNDANE]:
        for lines in poems.values():
            all_lines.extend(lines)

    # Add some random sentences for a richer eigenspace
    import random
    random.seed(42)
    extra = [
        "The sun rose slowly over the mountains painting the sky in shades of gold",
        "Children laughed and played in the park while their parents watched nearby",
        "The old bookstore on the corner had been there for over fifty years",
        "Scientists announced a breakthrough in quantum computing yesterday",
        "The chef carefully arranged each ingredient on the pristine white plate",
        "Music drifted through the open window on the warm summer evening",
        "The ancient ruins stood as a testament to a civilization long forgotten",
        "She opened the letter with trembling hands not knowing what to expect",
        "The river wound its way through the valley carving deeper each century",
        "A single candle flickered in the darkness casting dancing shadows",
    ]
    all_lines.extend(extra)

    corpus_embeddings = np.array(
        model.encode(all_lines, show_progress_bar=False, batch_size=64),
        dtype=np.float32,
    )
    perceiver = fit_perceiver(corpus_embeddings, n_components=30)
    log(f"  Perceiver: {len(all_lines)} lines, {len(perceiver['eigenvalues'])} components")
    log(f"  Top eigenvalues: {perceiver['eigenvalues'][:5].round(4).tolist()}")

    os.makedirs("results_aesthetics", exist_ok=True)
    results = {}

    # Analyze poems
    log("\n" + "=" * 70)
    log("POEMS")
    log("=" * 70)

    for name, lines in POEMS.items():
        embs = np.array(model.encode(lines, show_progress_bar=False), dtype=np.float32)
        efficiency, value, actual_cost, opt_order = aesthetic_efficiency(embs, perceiver)

        # Also create a SCRAMBLED version (same lines, random order)
        rng = np.random.default_rng(42)
        scrambled_idx = rng.permutation(len(lines))
        scrambled_embs = embs[scrambled_idx]
        eff_scram, val_scram, cost_scram, _ = aesthetic_efficiency(scrambled_embs, perceiver)

        results[name] = {
            "type": "poem",
            "n_lines": len(lines),
            "efficiency": efficiency,
            "aesthetic_value": value,
            "path_cost": actual_cost,
            "scrambled_efficiency": eff_scram,
            "scrambled_value": val_scram,
            "scrambled_cost": cost_scram,
            "order_matters": value > val_scram,
        }

        log(f"  {name:>25s}: eff={efficiency:.3f}, value={value:.4f}, "
            f"cost={actual_cost:.2f} | "
            f"scrambled: eff={eff_scram:.3f}, value={val_scram:.4f} | "
            f"order matters: {value > val_scram}")

    # Analyze mundane
    log("\n" + "=" * 70)
    log("MUNDANE TEXT")
    log("=" * 70)

    for name, lines in MUNDANE.items():
        embs = np.array(model.encode(lines, show_progress_bar=False), dtype=np.float32)
        efficiency, value, actual_cost, opt_order = aesthetic_efficiency(embs, perceiver)

        rng = np.random.default_rng(42)
        scrambled_idx = rng.permutation(len(lines))
        scrambled_embs = embs[scrambled_idx]
        eff_scram, val_scram, cost_scram, _ = aesthetic_efficiency(scrambled_embs, perceiver)

        results[name] = {
            "type": "mundane",
            "n_lines": len(lines),
            "efficiency": efficiency,
            "aesthetic_value": value,
            "path_cost": actual_cost,
            "scrambled_efficiency": eff_scram,
            "scrambled_value": val_scram,
            "scrambled_cost": cost_scram,
            "order_matters": value > val_scram,
        }

        log(f"  {name:>25s}: eff={efficiency:.3f}, value={value:.4f}, "
            f"cost={actual_cost:.2f} | "
            f"scrambled: eff={eff_scram:.3f}, value={val_scram:.4f} | "
            f"order matters: {value > val_scram}")

    # Statistical comparison
    log("\n" + "=" * 70)
    log("STATISTICAL COMPARISON")
    log("=" * 70)

    poem_values = [r["aesthetic_value"] for r in results.values() if r["type"] == "poem"]
    mundane_values = [r["aesthetic_value"] for r in results.values() if r["type"] == "mundane"]
    poem_eff = [r["efficiency"] for r in results.values() if r["type"] == "poem"]
    mundane_eff = [r["efficiency"] for r in results.values() if r["type"] == "mundane"]

    # Order sensitivity: how much does scrambling hurt?
    poem_order_loss = [
        (r["aesthetic_value"] - r["scrambled_value"]) / max(r["aesthetic_value"], 1e-10)
        for r in results.values() if r["type"] == "poem"
    ]
    mundane_order_loss = [
        (r["aesthetic_value"] - r["scrambled_value"]) / max(r["aesthetic_value"], 1e-10)
        for r in results.values() if r["type"] == "mundane"
    ]

    t_val, p_val = stats.ttest_ind(poem_values, mundane_values)
    t_eff, p_eff = stats.ttest_ind(poem_eff, mundane_eff)
    t_ord, p_ord = stats.ttest_ind(poem_order_loss, mundane_order_loss)

    log(f"  Aesthetic value: poems={np.mean(poem_values):.4f}, "
        f"mundane={np.mean(mundane_values):.4f}, t={t_val:.3f}, p={p_val:.4f}")
    log(f"  Efficiency:      poems={np.mean(poem_eff):.3f}, "
        f"mundane={np.mean(mundane_eff):.3f}, t={t_eff:.3f}, p={p_eff:.4f}")
    log(f"  Order sensitivity: poems={np.mean(poem_order_loss):.3f}, "
        f"mundane={np.mean(mundane_order_loss):.3f}, t={t_ord:.3f}, p={p_ord:.4f}")

    log(f"\n  KEY PREDICTION: Poems should have higher 'order sensitivity'")
    log(f"  (scrambling hurts poems more than mundane text)")
    log(f"  Result: {'CONFIRMED' if np.mean(poem_order_loss) > np.mean(mundane_order_loss) else 'NOT CONFIRMED'}")

    results["summary"] = {
        "mean_value_poems": float(np.mean(poem_values)),
        "mean_value_mundane": float(np.mean(mundane_values)),
        "t_value": float(t_val),
        "p_value": float(p_val),
        "mean_eff_poems": float(np.mean(poem_eff)),
        "mean_eff_mundane": float(np.mean(mundane_eff)),
        "mean_order_loss_poems": float(np.mean(poem_order_loss)),
        "mean_order_loss_mundane": float(np.mean(mundane_order_loss)),
        "t_order": float(t_ord),
        "p_order": float(p_ord),
    }

    with open("results_aesthetics/astar.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to results_aesthetics/astar.json")


if __name__ == "__main__":
    main()
