#!/usr/bin/env python3
"""
Tensor-based aesthetic analysis of poetry.

Instead of embedding a poem as one vector, embed each LINE separately.
The poem becomes a matrix (n_lines x embedding_dim) with tensor
structure that captures the poem's internal geometry:

- SVD singular values → how many independent threads of meaning
- Running PCA / compression progress → where the "aha" moments are
- Trajectory curvature → where the poem "turns"

Runs on Atlas with sentence-transformers.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy import stats


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ================================================================== #
# Tensor aesthetic metrics                                             #
# ================================================================== #


def effective_rank(matrix):
    """Effective rank via participation ratio of singular values."""
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    S_sq = S**2
    total = S_sq.sum()
    if total < 1e-30:
        return 1.0
    p = S_sq / total
    return float(1.0 / np.sum(p**2))


def compression_progress_per_line(line_embeddings):
    """Compute compression progress as each line is added.

    Returns array of length n_lines where entry i is the
    compression progress after adding line i.
    """
    n, d = line_embeddings.shape
    progress = np.zeros(n)

    if n < 2:
        return progress

    # Running statistics
    running_sum = np.zeros(d)
    running_sq = np.zeros((d, d))

    for t in range(n):
        z = line_embeddings[t]
        running_sum += z
        running_sq += np.outer(z, z)

        if t < 1:
            continue

        # Current covariance
        mean_t = running_sum / (t + 1)
        cov_t = running_sq / (t + 1) - np.outer(mean_t, mean_t)

        # Previous covariance
        mean_prev = (running_sum - z) / t
        cov_prev = (running_sq - np.outer(z, z)) / t - np.outer(mean_prev, mean_prev)

        # Compression progress = change in total coding cost
        # Approximate: change in effective rank (how compressible is the data?)
        try:
            eig_t = np.linalg.eigvalsh(cov_t)
            eig_prev = np.linalg.eigvalsh(cov_prev)

            eig_t = np.maximum(eig_t, 1e-30)
            eig_prev = np.maximum(eig_prev, 1e-30)

            # Log-determinant ratio as compression measure
            logdet_t = np.sum(np.log(eig_t[eig_t > 1e-20]))
            logdet_prev = np.sum(np.log(eig_prev[eig_prev > 1e-20]))

            progress[t] = logdet_prev - logdet_t  # positive if new line compresses
        except Exception:
            progress[t] = 0.0

    return progress


def trajectory_curvature(line_embeddings):
    """Compute curvature at each point of the trajectory.

    Curvature = angle between consecutive direction vectors.
    High curvature = the poem "turns" sharply.
    """
    n = len(line_embeddings)
    if n < 3:
        return np.zeros(n)

    curvature = np.zeros(n)
    for i in range(1, n - 1):
        v1 = line_embeddings[i] - line_embeddings[i - 1]
        v2 = line_embeddings[i + 1] - line_embeddings[i]

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-30 or n2 < 1e-30:
            continue

        cos_angle = np.dot(v1, v2) / (n1 * n2)
        cos_angle = np.clip(cos_angle, -1, 1)
        curvature[i] = np.arccos(cos_angle)  # radians

    return curvature


def analyze_poem(lines, model):
    """Full tensor analysis of a poem's lines."""
    # Embed each line
    embeddings = np.array(
        model.encode(lines, show_progress_bar=False), dtype=np.float32
    )

    n_lines = len(lines)
    result = {
        "n_lines": n_lines,
        "effective_rank": effective_rank(embeddings),
        "mean_curvature": 0.0,
        "max_curvature_line": 0,
        "max_progress_line": 0,
    }

    # Trajectory curvature
    curv = trajectory_curvature(embeddings)
    result["mean_curvature"] = float(curv.mean())
    if n_lines >= 3:
        result["max_curvature_line"] = int(np.argmax(curv[1:-1]) + 1)
        result["curvature_profile"] = curv.tolist()

    # Compression progress
    progress = compression_progress_per_line(embeddings)
    result["max_progress_line"] = int(np.argmax(progress[1:])) + 1
    result["progress_profile"] = progress.tolist()

    # Singular value spectrum
    U, S, Vt = np.linalg.svd(embeddings, full_matrices=False)
    result["singular_values"] = S.tolist()
    result["variance_ratio_top1"] = float(S[0] ** 2 / np.sum(S**2)) if len(S) > 0 else 0

    return result


# ================================================================== #
# Test poems                                                           #
# ================================================================== #


FAMOUS_POEMS = {
    "basho_frog": {
        "lines": [
            "An old silent pond",
            "A frog jumps into the pond",
            "Splash! Silence again",
        ],
        "type": "haiku",
        "expected_turn": 2,  # line 3 is the twist
    },
    "frost_roads": {
        "lines": [
            "Two roads diverged in a yellow wood",
            "And sorry I could not travel both",
            "And be one traveler, long I stood",
            "And looked down one as far as I could",
            "To where it bent in the undergrowth",
            "Then took the other, as just as fair",
            "And having perhaps the better claim",
            "Because it was grassy and wanted wear",
            "Though as for that the passing there",
            "Had worn them really about the same",
            "And both that morning equally lay",
            "In leaves no step had trodden black",
            "Oh, I kept the first for another day",
            "Yet knowing how way leads on to way",
            "I doubted if I should ever come back",
            "I shall be telling this with a sigh",
            "Somewhere ages and ages hence",
            "Two roads diverged in a wood, and I",
            "I took the one less traveled by",
            "And that has made all the difference",
        ],
        "type": "lyric",
        "expected_turn": 15,  # the shift to future tense
    },
    "shakespeare_18": {
        "lines": [
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
        "type": "sonnet",
        "expected_turn": 8,  # the volta at line 9
    },
    "williams_wheelbarrow": {
        "lines": [
            "so much depends upon",
            "a red wheel barrow",
            "glazed with rain water",
            "beside the white chickens",
        ],
        "type": "imagist",
        "expected_turn": 0,  # no clear turn, pure image
    },
    "dickinson_hope": {
        "lines": [
            "Hope is the thing with feathers",
            "That perches in the soul",
            "And sings the tune without the words",
            "And never stops at all",
            "And sweetest in the gale is heard",
            "And sore must be the storm",
            "That could abash the little bird",
            "That kept so many warm",
        ],
        "type": "lyric",
        "expected_turn": 5,  # shift from gentle to storm
    },
}

MUNDANE_TEXTS = {
    "meeting_notice": {
        "lines": [
            "The meeting has been scheduled for Tuesday",
            "Please bring your quarterly reports",
            "We will discuss the budget allocation",
            "Refreshments will be provided",
        ],
        "type": "mundane",
        "expected_turn": -1,
    },
    "instructions": {
        "lines": [
            "Remove the device from the packaging",
            "Connect the power cable to the outlet",
            "Press the power button for three seconds",
            "Wait for the indicator light to turn green",
            "Follow the on-screen setup instructions",
        ],
        "type": "mundane",
        "expected_turn": -1,
    },
    "weather": {
        "lines": [
            "Today will be partly cloudy with a high of seventy-two",
            "Winds from the southwest at ten to fifteen miles per hour",
            "Tonight expect clear skies with a low of fifty-five",
            "Tomorrow will be sunny with temperatures reaching seventy-eight",
        ],
        "type": "mundane",
        "expected_turn": -1,
    },
}


def main():
    log("=" * 70)
    log("POETRY TENSOR ANALYSIS")
    log("=" * 70)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    log("Model loaded")

    os.makedirs("results_aesthetics", exist_ok=True)
    all_results = {}

    # Analyze poems
    log("\nFamous poems:")
    poem_ranks = []
    poem_curvatures = []

    for name, poem in FAMOUS_POEMS.items():
        result = analyze_poem(poem["lines"], model)
        result["name"] = name
        result["type"] = poem["type"]
        result["expected_turn"] = poem["expected_turn"]
        all_results[name] = result

        poem_ranks.append(result["effective_rank"])
        poem_curvatures.append(result["mean_curvature"])

        turn_match = ""
        if poem["expected_turn"] > 0:
            actual_turn = result["max_progress_line"]
            match = abs(actual_turn - poem["expected_turn"]) <= 2
            turn_match = f", turn predict: expected={poem['expected_turn']}, got={actual_turn}, {'HIT' if match else 'MISS'}"

        log(f"  {name:>25s}: eff_rank={result['effective_rank']:.2f}, "
            f"curvature={result['mean_curvature']:.3f}, "
            f"var_top1={result['variance_ratio_top1']:.3f}"
            f"{turn_match}")

    # Analyze mundane text
    log("\nMundane text:")
    mundane_ranks = []
    mundane_curvatures = []

    for name, text in MUNDANE_TEXTS.items():
        result = analyze_poem(text["lines"], model)
        result["name"] = name
        result["type"] = text["type"]
        all_results[name] = result

        mundane_ranks.append(result["effective_rank"])
        mundane_curvatures.append(result["mean_curvature"])

        log(f"  {name:>25s}: eff_rank={result['effective_rank']:.2f}, "
            f"curvature={result['mean_curvature']:.3f}, "
            f"var_top1={result['variance_ratio_top1']:.3f}")

    # Compare poems vs mundane
    log("\n" + "=" * 70)
    log("COMPARISON: Poems vs Mundane")
    log("=" * 70)

    poem_ranks = np.array(poem_ranks)
    mundane_ranks = np.array(mundane_ranks)
    poem_curvatures = np.array(poem_curvatures)
    mundane_curvatures = np.array(mundane_curvatures)

    t_rank, p_rank = stats.ttest_ind(poem_ranks, mundane_ranks)
    t_curv, p_curv = stats.ttest_ind(poem_curvatures, mundane_curvatures)

    log(f"  Effective rank:  poems={poem_ranks.mean():.2f}, "
        f"mundane={mundane_ranks.mean():.2f}, t={t_rank:.3f}, p={p_rank:.4f}")
    log(f"  Mean curvature:  poems={poem_curvatures.mean():.3f}, "
        f"mundane={mundane_curvatures.mean():.3f}, t={t_curv:.3f}, p={p_curv:.4f}")

    # Volta/turn detection accuracy
    log("\nTurn detection (compression progress peak vs expected volta):")
    hits = 0
    total = 0
    for name, poem in FAMOUS_POEMS.items():
        if poem["expected_turn"] > 0:
            total += 1
            actual = all_results[name]["max_progress_line"]
            expected = poem["expected_turn"]
            hit = abs(actual - expected) <= 2
            if hit:
                hits += 1
            log(f"  {name}: expected line {expected}, "
                f"compression peak at line {actual}: {'HIT' if hit else 'MISS'}")

    if total > 0:
        log(f"\n  Turn detection accuracy: {hits}/{total} = {hits/total:.0%}")

    all_results["summary"] = {
        "mean_rank_poems": float(poem_ranks.mean()),
        "mean_rank_mundane": float(mundane_ranks.mean()),
        "t_rank": float(t_rank),
        "p_rank": float(p_rank),
        "mean_curv_poems": float(poem_curvatures.mean()),
        "mean_curv_mundane": float(mundane_curvatures.mean()),
        "t_curvature": float(t_curv),
        "p_curvature": float(p_curv),
        "turn_detection_hits": hits,
        "turn_detection_total": total,
    }

    with open("results_aesthetics/poetry_tensor.json", "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"\nResults saved to results_aesthetics/poetry_tensor.json")


if __name__ == "__main__":
    main()
