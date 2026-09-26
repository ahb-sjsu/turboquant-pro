# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""What is this representation currently certified to be used for?

An agent or a query planner holding a compressed artifact needs an answer to
"what may I safely do with this", not a list of actions it could attempt. The
answer exists already, scattered: a rank certificate says a statement held for
an observer under an environment (`rank_certificate`), its validity section
says whether that still applies (`validity`, issue #177), and the
compatibility matrix says whether a code built for one reader serves another
(`refinement.compatibility_matrix`, issue #179). This module reads those
artifacts and states the answer in three lists (issue #178).

- **certified**: a consumer named by a contract whose certificate is about
  *this* artifact, passes, and is still applicable.
- **conditional**: the certificate is about this artifact and passes, but is
  no longer applicable (the observer's read geometry moved, or the data left
  the calibration's coverage or its strata), or its applicability could not be
  decided (UNCHECKED, or INCONCLUSIVE on too few rows: only VALID certifies),
  or the reader was never certified and a
  certified code is only near enough for it. Each carries what would make it
  certified.
- **not certified**: a consumer with no certificate about this artifact at
  all, or one whose certificate fails.

Nothing here asserts a capability the artifacts do not carry. A certificate
for a different artifact is reported as such rather than counted, matched by
the sha256 the certificate recorded for its own inputs, so a capability list
cannot be borrowed from another corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CERTIFIED, CONDITIONAL, NOT_CERTIFIED = "certified", "conditional", "not_certified"

__all__ = [
    "CERTIFIED",
    "CONDITIONAL",
    "NOT_CERTIFIED",
    "Capability",
    "CapabilityReport",
    "capabilities",
]


def _label(cert: dict, path: str) -> str:
    """What a certificate is a capability *for*, in the words the artifacts
    use: the observer's consumers, else the declared task, else the reference
    provider, else the file."""
    obs = cert.get("observer") or {}
    names = obs.get("consumers") or []
    if obs.get("observer") and names:
        return f"{obs['observer']}: {', '.join(names)}"
    if obs.get("observer"):
        return str(obs["observer"])
    task = cert.get("task") or {}
    if task.get("target"):
        return f"{task.get('kind', 'task')}: {task['target']}"
    ref = cert.get("reference") or {}
    if ref.get("provider"):
        return f"read operator: {ref['provider']}"
    return path


@dataclass
class Capability:
    name: str
    status: str
    certificate: str
    reason: str
    observer_sha256: str | None = None
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "certificate": self.certificate,
            "reason": self.reason,
            "observer_sha256": self.observer_sha256,
            **({"detail": self.detail} if self.detail else {}),
        }


@dataclass
class CapabilityReport:
    artifact: str
    artifact_sha256: str
    items: list
    other_artifacts: list = field(default_factory=list)

    def by_status(self, status: str) -> list:
        return [c for c in self.items if c.status == status]

    def as_dict(self) -> dict:
        return {
            "schema": "turboquant-pro/capability-report",
            "schema_version": 1,
            "artifact": self.artifact,
            "artifact_sha256": self.artifact_sha256,
            "certified": [c.as_dict() for c in self.by_status(CERTIFIED)],
            "conditional": [c.as_dict() for c in self.by_status(CONDITIONAL)],
            "not_certified": [c.as_dict() for c in self.by_status(NOT_CERTIFIED)],
            "certificates_for_other_artifacts": list(self.other_artifacts),
        }

    def explain(self) -> str:
        L = [f"CAPABILITIES of {self.artifact}", f"  sha256 {self.artifact_sha256}", ""]
        for status, head in (
            (CERTIFIED, "certified"),
            (CONDITIONAL, "conditional"),
            (NOT_CERTIFIED, "not certified"),
        ):
            rows = self.by_status(status)
            L.append(f"{head}:")
            if not rows:
                L.append("  <none>")
            for c in rows:
                L.append(f"  {c.name}")
                L.append(f"    {c.reason}")
            L.append("")
        if self.other_artifacts:
            L.append(
                f"{len(self.other_artifacts)} certificate(s) are about a different "
                "artifact and were not counted:"
            )
            for o in self.other_artifacts:
                L.append(f"  {o['certificate']} (inputs {o['recorded_sha256']})")
        return "\n".join(L).rstrip()


def _sha256_array(a: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(a)).hexdigest()


def capabilities(
    artifact: np.ndarray,
    certificates: list,
    *,
    contracts: list | None = None,
    data: np.ndarray | None = None,
    queries: np.ndarray | None = None,
    artifact_path: str = "<array>",
) -> CapabilityReport:
    """What this artifact is certified for, from the certificates that are
    about it.

    Args:
        artifact: the original vectors the certificates were issued over.
        certificates: ``(path, document)`` pairs.
        contracts: observer contracts to check for. One whose hash no
            certificate names is reported as not certified, which is the
            useful half of the answer: it says what to certify next.
        data: a current sample, so each certificate's validity section can be
            checked (issue #177); without it applicability is unchecked and
            every passing certificate is reported as conditional, because an
            unchecked certificate is not a promise.
        queries: query sample for rebuilding a retrieval consumer's operator.
    """
    from turboquant_pro.validity import check_validity

    sha = _sha256_array(np.asarray(artifact))
    items: list[Capability] = []
    other: list[dict] = []
    seen_observers: set[str] = set()

    for path, cert in certificates:
        recorded = ((cert.get("inputs") or {}).get("original") or {}).get("sha256")
        if recorded != sha:
            other.append({"certificate": path, "recorded_sha256": recorded})
            continue
        name = _label(cert, path)
        obs_sha = (cert.get("observer") or {}).get("sha256")
        if obs_sha:
            seen_observers.add(obs_sha)
        if not cert.get("passed", False):
            items.append(
                Capability(
                    name=name,
                    status=NOT_CERTIFIED,
                    certificate=path,
                    reason="the certificate does not pass: "
                    + str(cert.get("interpretation", "no interpretation recorded")),
                    observer_sha256=obs_sha,
                )
            )
            continue
        contract = None
        if obs_sha and contracts:
            contract = next((c for c in contracts if c.digest() == obs_sha), None)
        v = check_validity(cert, contract=contract, data=data, queries=queries)
        if data is None:
            items.append(
                Capability(
                    name=name,
                    status=CONDITIONAL,
                    certificate=path,
                    reason=(
                        "the certificate passes, but nothing checked whether it "
                        "still applies; pass a current sample to check the "
                        "observer's read geometry and the data's coverage"
                    ),
                    observer_sha256=obs_sha,
                    detail={"validity": v},
                )
            )
        elif v["status"] == "VALID":
            items.append(
                Capability(
                    name=name,
                    status=CERTIFIED,
                    certificate=path,
                    reason=(
                        "the certificate passes and still applies "
                        f"(validity {v['status']})"
                    ),
                    observer_sha256=obs_sha,
                    detail={"validity": v},
                )
            )
        else:
            items.append(
                Capability(
                    name=name,
                    status=CONDITIONAL,
                    certificate=path,
                    reason=(
                        f"{v['status']}: {v['reason']}; action {v['action']}"
                        if v["status"] == "STALE"
                        else (
                            f"{v['status']}: {v['reason']}"
                            if v["reason"]
                            else f"{v['status']}: the certificate recorded nothing "
                            "the sample could be checked against"
                        )
                    ),
                    observer_sha256=obs_sha,
                    detail={"validity": v},
                )
            )

    for c in contracts or []:
        if c.digest() in seen_observers:
            continue
        items.append(
            Capability(
                name=f"{c.observer}: {', '.join(x.label for x in c.consumers)}",
                status=NOT_CERTIFIED,
                certificate="<none>",
                reason=(
                    "no certificate about this artifact names this observer; "
                    "certify it with `tqp certify --observer` to add the capability"
                ),
                observer_sha256=c.digest(),
            )
        )

    return CapabilityReport(
        artifact=artifact_path, artifact_sha256=sha, items=items, other_artifacts=other
    )
