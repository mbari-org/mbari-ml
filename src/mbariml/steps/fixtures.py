"""``mbariml fixtures``: link a fixed camera's boxes into individual animals.

A fixed (time-lapse) camera photographs the same sponge, coral or anemone in
every frame, so a database of half a million boxes may hold only a few hundred
animals that never move. Reviewing them box by box means confirming the same
sponge thousands of times. This command links boxes frame to frame by position;
``mbariml review``'s "By animal" sort then shows one tile per animal, and
verifying or relabelling that tile applies to every one of its boxes.

Linking ignores the label on purpose: the detector often flickers between two
names for one animal (one sponge called three things across the year), and
putting those boxes in one animal is exactly what lets one decision settle
them. Boxes must overlap (IoU >= ``--iou``) a fixture's running reference box,
an exponential moving average of its matches so slow growth or drift is
followed. A fixture lapses after ``max(--retire, 2 x its age)`` frames without
a match, so a long-lived animal survives occlusions while a short visitor drops
out quickly. Only fixtures seen in at least ``--min-frames`` frames are kept;
everything else (passing animals) stays with ordinary box-by-box review.

Two clean-ups follow, both by appearance (stored embeddings): a box unlike its
fixture (``--min-typicality``) is something else in that spot and is dropped,
and a group of boxes labelled differently from the rest whose mean look is
below ``--split-similarity`` to the rest is a different animal using the same
spot -- split off as its own fixture if seen in ``--min-frames`` frames, else
dropped. Similar-looking groups stay: that's one animal under two names.
Then pieces of one animal that linking left apart (a long absence, a coral
retracting its polyps) are joined: same spot, never in the same frame, alike
(``--merge-similarity``) -- including short pieces that would otherwise have
been dropped.

Frames are ordered by video and frame number, then by image name -- time-lapse
file names sort chronologically. One database should hold one camera: boxes are
linked by position, which only means something within one field of view.

Measured on the Coral Observatory 2023 full set (575,640 boxes, 7,479 hourly
frames, 6720 x 4480): see the CHANGELOG entry for this command.

Writes two tables (replacing earlier runs); ``predictions`` is not touched:

    fixtures      roi_index, fixture, typicality  (one row per linked box)
    fixture_info  fixture, rep_roi_index, n, first_image, last_image

``typicality`` is cosine similarity to the fixture's mean embedding (stored
embeddings from ``mbariml embed``, centred on the whole database), so the
review GUI can open an animal least-typical-first: frames where something else
sits in that spot come before thousands of identical ones. The representative
is the most typical box. Without embeddings, the most confident box represents
the fixture and typicality is NULL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import typer
from tqdm import tqdm

from mbariml import db
from mbariml.gui.query_service import _embedding_matrix
from mbariml.logging_utils import get_logger

app = typer.Typer(help="Link a fixed camera's boxes into individual animals for `review`'s By animal sort.")
logger = get_logger(__name__)

SCHEMA_SQL = """
CREATE OR REPLACE TABLE fixtures (roi_index INTEGER, fixture INTEGER, typicality DOUBLE);
CREATE OR REPLACE TABLE fixture_info (
    fixture INTEGER, rep_roi_index INTEGER, n INTEGER, first_image TEXT, last_image TEXT);
"""


def _iou(boxes: np.ndarray, refs: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(boxes[:, None, 2], refs[None, :, 2]) - np.maximum(boxes[:, None, 0], refs[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(boxes[:, None, 3], refs[None, :, 3]) - np.maximum(boxes[:, None, 1], refs[None, :, 1]), 0, None)
    inter = ix * iy
    area_b = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))[:, None]
    area_r = ((refs[:, 2] - refs[:, 0]) * (refs[:, 3] - refs[:, 1]))[None, :]
    return inter / np.maximum(area_b + area_r - inter, 1e-9)


def link(boxes: np.ndarray, frame: np.ndarray, *, iou_min: float, alpha: float, retire: int) -> np.ndarray:
    """Track number of every box. *boxes* (n, 4) sorted by *frame* (n,) ascending.

    Each frame's boxes are matched to the live tracks greedily, best overlap
    first (one box per track, one track per box); unmatched boxes start tracks.
    """
    n = len(boxes)
    out = np.full(n, -1, dtype=np.int64)
    ref = np.zeros((max(1024, n // 8), 4))
    first = np.zeros(len(ref), dtype=np.int64)
    last = np.zeros(len(ref), dtype=np.int64)
    active = np.zeros(0, dtype=np.int64)
    n_tracks = 0
    starts = np.flatnonzero(np.r_[True, frame[1:] != frame[:-1]])
    ends = np.r_[starts[1:], n]
    for s, e in tqdm(zip(starts, ends), total=len(starts), desc="linking", unit="frame"):
        f = frame[s]
        if active.size:
            active = active[(f - last[active]) <= np.maximum(retire, 2 * (last[active] - first[active]))]
        b = boxes[s:e]
        taken = np.zeros(len(b), dtype=bool)
        if active.size:
            iou = _iou(b, ref[active])
            rows, cols = np.nonzero(iou >= iou_min)
            used = set()
            for k in np.argsort(-iou[rows, cols], kind="stable"):
                r, c = rows[k], cols[k]
                if taken[r] or c in used:
                    continue
                taken[r] = True
                used.add(c)
                t = active[c]
                ref[t] = (1 - alpha) * ref[t] + alpha * b[r]
                last[t] = f
                out[s + r] = t
        new = np.flatnonzero(~taken)
        if new.size:
            if n_tracks + new.size > len(ref):
                grow = max(len(ref), new.size)
                ref = np.vstack([ref, np.zeros((grow, 4))])
                first, last = (np.concatenate([a, np.zeros(grow, dtype=np.int64)]) for a in (first, last))
            ids = np.arange(n_tracks, n_tracks + new.size)
            ref[ids], first[ids], last[ids] = b[new], f, f
            out[s + new] = ids
            active = np.concatenate([active, ids])
            n_tracks += new.size
    return out


def _typicality(track: np.ndarray, matrix: np.ndarray | None, rows: np.ndarray | None) -> np.ndarray:
    """Cosine of each box to its fixture's mean embedding; NaN without one (or off any fixture)."""
    out = np.full(len(track), np.nan)
    if matrix is None:
        return out
    has = (rows >= 0) & (track >= 0)
    sums = np.zeros((int(track.max()) + 1, matrix.shape[1]), dtype=np.float32)
    np.add.at(sums, track[has], matrix[rows[has]])
    sums /= np.maximum(np.linalg.norm(sums, axis=1, keepdims=True), 1e-12)
    out[has] = np.einsum("ij,ij->i", matrix[rows[has]], sums[track[has]])
    return out


def _split_label_groups(
    track: np.ndarray, lab: np.ndarray, matrix: np.ndarray, rows: np.ndarray, min_similarity: float, min_frames: int
) -> tuple[np.ndarray, int, int]:
    """Tease apart fixtures that hold more than one animal.

    Within a fixture, a group of boxes carrying another label is either the
    detector (or a reviewer) calling one animal two names, or a different
    animal in the same spot. Their looks tell them apart: measured against
    the Coral Observatory play set's human labels, a group whose mean
    embedding was below 0.9 cosine to the fixture's main label group was a
    different kind of animal 92-100% of the time (a Chorilia using a Munida's
    crevice, a Pandalus on a sponge), while groups above it were mostly one
    animal under two names (Porifera / "marine organism", Hexactinellida /
    Farrea, one coral reviewed as both Isididae and Paragorgia). Groups below
    *min_similarity* become their own fixture if big enough, else leave.

    Returns ``(track, groups split off, boxes left for box-by-box review)``.
    """
    track = track.copy()
    next_track = int(track.max()) + 1
    n_split = n_out = 0
    on = np.flatnonzero(track >= 0)
    order = on[np.argsort(track[on], kind="stable")]
    bounds = np.flatnonzero(np.diff(track[order])) + 1
    for members in np.split(order, bounds):
        labels, counts = np.unique(lab[members], return_counts=True)
        if len(labels) < 2:
            continue
        main = members[lab[members] == labels[np.argmax(counts)]]
        main_rows = rows[main][rows[main] >= 0]
        if not len(main_rows):
            continue
        centre = matrix[main_rows].mean(axis=0)
        centre /= np.linalg.norm(centre) or 1.0
        for label in labels:
            group = members[lab[members] == label]
            group_rows = rows[group][rows[group] >= 0]
            if label == labels[np.argmax(counts)] or not len(group_rows):
                continue
            look = matrix[group_rows].mean(axis=0)
            if float(look @ centre) / (np.linalg.norm(look) or 1.0) >= min_similarity:
                continue  # same animal, another name: one decision settles it
            if len(group) >= min_frames:
                track[group] = next_track
                next_track += 1
                n_split += 1
            else:
                track[group] = -1
                n_out += len(group)
    return track, n_split, n_out


def _merge_fragments(
    track: np.ndarray, frame: np.ndarray, boxes: np.ndarray, matrix: np.ndarray, rows: np.ndarray,
    min_similarity: float, min_frames: int, min_overlap: float = 0.6, max_together: float = 0.05,
) -> tuple[np.ndarray, int, int]:
    """Join pieces of one animal that linking left apart.

    Frame-to-frame linking splits an animal when it vanishes for longer than
    the retire window, or changes size too much (a soft coral retracting its
    polyps), and drops pieces seen in fewer than *min_frames* frames. Two
    tracks are one animal when they sit in the same spot (median boxes overlap
    by *min_overlap* of the smaller), are (almost) never in the same frame,
    and look alike (mean embeddings >= *min_similarity*). Big tracks are
    merged with each other first; then small pieces join the big one whose
    spot they sit in. On the Coral Observatory set one anemone was three
    fixtures and a Heteropolypus that retracted was two.

    Returns ``(track, big tracks merged away, small-piece boxes attached)``.
    """
    on = track >= 0
    ids, inverse = np.unique(track[on], return_inverse=True)
    if len(ids) < 2:
        return track, 0, 0
    size = np.bincount(inverse)
    idx_on = np.flatnonzero(on)
    med = np.stack([pd.Series(boxes[idx_on, k]).groupby(inverse).median().to_numpy() for k in range(4)], axis=1)
    has = rows[idx_on] >= 0
    look = np.zeros((len(ids), matrix.shape[1]), dtype=np.float32)
    np.add.at(look, inverse[has], matrix[rows[idx_on][has]])
    look /= np.maximum(np.linalg.norm(look, axis=1, keepdims=True), 1e-12)
    frames_of = pd.Series(frame[idx_on]).groupby(inverse).agg(set).to_list()

    def overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:  # intersection over the smaller box
        ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
        iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        return ix * iy / np.maximum(np.minimum(area_a[:, None], area_b[None, :]), 1e-9)

    big = np.flatnonzero(size >= min_frames)
    root = np.arange(len(ids))
    members = {int(i): set(frames_of[i]) for i in big}

    def find(i: int) -> int:
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i

    # 1. big with big, most alike first; "never together" is checked against
    # whole merged groups, so a chain can't pull in an animal that coexists
    # with any member.
    sim_bb = look[big] @ look[big].T
    a, b = np.nonzero(np.triu((overlap(med[big], med[big]) >= min_overlap) & (sim_bb >= min_similarity), 1))
    merged = 0
    for k in np.argsort(-sim_bb[a, b], kind="stable"):
        ra, rb = find(int(big[a[k]])), find(int(big[b[k]]))
        if ra == rb:
            continue
        if len(members[ra] & members[rb]) <= max_together * min(len(members[ra]), len(members[rb])):
            keep, drop = (ra, rb) if len(members[ra]) >= len(members[rb]) else (rb, ra)
            root[drop] = keep
            members[keep] |= members.pop(drop)
            merged += 1

    # 2. small pieces join the most alike big group whose spot they sit in.
    attached = 0
    small = np.flatnonzero(size < min_frames)
    for s in range(0, len(small), 4096):
        chunk = small[s:s + 4096]
        ok = (overlap(med[chunk], med[big]) >= min_overlap) & ((look[chunk] @ look[big].T) >= min_similarity)
        score = np.where(ok, look[chunk] @ look[big].T, -np.inf)
        for r in np.flatnonzero(ok.any(axis=1)):
            for c in np.argsort(-score[r]):
                if not np.isfinite(score[r, c]):
                    break
                g = find(int(big[c]))
                piece = frames_of[chunk[r]]
                if len(piece & members[g]) <= max_together * len(piece):
                    root[chunk[r]] = g
                    members[g] |= piece
                    attached += int(size[chunk[r]])
                    break

    new_ids = ids[[find(i) for i in range(len(ids))]]
    out = track.copy()
    out[on] = new_ids[inverse]
    return out, merged, attached


@app.command()
def fixtures(
    db_path: str = typer.Argument(..., help="Path to the DuckDB database (one fixed camera)."),
    iou: float = typer.Option(0.5, help="Overlap with a fixture's reference box needed to join it."),
    min_frames: int = typer.Option(20, help="Keep fixtures seen in at least this many frames; the rest are passers-by."),
    retire: int = typer.Option(24, help="Frames a fixture may go unseen (at least; long-lived ones get 2 x their age)."),
    alpha: float = typer.Option(0.3, help="Weight of each new match in the fixture's moving reference box."),
    min_typicality: float = typer.Option(
        0.5, help="Boxes less like their fixture than this (cosine to its mean embedding) are something else in "
                  "that spot, e.g. a crab on a sponge, and are left for box-by-box review."),
    merge_similarity: float = typer.Option(
        0.9, help="Pieces of one animal that linking left apart (same spot, never in the same frame) are joined "
                  "when their mean embeddings are at least this similar."),
    split_similarity: float = typer.Option(
        0.9, help="A group of boxes labelled differently from the rest of their fixture is split off unless its "
                  "mean embedding is at least this similar to the rest's (then it's one animal under two names)."),
) -> None:
    """Link boxes frame to frame by position into animals, for review's "By animal" sort."""
    with db.connect(db_path, must_exist=True) as conn:
        data = conn.execute("""
            SELECT roi_index, x_min, y_min, x_max, y_max, confidence,
                   COALESCE(NULLIF(new_label, ''), label, '') AS lab,
                   dense_rank() OVER (ORDER BY video_path NULLS FIRST, frame_number NULLS FIRST, image_path) AS frame,
                   image_path
            FROM predictions ORDER BY frame, roi_index""").fetchnumpy()
        if len(data["roi_index"]) == 0:
            logger.warning("No boxes in %s; nothing to link.", db_path)
            return
        roi = np.asarray(data["roi_index"], dtype=np.int64)
        boxes = np.stack([np.asarray(data[k], dtype=float) for k in ("x_min", "y_min", "x_max", "y_max")], axis=1)
        frame = np.asarray(data["frame"], dtype=np.int64)
        n_frames = int(frame.max())
        logger.info("Linking %d box(es) across %d frame(s)", len(roi), n_frames)
        track = link(boxes, frame, iou_min=iou, alpha=alpha, retire=retire)

        matrix = rows = None
        dim_row = conn.execute("SELECT len(embedding) FROM predictions WHERE embedding IS NOT NULL LIMIT 1").fetchone()
        if dim_row is not None:
            logger.info("Scoring typicality from stored embeddings")
            indices, matrix = _embedding_matrix(conn, " WHERE embedding IS NOT NULL", [], dim_row[0])
            matrix -= matrix.mean(axis=0)
            matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
            row_of = np.full(int(max(indices.max(), roi.max())) + 1, -1, dtype=np.int64)
            row_of[indices] = np.arange(len(indices))
            rows = row_of[roi]

        # A box that doesn't look like its fixture is something else in that
        # spot -- on the Coral Observatory set, a crab sitting on a sponge
        # (typicality 0.13-0.55 against the sponge's own 0.74-0.97). It goes
        # back to box-by-box review rather than inheriting the sponge's label.
        typicality = _typicality(track, matrix, rows)
        intruder = typicality < min_typicality  # NaN (no embedding) compares False: kept
        track = np.where(intruder, -1, track)
        n_split = n_out = n_merged = n_attached = 0
        if matrix is not None:
            track, n_split, n_out = _split_label_groups(
                track, np.asarray(data["lab"], dtype=object), matrix, rows, split_similarity, min_frames
            )
            track, n_merged, n_attached = _merge_fragments(
                track, frame, boxes, matrix, rows, merge_similarity, min_frames
            )
            typicality = _typicality(track, matrix, rows)
        sizes = np.bincount(track[track >= 0], minlength=int(track.max()) + 1)
        keep = (track >= 0) & (sizes[np.maximum(track, 0)] >= min_frames)
        _, fixture = np.unique(track[keep], return_inverse=True)  # renumber kept tracks 0..k-1
        k_roi, k_frame = roi[keep], frame[keep]
        k_conf = np.nan_to_num(np.asarray(data["confidence"], dtype=float)[keep], nan=-1.0)
        image = np.asarray(data["image_path"], dtype=object)[keep]
        typicality = typicality[keep]
        n_fix = int(fixture.max()) + 1 if fixture.size else 0
        if not n_fix:
            logger.warning("No fixture seen in %d or more frames -- is this one fixed camera?", min_frames)

        # Representative: most typical box, else most confident.
        key = np.where(np.isnan(typicality), k_conf - 10.0, typicality)
        order = np.lexsort((-key, fixture))
        first_of = np.r_[0, np.flatnonzero(np.diff(fixture[order])) + 1]
        rep = k_roi[order[first_of]]
        by_frame = np.lexsort((k_frame, fixture))
        bounds = np.r_[np.flatnonzero(np.diff(fixture[by_frame])) + 1, len(by_frame)]
        first_image = image[by_frame[np.r_[0, bounds[:-1]]]]
        last_image = image[by_frame[bounds - 1]]
        n = np.bincount(fixture, minlength=n_fix)

        conn.execute(SCHEMA_SQL)
        conn.execute(
            "INSERT INTO fixtures SELECT unnest(?::INTEGER[]), unnest(?::INTEGER[]), unnest(?::DOUBLE[])",
            [k_roi.tolist(), fixture.tolist(), [None if np.isnan(v) else float(v) for v in typicality]],
        )
        conn.execute(
            "INSERT INTO fixture_info SELECT unnest(?::INTEGER[]), unnest(?::INTEGER[]), unnest(?::INTEGER[]), "
            "unnest(?::TEXT[]), unnest(?::TEXT[])",
            [list(range(n_fix)), rep.tolist(), n.tolist(), first_image.tolist(), last_image.tolist()],
        )
    logger.info(
        "%d fixture(s) seen in >= %d frames hold %d of %d box(es) (%.1f%%); the other %d stay box-by-box "
        "(%d of them unlike the fixture in their spot). %d differently-labelled group(s) that looked like "
        "another animal were split off as their own fixture, %d box(es) in smaller ones left box-by-box. "
        "%d fixture(s) were pieces of another and joined it; %d box(es) in short pieces joined the animal in their spot.",
        n_fix, min_frames, len(k_roi), len(roi), 100 * len(k_roi) / len(roi), len(roi) - len(k_roi),
        int(intruder.sum()), n_split, n_out, n_merged, n_attached,
    )


if __name__ == "__main__":
    app()
