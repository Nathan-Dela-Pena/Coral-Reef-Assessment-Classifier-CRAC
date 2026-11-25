"""The grouped split is the thing most worth protecting in Phase 2.

A random split leaks near-duplicate frames from one survey across the train/test
boundary and inflates test accuracy. These tests fail if that protection is lost.
"""

import pandas as pd
import pytest

from crac.model import dataset


def manifest(n_sites=10, per_site=6):
    rows = []
    for s in range(n_sites):
        for i in range(per_site):
            rows.append({
                "image_path": f"img/site{s}_{i}.jpg",
                "label": dataset.LABELS[i % len(dataset.LABELS)],
                "site_id": f"site{s}",
                "captured_on": "2024-01-01",
            })
    return pd.DataFrame(rows)


def test_split_keeps_sites_in_exactly_one_split():
    train, val, test = dataset.split_by_site(manifest(), seed=0)
    groups = [set(d["site_id"]) for d in (train, val, test)]
    assert groups[0] & groups[1] == set()
    assert groups[0] & groups[2] == set()
    assert groups[1] & groups[2] == set()


def test_split_covers_every_row_exactly_once():
    df = manifest()
    train, val, test = dataset.split_by_site(df, seed=0)
    assert len(train) + len(val) + len(test) == len(df)
    assert set(pd.concat([train, val, test])["image_path"]) == set(df["image_path"])


def test_split_is_deterministic_for_a_seed():
    a = dataset.split_by_site(manifest(), seed=7)[2]["site_id"].tolist()
    b = dataset.split_by_site(manifest(), seed=7)[2]["site_id"].tolist()
    assert a == b


def test_different_seeds_can_give_different_test_sites():
    seeds = {tuple(sorted(set(dataset.split_by_site(manifest(), seed=s)[2]["site_id"])))
             for s in range(6)}
    assert len(seeds) > 1


def test_split_drops_its_internal_grouping_column():
    for part in dataset.split_by_site(manifest(), seed=0):
        assert "_group" not in part.columns


def test_read_manifest_rejects_labels_outside_the_taxonomy(tmp_path):
    df = manifest(n_sites=3, per_site=2)
    df.loc[0, "label"] = "slightly_sad"
    path = tmp_path / "m.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="not in the taxonomy"):
        dataset.read_manifest(path)


def test_class_weights_favour_the_rare_class():
    df = manifest(n_sites=4, per_site=4)
    df.loc[df.index[:12], "label"] = "healthy"        # heavy imbalance
    ds = dataset.ReefImageDataset(df)
    weights = dict(zip(dataset.LABELS, ds.class_weights(), strict=True))
    assert weights["healthy"] < max(w for w in weights.values())


@pytest.mark.parametrize("name,expected", [
    ("site=great_barrier_north_2019-03-04_0001.jpg", ("great_barrier_north", "2019-03-04")),
    ("site=moorea_0007.png", ("moorea", None)),
    ("IMG_4821.jpg", (None, None)),
])
def test_parse_image_name(name, expected):
    assert dataset.parse_image_name(name) == expected
