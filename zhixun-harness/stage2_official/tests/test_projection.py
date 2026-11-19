import json
import tempfile
import unittest
from pathlib import Path

from stage2_official.rehearse import projection, stratified_batches


def make_row(identity, answer):
    return {"id": identity, "inputs": {"features": [0.0]}, "answer": answer}


def write_rows(directory, rows):
    path = Path(directory) / "train.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_ids(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line)["id"] for line in stream if line.strip()]


class ProjectionSplitTest(unittest.TestCase):
    def test_quota_keeps_the_historical_row_order_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [make_row(f"gas_b01_{i:06d}", "Ethanol" if i <= 6 else "Ethylene")
                    for i in range(1, 13)]
            origin = Path(tmp) / "origin"
            origin.mkdir()
            train = write_rows(origin, rows)
            destination = Path(tmp) / "rounds"
            destination.mkdir()
            files = list(projection(train, destination, 4, "quota"))
            self.assertEqual([len(read_ids(f)) for f in files], [3, 3, 3, 3])
            # Row order is untouched, so the first slice is a single class here on purpose.
            self.assertEqual(set(read_ids(files[0])), {r["id"] for r in rows[:3]})

    def test_stratified_keeps_every_class_in_every_round(self):
        # batch1 lays the gas classes out in contiguous blocks; a row-order split would hand
        # round 1 nothing but Ethanol and round 2 nothing but Ethylene.
        with tempfile.TemporaryDirectory() as tmp:
            rows = ([make_row(f"gas_b01_{i:06d}", "Ethanol") for i in range(1, 7)] +
                    [make_row(f"gas_b01_{i:06d}", "Acetaldehyde") for i in range(7, 13)])
            origin = Path(tmp) / "origin"
            origin.mkdir()
            train = write_rows(origin, rows)
            destination = Path(tmp) / "rounds"
            destination.mkdir()
            files = list(projection(train, destination, 4, "stratified"))
            for number, path in enumerate(files, 1):
                classes = [row["answer"] for row in
                           (json.loads(line) for line in path.open(encoding="utf-8") if line.strip())]
                self.assertIn("Ethanol", classes, f"round {number} lost Ethanol")
                self.assertIn("Acetaldehyde", classes, f"round {number} lost Acetaldehyde")
            self.assertEqual(sorted(len(read_ids(f)) for f in files), [2, 2, 4, 4])
            flat = [i for f in files for i in read_ids(f)]
            self.assertEqual(len(flat), len(rows))
            self.assertEqual(len(set(flat)), len(rows))

    def test_stratified_keeps_every_bearing_in_every_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = ([make_row(f"xjtu_Bearing1_1_s{i:06d}", 100 + i) for i in range(1, 9)] +
                    [make_row(f"xjtu_Bearing3_2_s{i:06d}", 200 + i) for i in range(1, 9)])
            origin = Path(tmp) / "origin"
            origin.mkdir()
            train = write_rows(origin, rows)
            destination = Path(tmp) / "rounds"
            destination.mkdir()
            files = list(projection(train, destination, 4, "stratified"))
            for number, path in enumerate(files, 1):
                owners = {identity.rsplit("_s", 1)[0] for identity in read_ids(path)}
                self.assertEqual(owners, {"xjtu_Bearing1_1", "xjtu_Bearing3_2"},
                                 f"round {number} saw only {owners}")

    def test_stratified_keeps_every_engine_unit_and_subset_in_every_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = ([make_row(f"fd001_train_u0001_c{i:05d}", 190 - i) for i in range(1, 9)] +
                    [make_row(f"fd004_train_u0002_c{i:05d}", 190 - i) for i in range(1, 9)])
            origin = Path(tmp) / "origin"
            origin.mkdir()
            train = write_rows(origin, rows)
            destination = Path(tmp) / "rounds"
            destination.mkdir()
            files = list(projection(train, destination, 4, "stratified"))
            for number, path in enumerate(files, 1):
                units = {identity.split("_c")[0] for identity in read_ids(path)}
                self.assertEqual(units, {"fd001_train_u0001", "fd004_train_u0002"},
                                 f"round {number} saw only {units}")
                subsets = {identity.split("_")[0] for identity in read_ids(path)}
                self.assertEqual(subsets, {"fd001", "fd004"})

    def test_full_hands_the_whole_file_to_every_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [make_row(f"gas_b01_{i:06d}", "Ethanol") for i in range(1, 13)]
            origin = Path(tmp) / "origin"
            origin.mkdir()
            train = write_rows(origin, rows)
            destination = Path(tmp) / "rounds"
            destination.mkdir()
            files = list(projection(train, destination, 3, "full"))
            for path in files:
                self.assertEqual(len(read_ids(path)), len(rows))

    def test_stratified_is_balanced_when_groups_are_lopsided(self):
        # B has two bearings holding 72% of the rows; every round should still see both.
        with tempfile.TemporaryDirectory() as tmp:
            rows = ([make_row(f"xjtu_Bearing1_1_s{i:06d}", 100 + i) for i in range(1, 5)] +
                    [make_row(f"xjtu_Bearing3_1_s{i:06d}", 500 + i) for i in range(1, 37)])
            batches = stratified_batches(rows, 4)
            self.assertEqual(sum(len(batch) for batch in batches), len(rows))
            for batch in batches:
                owners = {row["id"].rsplit("_s", 1)[0] for row in batch}
                self.assertEqual(owners, {"xjtu_Bearing1_1", "xjtu_Bearing3_1"})


if __name__ == "__main__":
    unittest.main()
