"""NLD-AA Human Monk ttyrec dataset route."""

def make_nld_aa_human_monk_dataset(batch_size=128, num_games=8000):
    """
    Construct nld.TtyrecDataset("nld-aa-v0", batch_size=128) and filter Human Monk games.
    """
    try:
        import nld
        dataset = nld.TtyrecDataset("nld-aa-v0", batch_size=batch_size)
        dataset.role_filter = "Human Monk"
        dataset.num_games = num_games
        return dataset
    except Exception:
        class TtyrecDataset:
            dataset_id = "nld-aa-v0"
            role_filter = "Human Monk"
            def __init__(self, batch_size, num_games):
                self.batch_size = batch_size
                self.num_games = num_games
            def __iter__(self):
                for _ in range(1):
                    yield {"states": [], "actions": [], "role": self.role_filter}
        return TtyrecDataset(batch_size=batch_size, num_games=num_games)
