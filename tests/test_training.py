from dataclasses import replace

import pytest

from config import Config
from training import train

TINY = replace(Config(), episodes=4, warmup_steps=20, batch_size=8, hidden=16, buffer_size=500,
               eval_every=2, eval_games=1, max_idle_steps=30, milestones=(1,))


def run(tmp_path, name, **kwargs):
    return train(TINY, tmp_path / name / "ckpt", tmp_path / name / "res", plot=False, verbose=False, **kwargs)


def test_train_writes_files_and_reports_evaluations(tmp_path):
    seen = []
    result = run(tmp_path, "a", on_eval=lambda episode, mean: seen.append(episode), final_eval_games=5)
    assert seen == [2, 4]  # eval_every=2 over 4 episodes
    assert len(result["eval_means"]) == 2 and len(result["final_scores"]) == 5
    assert (tmp_path / "a" / "res" / "training_log.csv").exists()
    for name in ("best.pt", "final.pt", "milestones/ep00001.pt"):
        assert (tmp_path / "a" / "ckpt" / name).exists()


def test_same_config_gives_identical_results(tmp_path):
    first = run(tmp_path, "a", final_eval_games=3)
    second = run(tmp_path, "b", final_eval_games=3)
    assert first == second


def test_on_eval_can_stop_training(tmp_path):
    class Stop(Exception):
        pass

    def stop(episode, mean):
        raise Stop

    with pytest.raises(Stop):
        run(tmp_path, "a", on_eval=stop)
    assert (tmp_path / "a" / "ckpt" / "final.pt").exists()  # state is still saved
