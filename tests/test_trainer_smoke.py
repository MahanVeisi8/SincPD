from pathlib import Path
from sincpd.training.trainer import demo_training

def test_demo_training(tmp_path: Path):
    demo_training()
    assert (Path('runs/demo') / 'model.pt').exists()
