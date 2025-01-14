from mmengine.config import Config, ConfigDict
from mmengine.runner import Runner
from mmdet.utils import register_all_modules
from mmengine.logging import MMLogger

from pathlib import Path
from src.indecs.config import Config as app_config

def main():
    # Register all modules for MMDetection
    register_all_modules()

    # Load config
    script_path = Path(__file__).resolve()
    config_path = script_path.parent.parent / 'config' / 'mmdet_config.py'
    cfg = Config.fromfile(str(config_path))

    # Convert old format settings to new MMEngine format
    cfg.train_dataloader = ConfigDict(
        batch_size=cfg.data.samples_per_gpu,
        num_workers=cfg.data.workers_per_gpu,
        dataset=cfg.data.train
    )

    cfg.val_dataloader = ConfigDict(
        batch_size=cfg.data.samples_per_gpu,
        num_workers=cfg.data.workers_per_gpu,
        dataset=cfg.data.val
    )

    # Convert optimizer config
    cfg.optim_wrapper = ConfigDict(
        type='OptimWrapper',
        optimizer=cfg.optimizer
    )

    # Convert training config
    cfg.train_cfg = ConfigDict(
        type='EpochBasedTrainLoop',
        max_epochs=cfg.runner.max_epochs,
        val_interval=1
    )

    # Set evaluation config
    cfg.val_cfg = ConfigDict(type='ValLoop')
    cfg.val_evaluator = ConfigDict(type='CocoMetric')

    # Set work directory
    cfg.work_dir = f'./{app_config.WORK_DIR}/indecs_retinanet'
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)

    # Set default runtime settings
    cfg.default_scope = 'mmdet'
    cfg.default_hooks = dict(
        timer=dict(type='IterTimerHook'),
        logger=dict(type='LoggerHook', interval=50),
        param_scheduler=dict(type='ParamSchedulerHook'),
        checkpoint=dict(type='CheckpointHook', interval=1),
        sampler_seed=dict(type='DistSamplerSeedHook'),
    )

    # Set launcher for training
    cfg.launcher = 'none'  # Set to 'pytorch' for distributed training

    # Create logger
    logger = MMLogger.get_instance('mmdet', log_level=cfg.log_level)

    # Create runner
    runner = Runner.from_cfg(cfg)

    # Start training
    runner.train()


if __name__ == '__main__':
    main()