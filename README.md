# Planning
# ref
- https://github.com/karlkurzer/path_planner
- https://github.com/dawnjeanh/motionplanning




# Following algorithm

학습 시
1. `SAC_config.json` 에서 "inference_mode" : "False" 로 설정
2. `SAC_config.json` 에서 "load_model" : "False" 로 설정
3. `SAC_Trainer.py` 에서 `run` 함수안의 `env_config "followingMode": 0` 으로 설정

알고리즘 실행

```
python SAC_following.py
```

e
