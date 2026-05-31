# Model Routes

- craft_baseline.py: exists=True real_data=False jittor=True priority=low
- track1_dynamic_rec: exists=True real_data=True jittor=True priority=high
- train_enhanced_jittor.py: exists=True real_data=True jittor=True priority=high
- make_enhanced_jittor_result_zip.py: exists=True real_data=True jittor=True priority=high
- simple_heuristic.py: exists=True real_data=True jittor=False priority=high
- enhanced_heuristic.py: exists=True real_data=True jittor=False priority=high
- run_tune_sequential.sh: exists=True real_data=True jittor=False priority=high
- run_make_sequential_submit.sh: exists=True real_data=True jittor=True priority=high
- run_train_simple_jittor.sh: exists=True real_data=False jittor=True priority=low
- run_make_jittor_submit.sh: exists=True real_data=False jittor=True priority=low
- run_train_enhanced_jittor.sh: exists=True real_data=True jittor=True priority=high
- run_make_enhanced_jittor_submit.sh: exists=True real_data=True jittor=True priority=high
