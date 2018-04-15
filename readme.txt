1. We have implemented 2 CNN models - ConvNet and LeNet. The default one is LeNet. 

2. To change from LeNet to ConvNet, please modify model_fn.py to change "logits_train" & "logits_test" declarations to use "conv_net" instead of "le_net".

3. Most of the tuning and configurations parameters can be found in setting.py. 

4. The output of predict.py is saved as ./result.csv.

5. Checkpoints evaluation requires all checkpoints to be saved locally, otherwise the evaluation is skipped. To save all checkpoints, "max_ckpt" and "ckpt_steps" need to be properly tuned according to local space availability and training length. The results can be found at ./eval_ckpts.csv
e.g. num_steps = 50 & ckpt_steps = 10, to save all checkpoints --> max_ckpt >= 5 (checkpoints saved will be at step 11, 21, 31, 41 and 50)