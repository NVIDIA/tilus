## TASK

Currently, tilus support fp8 and fp16 tensor core, but not nvfp4 tensor core on B200.

You can check the example of fp16 tensor core gemm in examples/blackwell_gemm. And some usage in tests as well.

## Target

Support nvfp4 tensor core, and add a minimal example to illusrate the usage. Add a test for the nvfp4 tensor core. If's okay to add a new instruction like self.tcgen05.scaled_mma to show that we need scales for the mma operation (used by nvfp4 mma).

## Others

PTX documentaion can be accessed.

Feel free to read the codebase.

Try your best, I will help you after I come to work.

When you try to commit some changes, please disable gpg-sign since an interactive window will appear to ask for password, we should avoid this. Try to avoid operations that will require my
input. 

Generate a report if you tries all your best but still can not finish the job. 

If a kernel runs for a long time, it must be deadlock, you should kill it and think if there is any memory fence related issue.
