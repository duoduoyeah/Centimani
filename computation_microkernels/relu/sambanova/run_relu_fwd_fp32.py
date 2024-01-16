import os

config = (
    #    w, h, c, n, k, filter_w(s), filter_h(r), pad_w, pad_h, wstride, hstride
#(128, 128, 128, 1024, 128, 5, 5, 0, 0, 1, 1), # resnet50  200Tflops
#(7, 7, 2048, 1024, 2048, 3, 3, 0, 0, 1, 1), # resnet50
#(54, 54, 1024, 256, 1024, 3, 3, 1, 1, 1, 1), # DeepSpeech
#(700, 161, 32, 512, 32, 20, 20, 0, 0, 2, 2), # Face Recognition
#(480, 48, 16, 1024, 16, 3, 3, 1, 1, 1, 1), # OCR
#(27, 27, 128, 8, 128, 3, 3, 1, 1, 1, 1),
#(60, 6, 64, 16, 128, 3, 3, 1, 1, 1, 1),
#(14,14,128,4,256,3,3,1,1,1,1),
(7, 7, 32, 131072, 32, 3, 3, 0, 0, 1, 1),
(14,14,128,8192,256,3,3,1,1,1,1),
(54, 54, 1024, 64, 1024, 3, 3, 1, 1, 1, 1), # DeepSpeech
(128, 128, 128, 64, 128, 5, 5, 0, 0, 1, 1), # resnet50  200Tflops
)

print("| w, h, c, n, k, filter_w(s), filter_h(r), pad_w, pad_h, wstride, hstride, Dtype, TFLOPS_FORW, TFLOPS_BACK|")
print("|---------------------------------------------------------------------------------------------------------|")

#for dataType in ("float32", "bfloat16"): #np.float32, np.float16,
for dataType in ("bfloat16",): #np.float32, np.float16,

    for w, h, c, n, k, s, r, pad_w, pad_h, wstride, hstride in config:

        str_compile = "time -p python relu_fwd.py compile -b " + str(n) + " -mb " + str(n) + " --dtype='" + dataType + "' --inference --w=" + str(w) + " --h=" + str(h) + " --c=" + str(c) + " --n=" + str(n) + " --output-folder='pef' --pef-name='relu_net'"
        print(str_compile)
        str_execute = "time -p python relu_fwd.py run -b " + str(n) + " -mb " + str(n) + " --dtype='" + dataType + "' --inference --w=" + str(w) + " --h=" + str(h) + " --c=" + str(c) + " --n=" + str(n) + " --pef='pef/relu_net/relu_net.pef'"
        print(str_execute)
        os.system(str_compile)
        os.system(str_execute)
