# Supplemental classification diagnostics

Class distributions describe the screened frozen subset and each model's declared admitted subset; they do not estimate class prevalence or performance on the full upstream dataset.

The frozen main accuracy ranking and its equal source/model weights remain unchanged. No combined F1 ranking is constructed.

W4A4: 10 complete matched models; 0 pending models; Native + 12 methods.

Confusion matrices use gold rows and predicted columns. Candidate-to-class mapping follows the frozen source and evaluator ordering contract (choice criteria insertion order; Boolean false/true). Prediction keys, source-derived gold indices and logit widths are checked; optional saved options are checked when present. Rows without saved option keys do not independently capture observed semantic ordering, so logits alone cannot establish it.

| Source | Semantic classes | Frozen requests / decisions | Gold counts | Majority-class accuracy % |
| --- | --- | ---: | --- | ---: |
| mnli | entailment, neutral, contradiction | 122 / 122 | entailment: 41, neutral: 45, contradiction: 36 | 36.885 |
| tweet-offensive | not_offensive, offensive | 128 / 128 | not_offensive: 97, offensive: 31 | 75.781 |
| wanli | supported, insufficient, contradicted | 256 / 256 | supported: 95, insufficient: 78, contradicted: 83 | 37.109 |
| wildjailbreak | harmful, benign | 128 / 128 | harmful: 119, benign: 9 | 92.969 |

| Model | Source | Method | Accuracy % | Balanced accuracy % | Macro-F1 % | Predicted counts | Per-class recall | Confusion matrix |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| kev-0.8b | mnli | native | 86.885 | 87.159 | 87.104 | entailment: 38, neutral: 46, contradiction: 38 | entailment: 0.8537, neutral: 0.8444, contradiction: 0.9167 | [[35, 6, 0], [2, 38, 5], [1, 2, 33]] |
| kev-0.8b | mnli | rtn | 29.508 | 29.715 | 29.596 | entailment: 39, neutral: 41, contradiction: 42 | entailment: 0.3415, neutral: 0.2444, contradiction: 0.3056 | [[14, 15, 12], [15, 11, 19], [10, 15, 11]] |
| kev-0.8b | mnli | s1q-local | 54.918 | 55.221 | 55.448 | entailment: 37, neutral: 50, contradiction: 35 | entailment: 0.5122, neutral: 0.5333, contradiction: 0.6111 | [[21, 16, 4], [12, 24, 9], [4, 10, 22]] |
| kev-0.8b | mnli | s1q2-beta05 | 48.361 | 47.927 | 48.241 | entailment: 43, neutral: 52, contradiction: 27 | entailment: 0.4878, neutral: 0.5333, contradiction: 0.4167 | [[20, 17, 4], [13, 24, 8], [10, 11, 15]] |
| kev-0.8b | mnli | smoothquant-adapted | 31.148 | 30.384 | 30.167 | entailment: 37, neutral: 57, contradiction: 28 | entailment: 0.3171, neutral: 0.4000, contradiction: 0.1944 | [[13, 21, 7], [13, 18, 14], [11, 18, 7]] |
| kev-0.8b | mnli | awq-adapted | 45.082 | 43.491 | 42.753 | entailment: 28, neutral: 77, contradiction: 17 | entailment: 0.3659, neutral: 0.6889, contradiction: 0.2500 | [[15, 25, 1], [7, 31, 7], [6, 21, 9]] |
| kev-0.8b | mnli | gptq-blockdiag-adapted | 32.787 | 33.338 | 32.567 | entailment: 42, neutral: 32, contradiction: 48 | entailment: 0.4390, neutral: 0.2000, contradiction: 0.3611 | [[18, 11, 12], [13, 9, 23], [11, 12, 13]] |
| kev-0.8b | mnli | spinquant-nohad-adapted | 32.787 | 31.938 | 31.705 | entailment: 41, neutral: 62, contradiction: 19 | entailment: 0.3415, neutral: 0.4222, contradiction: 0.1944 | [[14, 22, 5], [19, 19, 7], [8, 21, 7]] |
| kev-0.8b | mnli | spinquant-had-adapted | 49.180 | 49.142 | 49.049 | entailment: 48, neutral: 43, contradiction: 31 | entailment: 0.5854, neutral: 0.4444, contradiction: 0.4444 | [[24, 11, 6], [16, 20, 9], [8, 12, 16]] |
| kev-0.8b | mnli | s1q-joint | 49.180 | 49.151 | 49.488 | entailment: 38, neutral: 52, contradiction: 32 | entailment: 0.4634, neutral: 0.5111, contradiction: 0.5000 | [[19, 18, 4], [12, 23, 10], [7, 11, 18]] |
| kev-0.8b | mnli | s1q-ac | 54.918 | 54.512 | 54.673 | entailment: 41, neutral: 50, contradiction: 31 | entailment: 0.5854, neutral: 0.5778, contradiction: 0.4722 | [[24, 13, 4], [9, 26, 10], [8, 11, 17]] |
| kev-0.8b | mnli | s1q-margin | 59.016 | 58.401 | 58.978 | entailment: 37, neutral: 58, contradiction: 27 | entailment: 0.5854, neutral: 0.6667, contradiction: 0.5000 | [[24, 14, 3], [9, 30, 6], [4, 14, 18]] |
| kev-0.8b | mnli | s1q | 50.820 | 49.819 | 50.001 | entailment: 34, neutral: 65, contradiction: 23 | entailment: 0.4390, neutral: 0.6667, contradiction: 0.3889 | [[18, 19, 4], [10, 30, 5], [6, 16, 14]] |
| kev-0.8b | tweet-offensive | native | 73.438 | 70.402 | 67.686 | not_offensive: 85, offensive: 43 | not_offensive: 0.7629, offensive: 0.6452 | [[74, 23], [11, 20]] |
| kev-0.8b | tweet-offensive | rtn | 53.125 | 48.221 | 46.844 | not_offensive: 75, offensive: 53 | not_offensive: 0.5773, offensive: 0.3871 | [[56, 41], [19, 12]] |
| kev-0.8b | tweet-offensive | s1q-local | 72.656 | 54.523 | 54.393 | not_offensive: 112, offensive: 16 | not_offensive: 0.8969, offensive: 0.1935 | [[87, 10], [25, 6]] |
| kev-0.8b | tweet-offensive | s1q2-beta05 | 71.094 | 56.784 | 57.160 | not_offensive: 104, offensive: 24 | not_offensive: 0.8454, offensive: 0.2903 | [[82, 15], [22, 9]] |
| kev-0.8b | tweet-offensive | smoothquant-adapted | 51.562 | 51.580 | 47.885 | not_offensive: 65, offensive: 63 | not_offensive: 0.5155, offensive: 0.5161 | [[50, 47], [15, 16]] |
| kev-0.8b | tweet-offensive | awq-adapted | 75.000 | 62.654 | 63.429 | not_offensive: 103, offensive: 25 | not_offensive: 0.8660, offensive: 0.3871 | [[84, 13], [19, 12]] |
| kev-0.8b | tweet-offensive | gptq-blockdiag-adapted | 50.000 | 48.354 | 45.705 | not_offensive: 67, offensive: 61 | not_offensive: 0.5155, offensive: 0.4516 | [[50, 47], [17, 14]] |
| kev-0.8b | tweet-offensive | spinquant-nohad-adapted | 62.500 | 51.114 | 51.005 | not_offensive: 93, offensive: 35 | not_offensive: 0.7320, offensive: 0.2903 | [[71, 26], [22, 9]] |
| kev-0.8b | tweet-offensive | spinquant-had-adapted | 70.312 | 58.464 | 58.639 | not_offensive: 99, offensive: 29 | not_offensive: 0.8144, offensive: 0.3548 | [[79, 18], [20, 11]] |
| kev-0.8b | tweet-offensive | s1q-joint | 75.781 | 64.267 | 65.018 | not_offensive: 102, offensive: 26 | not_offensive: 0.8660, offensive: 0.4194 | [[84, 13], [18, 13]] |
| kev-0.8b | tweet-offensive | s1q-ac | 68.750 | 58.530 | 58.333 | not_offensive: 95, offensive: 33 | not_offensive: 0.7835, offensive: 0.3871 | [[76, 21], [19, 12]] |
| kev-0.8b | tweet-offensive | s1q-margin | 68.750 | 55.238 | 55.416 | not_offensive: 101, offensive: 27 | not_offensive: 0.8144, offensive: 0.2903 | [[79, 18], [22, 9]] |
| kev-0.8b | tweet-offensive | s1q | 72.656 | 60.010 | 60.504 | not_offensive: 102, offensive: 26 | not_offensive: 0.8454, offensive: 0.3548 | [[82, 15], [20, 11]] |
| kev-0.8b | wanli | native | 60.547 | 58.476 | 56.243 | supported: 150, insufficient: 32, contradicted: 74 | supported: 0.8737, insufficient: 0.2179, contradicted: 0.6627 | [[83, 7, 5], [47, 17, 14], [20, 8, 55]] |
| kev-0.8b | wanli | rtn | 35.547 | 35.951 | 35.371 | supported: 72, insufficient: 106, contradicted: 78 | supported: 0.3158, insufficient: 0.4615, contradicted: 0.3012 | [[30, 32, 33], [22, 36, 20], [20, 38, 25]] |
| kev-0.8b | wanli | s1q-local | 44.922 | 44.244 | 44.048 | supported: 119, insufficient: 89, contradicted: 48 | supported: 0.5789, insufficient: 0.4231, contradicted: 0.3253 | [[55, 32, 8], [32, 33, 13], [32, 24, 27]] |
| kev-0.8b | wanli | s1q2-beta05 | 45.312 | 44.517 | 44.905 | supported: 117, insufficient: 87, contradicted: 52 | supported: 0.5789, insufficient: 0.3590, contradicted: 0.3976 | [[55, 29, 11], [42, 28, 8], [20, 30, 33]] |
| kev-0.8b | wanli | smoothquant-adapted | 32.031 | 32.031 | 31.280 | supported: 89, insufficient: 111, contradicted: 56 | supported: 0.3579, insufficient: 0.4103, contradicted: 0.1928 | [[34, 39, 22], [28, 32, 18], [27, 40, 16]] |
| kev-0.8b | wanli | awq-adapted | 37.500 | 36.993 | 36.822 | supported: 114, insufficient: 108, contradicted: 34 | supported: 0.4842, insufficient: 0.3846, contradicted: 0.2410 | [[46, 42, 7], [41, 30, 7], [27, 36, 20]] |
| kev-0.8b | wanli | gptq-blockdiag-adapted | 37.891 | 38.284 | 37.920 | supported: 70, insufficient: 109, contradicted: 77 | supported: 0.3263, insufficient: 0.4487, contradicted: 0.3735 | [[31, 43, 21], [18, 35, 25], [21, 31, 31]] |
| kev-0.8b | wanli | spinquant-nohad-adapted | 36.719 | 35.882 | 35.784 | supported: 113, insufficient: 87, contradicted: 56 | supported: 0.5053, insufficient: 0.2821, contradicted: 0.2892 | [[48, 33, 14], [38, 22, 18], [27, 32, 24]] |
| kev-0.8b | wanli | spinquant-had-adapted | 34.766 | 33.925 | 33.430 | supported: 118, insufficient: 76, contradicted: 62 | supported: 0.4947, insufficient: 0.2821, contradicted: 0.2410 | [[47, 29, 19], [33, 22, 23], [38, 25, 20]] |
| kev-0.8b | wanli | s1q-joint | 44.531 | 43.713 | 44.000 | supported: 125, insufficient: 75, contradicted: 56 | supported: 0.5684, insufficient: 0.3333, contradicted: 0.4096 | [[54, 28, 13], [43, 26, 9], [28, 21, 34]] |
| kev-0.8b | wanli | s1q-ac | 45.703 | 44.639 | 44.467 | supported: 132, insufficient: 73, contradicted: 51 | supported: 0.6316, insufficient: 0.3462, contradicted: 0.3614 | [[60, 27, 8], [38, 27, 13], [34, 19, 30]] |
| kev-0.8b | wanli | s1q-margin | 41.797 | 40.188 | 39.198 | supported: 154, insufficient: 54, contradicted: 48 | supported: 0.6632, insufficient: 0.2051, contradicted: 0.3373 | [[63, 26, 6], [48, 16, 14], [43, 12, 28]] |
| kev-0.8b | wanli | s1q | 39.453 | 38.313 | 37.887 | supported: 138, insufficient: 69, contradicted: 49 | supported: 0.5789, insufficient: 0.2692, contradicted: 0.3012 | [[55, 28, 12], [45, 21, 12], [38, 20, 25]] |
| kev-0.8b | wildjailbreak | native | 28.906 | 61.765 | 27.304 | harmful: 28, benign: 100 | harmful: 0.2353, benign: 1.0000 | [[28, 91], [0, 9]] |
| kev-0.8b | wildjailbreak | rtn | 47.656 | 51.307 | 37.778 | harmful: 60, benign: 68 | harmful: 0.4706, benign: 0.5556 | [[56, 63], [4, 5]] |
| kev-0.8b | wildjailbreak | s1q-local | 32.812 | 53.595 | 29.436 | harmful: 37, benign: 91 | harmful: 0.2941, benign: 0.7778 | [[35, 84], [2, 7]] |
| kev-0.8b | wildjailbreak | s1q2-beta05 | 35.156 | 59.991 | 31.648 | harmful: 38, benign: 90 | harmful: 0.3109, benign: 0.8889 | [[37, 82], [1, 8]] |
| kev-0.8b | wildjailbreak | smoothquant-adapted | 19.531 | 46.452 | 18.933 | harmful: 20, benign: 108 | harmful: 0.1513, benign: 0.7778 | [[18, 101], [2, 7]] |
| kev-0.8b | wildjailbreak | awq-adapted | 35.156 | 54.855 | 31.116 | harmful: 40, benign: 88 | harmful: 0.3193, benign: 0.7778 | [[38, 81], [2, 7]] |
| kev-0.8b | wildjailbreak | gptq-blockdiag-adapted | 46.094 | 45.331 | 35.921 | harmful: 60, benign: 68 | harmful: 0.4622, benign: 0.4444 | [[55, 64], [5, 4]] |
| kev-0.8b | wildjailbreak | spinquant-nohad-adapted | 34.375 | 49.300 | 30.000 | harmful: 41, benign: 87 | harmful: 0.3193, benign: 0.6667 | [[38, 81], [3, 6]] |
| kev-0.8b | wildjailbreak | spinquant-had-adapted | 29.688 | 51.914 | 27.126 | harmful: 33, benign: 95 | harmful: 0.2605, benign: 0.7778 | [[31, 88], [2, 7]] |
| kev-0.8b | wildjailbreak | s1q-joint | 24.219 | 54.108 | 23.164 | harmful: 24, benign: 104 | harmful: 0.1933, benign: 0.8889 | [[23, 96], [1, 8]] |
| kev-0.8b | wildjailbreak | s1q-ac | 23.438 | 58.824 | 22.759 | harmful: 21, benign: 107 | harmful: 0.1765, benign: 1.0000 | [[21, 98], [0, 9]] |
| kev-0.8b | wildjailbreak | s1q-margin | 44.531 | 65.033 | 38.189 | harmful: 50, benign: 78 | harmful: 0.4118, benign: 0.8889 | [[49, 70], [1, 8]] |
| kev-0.8b | wildjailbreak | s1q | 53.125 | 69.655 | 43.860 | harmful: 61, benign: 67 | harmful: 0.5042, benign: 0.8889 | [[60, 59], [1, 8]] |
| kev-4b | mnli | native | 92.623 | 92.931 | 92.729 | entailment: 41, neutral: 43, contradiction: 38 | entailment: 0.9268, neutral: 0.8889, contradiction: 0.9722 | [[38, 3, 0], [2, 40, 3], [1, 0, 35]] |
| kev-4b | mnli | rtn | 37.705 | 37.647 | 37.537 | entailment: 40, neutral: 42, contradiction: 40 | entailment: 0.2683, neutral: 0.4444, contradiction: 0.4167 | [[11, 16, 14], [14, 20, 11], [15, 6, 15]] |
| kev-4b | mnli | s1q-local | 38.525 | 37.638 | 37.745 | entailment: 38, neutral: 65, contradiction: 19 | entailment: 0.3902, neutral: 0.4889, contradiction: 0.2500 | [[16, 22, 3], [16, 22, 7], [6, 21, 9]] |
| kev-4b | mnli | s1q2-beta05 | 45.082 | 44.313 | 44.145 | entailment: 22, neutral: 75, contradiction: 25 | entailment: 0.2683, neutral: 0.6444, contradiction: 0.4167 | [[11, 28, 2], [8, 29, 8], [3, 18, 15]] |
| kev-4b | mnli | smoothquant-adapted | 42.623 | 41.743 | 41.463 | entailment: 47, neutral: 50, contradiction: 25 | entailment: 0.4634, neutral: 0.5111, contradiction: 0.2778 | [[19, 16, 6], [13, 23, 9], [15, 11, 10]] |
| kev-4b | mnli | awq-adapted | 44.262 | 42.913 | 40.889 | entailment: 13, neutral: 88, contradiction: 21 | entailment: 0.1707, neutral: 0.7556, contradiction: 0.3611 | [[7, 33, 1], [4, 34, 7], [2, 21, 13]] |
| kev-4b | mnli | gptq-blockdiag-adapted | 32.787 | 33.049 | 32.811 | entailment: 41, neutral: 39, contradiction: 42 | entailment: 0.3415, neutral: 0.2889, contradiction: 0.3611 | [[14, 16, 11], [14, 13, 18], [13, 10, 13]] |
| kev-4b | mnli | spinquant-nohad-adapted | 27.869 | 27.832 | 27.849 | entailment: 34, neutral: 44, contradiction: 44 | entailment: 0.2683, neutral: 0.2889, contradiction: 0.2778 | [[11, 15, 15], [13, 13, 19], [10, 16, 10]] |
| kev-4b | mnli | spinquant-had-adapted | 40.984 | 40.086 | 39.884 | entailment: 40, neutral: 57, contradiction: 25 | entailment: 0.3415, neutral: 0.5556, contradiction: 0.3056 | [[14, 21, 6], [12, 25, 8], [14, 11, 11]] |
| kev-4b | mnli | s1q-joint | 69.672 | 69.255 | 70.105 | entailment: 28, neutral: 64, contradiction: 30 | entailment: 0.5610, neutral: 0.8222, contradiction: 0.6944 | [[23, 17, 1], [4, 37, 4], [1, 10, 25]] |
| kev-4b | mnli | s1q-ac | 64.754 | 63.627 | 64.396 | entailment: 28, neutral: 70, contradiction: 24 | entailment: 0.5366, neutral: 0.8444, contradiction: 0.5278 | [[22, 18, 1], [3, 38, 4], [3, 14, 19]] |
| kev-4b | mnli | s1q-margin | 68.852 | 68.803 | 69.467 | entailment: 33, neutral: 55, contradiction: 34 | entailment: 0.6585, neutral: 0.7111, contradiction: 0.6944 | [[27, 13, 1], [5, 32, 8], [1, 10, 25]] |
| kev-4b | mnli | s1q | 67.213 | 67.258 | 67.483 | entailment: 26, neutral: 60, contradiction: 36 | entailment: 0.5122, neutral: 0.7556, contradiction: 0.7500 | [[21, 18, 2], [4, 34, 7], [1, 8, 27]] |
| kev-4b | tweet-offensive | native | 78.125 | 64.716 | 66.214 | not_offensive: 107, offensive: 21 | not_offensive: 0.9072, offensive: 0.3871 | [[88, 9], [19, 12]] |
| kev-4b | tweet-offensive | rtn | 46.094 | 52.361 | 44.879 | not_offensive: 50, offensive: 78 | not_offensive: 0.4021, offensive: 0.6452 | [[39, 58], [11, 20]] |
| kev-4b | tweet-offensive | s1q-local | 69.531 | 49.169 | 47.425 | not_offensive: 114, offensive: 14 | not_offensive: 0.8866, offensive: 0.0968 | [[86, 11], [28, 3]] |
| kev-4b | tweet-offensive | s1q2-beta05 | 77.344 | 55.421 | 54.190 | not_offensive: 122, offensive: 6 | not_offensive: 0.9794, offensive: 0.1290 | [[95, 2], [27, 4]] |
| kev-4b | tweet-offensive | smoothquant-adapted | 42.969 | 42.617 | 39.883 | not_offensive: 60, offensive: 68 | not_offensive: 0.4330, offensive: 0.4194 | [[42, 55], [18, 13]] |
| kev-4b | tweet-offensive | awq-adapted | 73.438 | 49.551 | 45.051 | not_offensive: 123, offensive: 5 | not_offensive: 0.9588, offensive: 0.0323 | [[93, 4], [30, 1]] |
| kev-4b | tweet-offensive | gptq-blockdiag-adapted | 51.562 | 55.969 | 49.478 | not_offensive: 57, offensive: 71 | not_offensive: 0.4742, offensive: 0.6452 | [[46, 51], [11, 20]] |
| kev-4b | tweet-offensive | spinquant-nohad-adapted | 46.094 | 49.069 | 43.956 | not_offensive: 56, offensive: 72 | not_offensive: 0.4330, offensive: 0.5484 | [[42, 55], [14, 17]] |
| kev-4b | tweet-offensive | spinquant-had-adapted | 67.969 | 55.820 | 55.881 | not_offensive: 98, offensive: 30 | not_offensive: 0.7938, offensive: 0.3226 | [[77, 20], [21, 10]] |
| kev-4b | tweet-offensive | s1q-joint | 77.344 | 56.518 | 56.138 | not_offensive: 120, offensive: 8 | not_offensive: 0.9691, offensive: 0.1613 | [[94, 3], [26, 5]] |
| kev-4b | tweet-offensive | s1q-ac | 75.781 | 54.390 | 53.114 | not_offensive: 120, offensive: 8 | not_offensive: 0.9588, offensive: 0.1290 | [[93, 4], [27, 4]] |
| kev-4b | tweet-offensive | s1q-margin | 77.344 | 57.616 | 57.890 | not_offensive: 118, offensive: 10 | not_offensive: 0.9588, offensive: 0.1935 | [[93, 4], [25, 6]] |
| kev-4b | tweet-offensive | s1q | 75.781 | 56.585 | 56.676 | not_offensive: 116, offensive: 12 | not_offensive: 0.9381, offensive: 0.1935 | [[91, 6], [25, 6]] |
| kev-4b | wanli | native | 66.797 | 66.157 | 66.405 | supported: 99, insufficient: 100, contradicted: 57 | supported: 0.7895, insufficient: 0.6410, contradicted: 0.5542 | [[75, 19, 1], [18, 50, 10], [6, 31, 46]] |
| kev-4b | wanli | rtn | 32.031 | 31.723 | 31.723 | supported: 96, insufficient: 77, contradicted: 83 | supported: 0.3789, insufficient: 0.3077, contradicted: 0.2651 | [[36, 26, 33], [26, 24, 28], [34, 27, 22]] |
| kev-4b | wanli | s1q-local | 37.500 | 38.855 | 34.923 | supported: 55, insufficient: 171, contradicted: 30 | supported: 0.2421, insufficient: 0.7308, contradicted: 0.1928 | [[23, 63, 9], [16, 57, 5], [16, 51, 16]] |
| kev-4b | wanli | s1q2-beta05 | 32.812 | 34.697 | 27.734 | supported: 24, insufficient: 194, contradicted: 38 | supported: 0.1263, insufficient: 0.7821, contradicted: 0.1325 | [[12, 68, 15], [5, 61, 12], [7, 65, 11]] |
| kev-4b | wanli | smoothquant-adapted | 32.812 | 32.680 | 32.531 | supported: 88, insufficient: 104, contradicted: 64 | supported: 0.3684, insufficient: 0.3590, contradicted: 0.2530 | [[35, 39, 21], [28, 28, 22], [25, 37, 21]] |
| kev-4b | wanli | awq-adapted | 37.109 | 38.862 | 32.492 | supported: 28, insufficient: 200, contradicted: 28 | supported: 0.2000, insufficient: 0.8333, contradicted: 0.1325 | [[19, 67, 9], [5, 65, 8], [4, 68, 11]] |
| kev-4b | wanli | gptq-blockdiag-adapted | 31.641 | 31.651 | 31.661 | supported: 74, insufficient: 94, contradicted: 88 | supported: 0.3053, insufficient: 0.2949, contradicted: 0.3494 | [[29, 41, 25], [21, 23, 34], [24, 30, 29]] |
| kev-4b | wanli | spinquant-nohad-adapted | 31.641 | 31.473 | 31.476 | supported: 86, insufficient: 77, contradicted: 93 | supported: 0.3368, insufficient: 0.2821, contradicted: 0.3253 | [[32, 25, 38], [28, 22, 28], [26, 30, 27]] |
| kev-4b | wanli | spinquant-had-adapted | 37.109 | 37.712 | 36.557 | supported: 76, insufficient: 117, contradicted: 63 | supported: 0.3158, insufficient: 0.5385, contradicted: 0.2771 | [[30, 40, 25], [21, 42, 15], [25, 35, 23]] |
| kev-4b | wanli | s1q-joint | 44.922 | 46.488 | 43.229 | supported: 38, insufficient: 185, contradicted: 33 | supported: 0.2842, insufficient: 0.8333, contradicted: 0.2771 | [[27, 63, 5], [8, 65, 5], [3, 57, 23]] |
| kev-4b | wanli | s1q-ac | 40.625 | 42.070 | 37.650 | supported: 41, insufficient: 189, contradicted: 26 | supported: 0.2737, insufficient: 0.8077, contradicted: 0.1807 | [[26, 63, 6], [10, 63, 5], [5, 63, 15]] |
| kev-4b | wanli | s1q-margin | 46.094 | 47.287 | 44.270 | supported: 43, insufficient: 187, contradicted: 26 | supported: 0.3684, insufficient: 0.8333, contradicted: 0.2169 | [[35, 58, 2], [7, 65, 6], [1, 64, 18]] |
| kev-4b | wanli | s1q | 47.656 | 49.225 | 45.859 | supported: 46, insufficient: 179, contradicted: 31 | supported: 0.3158, insufficient: 0.8718, contradicted: 0.2892 | [[30, 61, 4], [7, 68, 3], [9, 50, 24]] |
| kev-4b | wildjailbreak | native | 92.188 | 90.663 | 78.595 | harmful: 111, benign: 17 | harmful: 0.9244, benign: 0.8889 | [[110, 9], [1, 8]] |
| kev-4b | wildjailbreak | rtn | 46.875 | 45.752 | 36.374 | harmful: 61, benign: 67 | harmful: 0.4706, benign: 0.4444 | [[56, 63], [5, 4]] |
| kev-4b | wildjailbreak | s1q-local | 41.406 | 63.352 | 36.064 | harmful: 46, benign: 82 | harmful: 0.3782, benign: 0.8889 | [[45, 74], [1, 8]] |
| kev-4b | wildjailbreak | s1q2-beta05 | 35.156 | 65.126 | 32.137 | harmful: 36, benign: 92 | harmful: 0.3025, benign: 1.0000 | [[36, 83], [0, 9]] |
| kev-4b | wildjailbreak | smoothquant-adapted | 51.562 | 38.002 | 36.715 | harmful: 71, benign: 57 | harmful: 0.5378, benign: 0.2222 | [[64, 55], [7, 2]] |
| kev-4b | wildjailbreak | awq-adapted | 35.156 | 59.991 | 31.648 | harmful: 38, benign: 90 | harmful: 0.3109, benign: 0.8889 | [[37, 82], [1, 8]] |
| kev-4b | wildjailbreak | gptq-blockdiag-adapted | 47.656 | 35.901 | 34.709 | harmful: 66, benign: 62 | harmful: 0.4958, benign: 0.2222 | [[59, 60], [7, 2]] |
| kev-4b | wildjailbreak | spinquant-nohad-adapted | 39.062 | 36.415 | 30.897 | harmful: 53, benign: 75 | harmful: 0.3950, benign: 0.3333 | [[47, 72], [6, 3]] |
| kev-4b | wildjailbreak | spinquant-had-adapted | 42.969 | 48.786 | 34.926 | harmful: 54, benign: 74 | harmful: 0.4202, benign: 0.5556 | [[50, 69], [4, 5]] |
| kev-4b | wildjailbreak | s1q-joint | 59.375 | 73.016 | 47.935 | harmful: 69, benign: 59 | harmful: 0.5714, benign: 0.8889 | [[68, 51], [1, 8]] |
| kev-4b | wildjailbreak | s1q-ac | 61.719 | 74.276 | 49.480 | harmful: 72, benign: 56 | harmful: 0.5966, benign: 0.8889 | [[71, 48], [1, 8]] |
| kev-4b | wildjailbreak | s1q-margin | 59.375 | 67.880 | 46.922 | harmful: 71, benign: 57 | harmful: 0.5798, benign: 0.7778 | [[69, 50], [2, 7]] |
| kev-4b | wildjailbreak | s1q | 56.250 | 71.335 | 45.894 | harmful: 65, benign: 63 | harmful: 0.5378, benign: 0.8889 | [[64, 55], [1, 8]] |
| kev-9b | mnli | native | 90.164 | 90.709 | 90.243 | entailment: 42, neutral: 40, contradiction: 40 | entailment: 0.9268, neutral: 0.8222, contradiction: 0.9722 | [[38, 3, 0], [3, 37, 5], [1, 0, 35]] |
| kev-9b | mnli | rtn | 29.508 | 29.458 | 29.435 | entailment: 40, neutral: 41, contradiction: 41 | entailment: 0.3171, neutral: 0.2889, contradiction: 0.2778 | [[13, 13, 15], [16, 13, 16], [11, 15, 10]] |
| kev-9b | mnli | s1q-local | 32.787 | 32.132 | 31.826 | entailment: 34, neutral: 48, contradiction: 40 | entailment: 0.2195, neutral: 0.4667, contradiction: 0.2778 | [[9, 15, 17], [11, 21, 13], [14, 12, 10]] |
| kev-9b | mnli | s1q2-beta05 | 27.049 | 26.978 | 27.155 | entailment: 50, neutral: 44, contradiction: 28 | entailment: 0.2927, neutral: 0.2667, contradiction: 0.2500 | [[12, 24, 5], [19, 12, 14], [19, 8, 9]] |
| kev-9b | mnli | smoothquant-adapted | 31.967 | 31.680 | 31.732 | entailment: 44, neutral: 41, contradiction: 37 | entailment: 0.3171, neutral: 0.3556, contradiction: 0.2778 | [[13, 13, 15], [17, 16, 12], [14, 12, 10]] |
| kev-9b | mnli | awq-adapted | 39.344 | 38.482 | 37.938 | entailment: 47, neutral: 49, contradiction: 26 | entailment: 0.4878, neutral: 0.4444, contradiction: 0.2222 | [[20, 15, 6], [13, 20, 12], [14, 14, 8]] |
| kev-9b | mnli | gptq-blockdiag-adapted | 36.885 | 36.752 | 36.790 | entailment: 36, neutral: 50, contradiction: 36 | entailment: 0.3415, neutral: 0.4000, contradiction: 0.3611 | [[14, 17, 10], [14, 18, 13], [8, 15, 13]] |
| kev-9b | mnli | spinquant-nohad-adapted | 34.426 | 34.715 | 34.516 | entailment: 35, neutral: 37, contradiction: 50 | entailment: 0.3415, neutral: 0.3111, contradiction: 0.3889 | [[14, 13, 14], [9, 14, 22], [12, 10, 14]] |
| kev-9b | mnli | spinquant-had-adapted | 40.984 | 39.995 | 39.105 | entailment: 52, neutral: 42, contradiction: 28 | entailment: 0.5610, neutral: 0.4444, contradiction: 0.1944 | [[23, 9, 9], [13, 20, 12], [16, 13, 7]] |
| kev-9b | mnli | s1q-joint | 67.213 | 67.642 | 67.179 | entailment: 52, neutral: 38, contradiction: 32 | entailment: 0.8293, neutral: 0.5333, contradiction: 0.6667 | [[34, 7, 0], [13, 24, 8], [5, 7, 24]] |
| kev-9b | mnli | s1q-ac | 75.410 | 75.790 | 75.636 | entailment: 45, neutral: 42, contradiction: 35 | entailment: 0.8293, neutral: 0.6667, contradiction: 0.7778 | [[34, 6, 1], [9, 30, 6], [2, 6, 28]] |
| kev-9b | mnli | s1q-margin | 55.738 | 55.253 | 55.979 | entailment: 40, neutral: 57, contradiction: 25 | entailment: 0.5854, neutral: 0.6000, contradiction: 0.4722 | [[24, 16, 1], [11, 27, 7], [5, 14, 17]] |
| kev-9b | mnli | s1q | 74.590 | 74.020 | 74.896 | entailment: 38, neutral: 57, contradiction: 27 | entailment: 0.7317, neutral: 0.8222, contradiction: 0.6667 | [[30, 10, 1], [6, 37, 2], [2, 10, 24]] |
| kev-9b | tweet-offensive | native | 82.031 | 68.390 | 70.975 | not_offensive: 110, offensive: 18 | not_offensive: 0.9485, offensive: 0.4194 | [[92, 5], [18, 13]] |
| kev-9b | tweet-offensive | rtn | 42.188 | 46.492 | 40.741 | not_offensive: 51, offensive: 77 | not_offensive: 0.3814, offensive: 0.5484 | [[37, 60], [14, 17]] |
| kev-9b | tweet-offensive | s1q-local | 56.250 | 45.893 | 45.894 | not_offensive: 87, offensive: 41 | not_offensive: 0.6598, offensive: 0.2581 | [[64, 33], [23, 8]] |
| kev-9b | tweet-offensive | s1q2-beta05 | 62.500 | 55.504 | 54.381 | not_offensive: 85, offensive: 43 | not_offensive: 0.6907, offensive: 0.4194 | [[67, 30], [18, 13]] |
| kev-9b | tweet-offensive | smoothquant-adapted | 53.906 | 49.834 | 48.043 | not_offensive: 74, offensive: 54 | not_offensive: 0.5773, offensive: 0.4194 | [[56, 41], [18, 13]] |
| kev-9b | tweet-offensive | awq-adapted | 67.969 | 52.527 | 52.528 | not_offensive: 104, offensive: 24 | not_offensive: 0.8247, offensive: 0.2258 | [[80, 17], [24, 7]] |
| kev-9b | tweet-offensive | gptq-blockdiag-adapted | 46.875 | 45.195 | 42.842 | not_offensive: 65, offensive: 63 | not_offensive: 0.4845, offensive: 0.4194 | [[47, 50], [18, 13]] |
| kev-9b | tweet-offensive | spinquant-nohad-adapted | 51.562 | 49.385 | 46.881 | not_offensive: 69, offensive: 59 | not_offensive: 0.5361, offensive: 0.4516 | [[52, 45], [17, 14]] |
| kev-9b | tweet-offensive | spinquant-had-adapted | 52.344 | 45.510 | 44.917 | not_offensive: 78, offensive: 50 | not_offensive: 0.5876, offensive: 0.3226 | [[57, 40], [21, 10]] |
| kev-9b | tweet-offensive | s1q-joint | 78.125 | 59.228 | 60.125 | not_offensive: 117, offensive: 11 | not_offensive: 0.9588, offensive: 0.2258 | [[93, 4], [24, 7]] |
| kev-9b | tweet-offensive | s1q-ac | 75.781 | 53.292 | 51.030 | not_offensive: 122, offensive: 6 | not_offensive: 0.9691, offensive: 0.0968 | [[94, 3], [28, 3]] |
| kev-9b | tweet-offensive | s1q-margin | 76.562 | 51.613 | 46.429 | not_offensive: 127, offensive: 1 | not_offensive: 1.0000, offensive: 0.0323 | [[97, 0], [30, 1]] |
| kev-9b | tweet-offensive | s1q | 78.125 | 57.034 | 56.736 | not_offensive: 121, offensive: 7 | not_offensive: 0.9794, offensive: 0.1613 | [[95, 2], [26, 5]] |
| kev-9b | wanli | native | 76.562 | 75.511 | 75.757 | supported: 119, insufficient: 68, contradicted: 69 | supported: 0.9158, insufficient: 0.6026, contradicted: 0.7470 | [[87, 8, 0], [24, 47, 7], [8, 13, 62]] |
| kev-9b | wanli | rtn | 35.938 | 35.791 | 35.779 | supported: 89, insufficient: 81, contradicted: 86 | supported: 0.3789, insufficient: 0.3333, contradicted: 0.3614 | [[36, 24, 35], [31, 26, 21], [22, 31, 30]] |
| kev-9b | wanli | s1q-local | 33.984 | 34.546 | 33.913 | supported: 68, insufficient: 104, contradicted: 84 | supported: 0.2526, insufficient: 0.4103, contradicted: 0.3735 | [[24, 41, 30], [23, 32, 23], [21, 31, 31]] |
| kev-9b | wanli | s1q2-beta05 | 35.156 | 35.344 | 35.164 | supported: 82, insufficient: 88, contradicted: 86 | supported: 0.3263, insufficient: 0.3846, contradicted: 0.3494 | [[31, 32, 32], [23, 30, 25], [28, 26, 29]] |
| kev-9b | wanli | smoothquant-adapted | 35.938 | 35.714 | 35.669 | supported: 82, insufficient: 75, contradicted: 99 | supported: 0.3789, insufficient: 0.2949, contradicted: 0.3976 | [[36, 24, 35], [24, 23, 31], [22, 28, 33]] |
| kev-9b | wanli | awq-adapted | 36.328 | 36.627 | 36.222 | supported: 85, insufficient: 103, contradicted: 68 | supported: 0.3368, insufficient: 0.4487, contradicted: 0.3133 | [[32, 37, 26], [27, 35, 16], [26, 31, 26]] |
| kev-9b | wanli | gptq-blockdiag-adapted | 29.688 | 29.336 | 29.092 | supported: 99, insufficient: 66, contradicted: 91 | supported: 0.3263, insufficient: 0.1923, contradicted: 0.3614 | [[31, 28, 36], [38, 15, 25], [30, 23, 30]] |
| kev-9b | wanli | spinquant-nohad-adapted | 34.375 | 34.873 | 34.371 | supported: 71, insufficient: 94, contradicted: 91 | supported: 0.2842, insufficient: 0.4487, contradicted: 0.3133 | [[27, 30, 38], [16, 35, 27], [28, 29, 26]] |
| kev-9b | wanli | spinquant-had-adapted | 26.953 | 27.189 | 26.661 | supported: 77, insufficient: 103, contradicted: 76 | supported: 0.2632, insufficient: 0.3718, contradicted: 0.1807 | [[25, 31, 39], [27, 29, 22], [25, 43, 15]] |
| kev-9b | wanli | s1q-joint | 52.344 | 51.420 | 50.896 | supported: 126, insufficient: 91, contradicted: 39 | supported: 0.7053, insufficient: 0.5000, contradicted: 0.3373 | [[67, 25, 3], [31, 39, 8], [28, 27, 28]] |
| kev-9b | wanli | s1q-ac | 54.297 | 53.099 | 51.916 | supported: 131, insufficient: 93, contradicted: 32 | supported: 0.7789, insufficient: 0.5128, contradicted: 0.3012 | [[74, 19, 2], [33, 40, 5], [24, 34, 25]] |
| kev-9b | wanli | s1q-margin | 52.734 | 51.594 | 50.146 | supported: 130, insufficient: 93, contradicted: 33 | supported: 0.7579, insufficient: 0.5128, contradicted: 0.2771 | [[72, 19, 4], [32, 40, 6], [26, 34, 23]] |
| kev-9b | wanli | s1q | 55.859 | 54.757 | 53.785 | supported: 121, insufficient: 99, contradicted: 36 | supported: 0.7789, insufficient: 0.5385, contradicted: 0.3253 | [[74, 18, 3], [30, 42, 6], [17, 39, 27]] |
| kev-9b | wildjailbreak | native | 53.906 | 70.075 | 44.368 | harmful: 62, benign: 66 | harmful: 0.5126, benign: 0.8889 | [[61, 58], [1, 8]] |
| kev-9b | wildjailbreak | rtn | 44.531 | 49.627 | 35.887 | harmful: 56, benign: 72 | harmful: 0.4370, benign: 0.5556 | [[52, 67], [4, 5]] |
| kev-9b | wildjailbreak | s1q-local | 57.812 | 56.769 | 43.750 | harmful: 73, benign: 55 | harmful: 0.5798, benign: 0.5556 | [[69, 50], [4, 5]] |
| kev-9b | wildjailbreak | s1q2-beta05 | 42.969 | 38.515 | 33.176 | harmful: 58, benign: 70 | harmful: 0.4370, benign: 0.3333 | [[52, 67], [6, 3]] |
| kev-9b | wildjailbreak | smoothquant-adapted | 53.125 | 49.113 | 39.925 | harmful: 69, benign: 59 | harmful: 0.5378, benign: 0.4444 | [[64, 55], [5, 4]] |
| kev-9b | wildjailbreak | awq-adapted | 54.688 | 60.224 | 42.980 | harmful: 67, benign: 61 | harmful: 0.5378, benign: 0.6667 | [[64, 55], [3, 6]] |
| kev-9b | wildjailbreak | gptq-blockdiag-adapted | 48.438 | 46.592 | 37.274 | harmful: 63, benign: 65 | harmful: 0.4874, benign: 0.4444 | [[58, 61], [5, 4]] |
| kev-9b | wildjailbreak | spinquant-nohad-adapted | 49.219 | 41.877 | 36.658 | harmful: 66, benign: 62 | harmful: 0.5042, benign: 0.3333 | [[60, 59], [6, 3]] |
| kev-9b | wildjailbreak | spinquant-had-adapted | 49.219 | 47.012 | 37.720 | harmful: 64, benign: 64 | harmful: 0.4958, benign: 0.4444 | [[59, 60], [5, 4]] |
| kev-9b | wildjailbreak | s1q-joint | 21.875 | 57.983 | 21.395 | harmful: 19, benign: 109 | harmful: 0.1597, benign: 1.0000 | [[19, 100], [0, 9]] |
| kev-9b | wildjailbreak | s1q-ac | 22.656 | 58.403 | 22.081 | harmful: 20, benign: 108 | harmful: 0.1681, benign: 1.0000 | [[20, 99], [0, 9]] |
| kev-9b | wildjailbreak | s1q-margin | 14.844 | 54.202 | 14.839 | harmful: 10, benign: 118 | harmful: 0.0840, benign: 1.0000 | [[10, 109], [0, 9]] |
| kev-9b | wildjailbreak | s1q | 20.312 | 57.143 | 20.000 | harmful: 17, benign: 111 | harmful: 0.1429, benign: 1.0000 | [[17, 102], [0, 9]] |
| laya | mnli | native | 63.934 | 64.119 | 63.872 | entailment: 24, neutral: 64, contradiction: 34 | entailment: 0.3902, neutral: 0.7556, contradiction: 0.7778 | [[16, 23, 2], [7, 34, 4], [1, 7, 28]] |
| laya | mnli | rtn | 44.262 | 44.291 | 38.442 | entailment: 3, neutral: 66, contradiction: 53 | entailment: 0.0732, neutral: 0.6444, contradiction: 0.6111 | [[3, 23, 15], [0, 29, 16], [0, 14, 22]] |
| laya | mnli | s1q-local | 57.377 | 57.389 | 54.758 | entailment: 12, neutral: 67, contradiction: 43 | entailment: 0.2439, neutral: 0.7556, contradiction: 0.7222 | [[10, 24, 7], [1, 34, 10], [1, 9, 26]] |
| laya | mnli | s1q2-beta05 | 53.279 | 52.913 | 50.324 | entailment: 12, neutral: 78, contradiction: 32 | entailment: 0.1707, neutral: 0.7778, contradiction: 0.6389 | [[7, 31, 3], [4, 35, 6], [1, 12, 23]] |
| laya | mnli | smoothquant-adapted | 52.459 | 52.398 | 47.657 | entailment: 8, neutral: 73, contradiction: 41 | entailment: 0.1220, neutral: 0.7556, contradiction: 0.6944 | [[5, 28, 8], [3, 34, 8], [0, 11, 25]] |
| laya | mnli | awq-adapted | 51.639 | 51.802 | 48.032 | entailment: 9, neutral: 66, contradiction: 47 | entailment: 0.1707, neutral: 0.6889, contradiction: 0.6944 | [[7, 24, 10], [2, 31, 12], [0, 11, 25]] |
| laya | mnli | gptq-blockdiag-adapted | 33.607 | 33.735 | 29.836 | entailment: 4, neutral: 59, contradiction: 59 | entailment: 0.0732, neutral: 0.4667, contradiction: 0.4722 | [[3, 19, 19], [1, 21, 23], [0, 19, 17]] |
| laya | mnli | spinquant-nohad-adapted | 59.016 | 59.160 | 59.121 | entailment: 23, neutral: 67, contradiction: 32 | entailment: 0.3415, neutral: 0.7111, contradiction: 0.7222 | [[14, 26, 1], [8, 32, 5], [1, 9, 26]] |
| laya | mnli | spinquant-had-adapted | 62.295 | 62.091 | 61.075 | entailment: 16, neutral: 76, contradiction: 30 | entailment: 0.2683, neutral: 0.8444, contradiction: 0.7500 | [[11, 29, 1], [5, 38, 2], [0, 9, 27]] |
| laya | mnli | s1q-joint | 56.557 | 56.238 | 55.322 | entailment: 18, neutral: 71, contradiction: 33 | entailment: 0.2927, neutral: 0.7556, contradiction: 0.6389 | [[12, 25, 4], [5, 34, 6], [1, 12, 23]] |
| laya | mnli | s1q-ac | 54.098 | 53.500 | 52.687 | entailment: 17, neutral: 78, contradiction: 27 | entailment: 0.2439, neutral: 0.7778, contradiction: 0.5833 | [[10, 28, 3], [7, 35, 3], [0, 15, 21]] |
| laya | mnli | s1q-margin | 53.279 | 52.760 | 51.513 | entailment: 13, neutral: 77, contradiction: 32 | entailment: 0.2439, neutral: 0.7556, contradiction: 0.5833 | [[10, 28, 3], [3, 34, 8], [0, 15, 21]] |
| laya | mnli | s1q | 56.557 | 55.980 | 54.983 | entailment: 13, neutral: 78, contradiction: 31 | entailment: 0.2683, neutral: 0.8000, contradiction: 0.6111 | [[11, 28, 2], [2, 36, 7], [0, 14, 22]] |
| laya | tweet-offensive | native | 77.344 | 55.421 | 54.190 | not_offensive: 122, offensive: 6 | not_offensive: 0.9794, offensive: 0.1290 | [[95, 2], [27, 4]] |
| laya | tweet-offensive | rtn | 76.562 | 51.613 | 46.429 | not_offensive: 127, offensive: 1 | not_offensive: 1.0000, offensive: 0.0323 | [[97, 0], [30, 1]] |
| laya | tweet-offensive | s1q-local | 77.344 | 53.226 | 49.558 | not_offensive: 126, offensive: 2 | not_offensive: 1.0000, offensive: 0.0645 | [[97, 0], [29, 2]] |
| laya | tweet-offensive | s1q2-beta05 | 77.344 | 54.323 | 52.010 | not_offensive: 124, offensive: 4 | not_offensive: 0.9897, offensive: 0.0968 | [[96, 1], [28, 3]] |
| laya | tweet-offensive | smoothquant-adapted | 75.781 | 50.000 | 43.111 | not_offensive: 128, offensive: 0 | not_offensive: 1.0000, offensive: 0.0000 | [[97, 0], [31, 0]] |
| laya | tweet-offensive | awq-adapted | 75.781 | 50.000 | 43.111 | not_offensive: 128, offensive: 0 | not_offensive: 1.0000, offensive: 0.0000 | [[97, 0], [31, 0]] |
| laya | tweet-offensive | gptq-blockdiag-adapted | 77.344 | 53.226 | 49.558 | not_offensive: 126, offensive: 2 | not_offensive: 1.0000, offensive: 0.0645 | [[97, 0], [29, 2]] |
| laya | tweet-offensive | spinquant-nohad-adapted | 77.344 | 54.323 | 52.010 | not_offensive: 124, offensive: 4 | not_offensive: 0.9897, offensive: 0.0968 | [[96, 1], [28, 3]] |
| laya | tweet-offensive | spinquant-had-adapted | 77.344 | 53.226 | 49.558 | not_offensive: 126, offensive: 2 | not_offensive: 1.0000, offensive: 0.0645 | [[97, 0], [29, 2]] |
| laya | tweet-offensive | s1q-joint | 75.781 | 50.000 | 43.111 | not_offensive: 128, offensive: 0 | not_offensive: 1.0000, offensive: 0.0000 | [[97, 0], [31, 0]] |
| laya | tweet-offensive | s1q-ac | 75.781 | 50.000 | 43.111 | not_offensive: 128, offensive: 0 | not_offensive: 1.0000, offensive: 0.0000 | [[97, 0], [31, 0]] |
| laya | tweet-offensive | s1q-margin | 75.000 | 49.485 | 42.857 | not_offensive: 127, offensive: 1 | not_offensive: 0.9897, offensive: 0.0000 | [[96, 1], [31, 0]] |
| laya | tweet-offensive | s1q | 75.781 | 50.000 | 43.111 | not_offensive: 128, offensive: 0 | not_offensive: 1.0000, offensive: 0.0000 | [[97, 0], [31, 0]] |
| laya | wanli | native | 57.031 | 55.907 | 56.200 | supported: 125, insufficient: 74, contradicted: 57 | supported: 0.7368, insufficient: 0.4103, contradicted: 0.5301 | [[70, 22, 3], [36, 32, 10], [19, 20, 44]] |
| laya | wanli | rtn | 37.109 | 38.471 | 35.158 | supported: 29, insufficient: 115, contradicted: 112 | supported: 0.1368, insufficient: 0.4872, contradicted: 0.5301 | [[13, 48, 34], [6, 38, 34], [10, 29, 44]] |
| laya | wanli | s1q-local | 42.188 | 42.320 | 42.302 | supported: 81, insufficient: 93, contradicted: 82 | supported: 0.3789, insufficient: 0.3846, contradicted: 0.5060 | [[36, 42, 17], [25, 30, 23], [20, 21, 42]] |
| laya | wanli | s1q2-beta05 | 46.094 | 46.341 | 46.060 | supported: 88, insufficient: 115, contradicted: 53 | supported: 0.4526, insufficient: 0.5641, contradicted: 0.3735 | [[43, 44, 8], [20, 44, 14], [25, 27, 31]] |
| laya | wanli | smoothquant-adapted | 40.234 | 40.539 | 40.049 | supported: 61, insufficient: 93, contradicted: 102 | supported: 0.3263, insufficient: 0.3718, contradicted: 0.5181 | [[31, 38, 26], [16, 29, 33], [14, 26, 43]] |
| laya | wanli | awq-adapted | 48.828 | 49.582 | 48.578 | supported: 62, insufficient: 100, contradicted: 94 | supported: 0.3474, insufficient: 0.5256, contradicted: 0.6145 | [[33, 42, 20], [14, 41, 23], [15, 17, 51]] |
| laya | wanli | gptq-blockdiag-adapted | 34.375 | 34.968 | 32.111 | supported: 40, insufficient: 72, contradicted: 144 | supported: 0.1789, insufficient: 0.2436, contradicted: 0.6265 | [[17, 32, 46], [13, 19, 46], [10, 21, 52]] |
| laya | wanli | spinquant-nohad-adapted | 51.172 | 50.188 | 50.226 | supported: 125, insufficient: 78, contradicted: 53 | supported: 0.6737, insufficient: 0.4103, contradicted: 0.4217 | [[64, 25, 6], [34, 32, 12], [27, 21, 35]] |
| laya | wanli | spinquant-had-adapted | 54.688 | 53.727 | 53.930 | supported: 125, insufficient: 79, contradicted: 52 | supported: 0.7053, insufficient: 0.4487, contradicted: 0.4578 | [[67, 25, 3], [32, 35, 11], [26, 19, 38]] |
| laya | wanli | s1q-joint | 48.438 | 47.935 | 48.102 | supported: 104, insufficient: 89, contradicted: 63 | supported: 0.5684, insufficient: 0.4359, contradicted: 0.4337 | [[54, 28, 13], [30, 34, 14], [20, 27, 36]] |
| laya | wanli | s1q-ac | 46.094 | 45.474 | 45.711 | supported: 111, insufficient: 92, contradicted: 53 | supported: 0.5684, insufficient: 0.4103, contradicted: 0.3855 | [[54, 34, 7], [32, 32, 14], [25, 26, 32]] |
| laya | wanli | s1q-margin | 50.000 | 48.854 | 48.918 | supported: 130, insufficient: 71, contradicted: 55 | supported: 0.6737, insufficient: 0.3462, contradicted: 0.4458 | [[64, 27, 4], [37, 27, 14], [29, 17, 37]] |
| laya | wanli | s1q | 49.609 | 48.964 | 48.835 | supported: 116, insufficient: 89, contradicted: 51 | supported: 0.6211, insufficient: 0.4744, contradicted: 0.3735 | [[59, 28, 8], [29, 37, 12], [28, 24, 31]] |
| laya | wildjailbreak | native | 10.938 | 52.101 | 10.850 | harmful: 5, benign: 123 | harmful: 0.0420, benign: 1.0000 | [[5, 114], [0, 9]] |
| laya | wildjailbreak | rtn | 31.250 | 52.754 | 28.291 | harmful: 35, benign: 93 | harmful: 0.2773, benign: 0.7778 | [[33, 86], [2, 7]] |
| laya | wildjailbreak | s1q-local | 17.969 | 50.747 | 17.723 | harmful: 16, benign: 112 | harmful: 0.1261, benign: 0.8889 | [[15, 104], [1, 8]] |
| laya | wildjailbreak | s1q2-beta05 | 14.844 | 49.066 | 14.797 | harmful: 12, benign: 116 | harmful: 0.0924, benign: 0.8889 | [[11, 108], [1, 8]] |
| laya | wildjailbreak | smoothquant-adapted | 20.312 | 57.143 | 20.000 | harmful: 17, benign: 111 | harmful: 0.1429, benign: 1.0000 | [[17, 102], [0, 9]] |
| laya | wildjailbreak | awq-adapted | 9.375 | 40.990 | 9.353 | harmful: 7, benign: 121 | harmful: 0.0420, benign: 0.7778 | [[5, 114], [2, 7]] |
| laya | wildjailbreak | gptq-blockdiag-adapted | 26.562 | 55.369 | 25.081 | harmful: 27, benign: 101 | harmful: 0.2185, benign: 0.8889 | [[26, 93], [1, 8]] |
| laya | wildjailbreak | spinquant-nohad-adapted | 32.812 | 58.730 | 29.921 | harmful: 35, benign: 93 | harmful: 0.2857, benign: 0.8889 | [[34, 85], [1, 8]] |
| laya | wildjailbreak | spinquant-had-adapted | 10.938 | 52.101 | 10.850 | harmful: 5, benign: 123 | harmful: 0.0420, benign: 1.0000 | [[5, 114], [0, 9]] |
| laya | wildjailbreak | s1q-joint | 14.062 | 48.646 | 14.042 | harmful: 11, benign: 117 | harmful: 0.0840, benign: 0.8889 | [[10, 109], [1, 8]] |
| laya | wildjailbreak | s1q-ac | 10.938 | 46.965 | 10.916 | harmful: 7, benign: 121 | harmful: 0.0504, benign: 0.8889 | [[6, 113], [1, 8]] |
| laya | wildjailbreak | s1q-margin | 17.188 | 55.462 | 17.107 | harmful: 13, benign: 115 | harmful: 0.1092, benign: 1.0000 | [[13, 106], [0, 9]] |
| laya | wildjailbreak | s1q | 10.938 | 52.101 | 10.850 | harmful: 5, benign: 123 | harmful: 0.0420, benign: 1.0000 | [[5, 114], [0, 9]] |
| intern-decision-0.8b | mnli | native | 47.541 | 46.807 | 47.079 | entailment: 58, neutral: 51, contradiction: 13 | entailment: 0.6098, neutral: 0.4889, contradiction: 0.3056 | [[25, 15, 1], [22, 22, 1], [11, 14, 11]] |
| intern-decision-0.8b | mnli | rtn | 33.607 | 33.627 | 32.371 | entailment: 68, neutral: 18, contradiction: 36 | entailment: 0.5366, neutral: 0.2222, contradiction: 0.2500 | [[22, 3, 16], [24, 10, 11], [22, 5, 9]] |
| intern-decision-0.8b | mnli | s1q-local | 37.705 | 36.838 | 33.326 | entailment: 78, neutral: 33, contradiction: 11 | entailment: 0.6829, neutral: 0.3111, contradiction: 0.1111 | [[28, 10, 3], [27, 14, 4], [23, 9, 4]] |
| intern-decision-0.8b | mnli | s1q2-beta05 | 33.607 | 32.981 | 26.621 | entailment: 87, neutral: 26, contradiction: 9 | entailment: 0.7561, neutral: 0.1778, contradiction: 0.0556 | [[31, 6, 4], [34, 8, 3], [22, 12, 2]] |
| intern-decision-0.8b | mnli | smoothquant-adapted | 31.148 | 30.678 | 19.792 | entailment: 99, neutral: 19, contradiction: 4 | entailment: 0.8537, neutral: 0.0667, contradiction: 0.0000 | [[35, 4, 2], [40, 3, 2], [24, 12, 0]] |
| intern-decision-0.8b | mnli | awq-adapted | 33.607 | 33.135 | 29.040 | entailment: 83, neutral: 22, contradiction: 17 | entailment: 0.6829, neutral: 0.2000, contradiction: 0.1111 | [[28, 5, 8], [31, 9, 5], [24, 8, 4]] |
| intern-decision-0.8b | mnli | gptq-blockdiag-adapted | 30.328 | 30.397 | 26.248 | entailment: 82, neutral: 17, contradiction: 23 | entailment: 0.6341, neutral: 0.1111, contradiction: 0.1667 | [[26, 6, 9], [32, 5, 8], [24, 6, 6]] |
| intern-decision-0.8b | mnli | spinquant-nohad-adapted | 35.246 | 35.108 | 34.757 | entailment: 67, neutral: 40, contradiction: 15 | entailment: 0.5366, neutral: 0.2667, contradiction: 0.2500 | [[22, 18, 1], [28, 12, 5], [17, 10, 9]] |
| intern-decision-0.8b | mnli | spinquant-had-adapted | 38.525 | 37.136 | 32.645 | entailment: 75, neutral: 41, contradiction: 6 | entailment: 0.6585, neutral: 0.4000, contradiction: 0.0556 | [[27, 12, 2], [25, 18, 2], [23, 11, 2]] |
| intern-decision-0.8b | mnli | s1q-joint | 36.066 | 34.892 | 33.520 | entailment: 57, neutral: 49, contradiction: 16 | entailment: 0.4634, neutral: 0.4444, contradiction: 0.1389 | [[19, 15, 7], [21, 20, 4], [17, 14, 5]] |
| intern-decision-0.8b | mnli | s1q-ac | 32.787 | 32.105 | 29.414 | entailment: 78, neutral: 31, contradiction: 13 | entailment: 0.5854, neutral: 0.2667, contradiction: 0.1111 | [[24, 11, 6], [30, 12, 3], [24, 8, 4]] |
| intern-decision-0.8b | mnli | s1q-margin | 34.426 | 33.297 | 31.463 | entailment: 58, neutral: 49, contradiction: 15 | entailment: 0.4878, neutral: 0.4000, contradiction: 0.1111 | [[20, 18, 3], [19, 18, 8], [19, 13, 4]] |
| intern-decision-0.8b | mnli | s1q | 31.967 | 31.405 | 29.514 | entailment: 67, neutral: 39, contradiction: 16 | entailment: 0.5366, neutral: 0.2667, contradiction: 0.1389 | [[22, 13, 6], [28, 12, 5], [17, 14, 5]] |
| intern-decision-0.8b | tweet-offensive | native | 76.562 | 63.685 | 64.796 | not_offensive: 105, offensive: 23 | not_offensive: 0.8866, offensive: 0.3871 | [[86, 11], [19, 12]] |
| intern-decision-0.8b | tweet-offensive | rtn | 68.750 | 49.751 | 48.718 | not_offensive: 111, offensive: 17 | not_offensive: 0.8660, offensive: 0.1290 | [[84, 13], [27, 4]] |
| intern-decision-0.8b | tweet-offensive | s1q-local | 65.625 | 52.078 | 52.109 | not_offensive: 99, offensive: 29 | not_offensive: 0.7835, offensive: 0.2581 | [[76, 21], [23, 8]] |
| intern-decision-0.8b | tweet-offensive | s1q2-beta05 | 77.344 | 59.810 | 60.906 | not_offensive: 114, offensive: 14 | not_offensive: 0.9381, offensive: 0.2581 | [[91, 6], [23, 8]] |
| intern-decision-0.8b | tweet-offensive | smoothquant-adapted | 60.938 | 45.693 | 45.578 | not_offensive: 99, offensive: 29 | not_offensive: 0.7526, offensive: 0.1613 | [[73, 24], [26, 5]] |
| intern-decision-0.8b | tweet-offensive | awq-adapted | 70.312 | 54.074 | 54.148 | not_offensive: 107, offensive: 21 | not_offensive: 0.8557, offensive: 0.2258 | [[83, 14], [24, 7]] |
| intern-decision-0.8b | tweet-offensive | gptq-blockdiag-adapted | 68.750 | 56.335 | 56.463 | not_offensive: 99, offensive: 29 | not_offensive: 0.8041, offensive: 0.3226 | [[78, 19], [21, 10]] |
| intern-decision-0.8b | tweet-offensive | spinquant-nohad-adapted | 61.719 | 49.501 | 49.480 | not_offensive: 94, offensive: 34 | not_offensive: 0.7320, offensive: 0.2581 | [[71, 26], [23, 8]] |
| intern-decision-0.8b | tweet-offensive | spinquant-had-adapted | 57.031 | 48.603 | 48.140 | not_offensive: 84, offensive: 44 | not_offensive: 0.6495, offensive: 0.3226 | [[63, 34], [21, 10]] |
| intern-decision-0.8b | tweet-offensive | s1q-joint | 56.250 | 44.795 | 44.946 | not_offensive: 89, offensive: 39 | not_offensive: 0.6701, offensive: 0.2258 | [[65, 32], [24, 7]] |
| intern-decision-0.8b | tweet-offensive | s1q-ac | 67.969 | 51.430 | 51.222 | not_offensive: 106, offensive: 22 | not_offensive: 0.8351, offensive: 0.1935 | [[81, 16], [25, 6]] |
| intern-decision-0.8b | tweet-offensive | s1q-margin | 54.688 | 47.057 | 46.528 | not_offensive: 81, offensive: 47 | not_offensive: 0.6186, offensive: 0.3226 | [[60, 37], [21, 10]] |
| intern-decision-0.8b | tweet-offensive | s1q | 64.062 | 55.437 | 54.777 | not_offensive: 89, offensive: 39 | not_offensive: 0.7216, offensive: 0.3871 | [[70, 27], [19, 12]] |
| intern-decision-0.8b | wanli | native | 44.141 | 42.779 | 41.588 | supported: 129, insufficient: 85, contradicted: 42 | supported: 0.6842, insufficient: 0.3462, contradicted: 0.2530 | [[65, 23, 7], [37, 27, 14], [27, 35, 21]] |
| intern-decision-0.8b | wanli | rtn | 35.547 | 35.085 | 34.985 | supported: 98, insufficient: 82, contradicted: 76 | supported: 0.4421, insufficient: 0.3333, contradicted: 0.2771 | [[42, 28, 25], [24, 26, 28], [32, 28, 23]] |
| intern-decision-0.8b | wanli | s1q-local | 31.641 | 31.728 | 31.636 | supported: 80, insufficient: 86, contradicted: 90 | supported: 0.3053, insufficient: 0.3333, contradicted: 0.3133 | [[29, 32, 34], [22, 26, 30], [29, 28, 26]] |
| intern-decision-0.8b | wanli | s1q2-beta05 | 28.906 | 28.993 | 28.896 | supported: 88, insufficient: 85, contradicted: 83 | supported: 0.2842, insufficient: 0.3205, contradicted: 0.2651 | [[27, 31, 37], [29, 25, 24], [32, 29, 22]] |
| intern-decision-0.8b | wanli | smoothquant-adapted | 28.906 | 29.223 | 28.745 | supported: 77, insufficient: 100, contradicted: 79 | supported: 0.2632, insufficient: 0.3846, contradicted: 0.2289 | [[25, 36, 34], [22, 30, 26], [30, 34, 19]] |
| intern-decision-0.8b | wanli | awq-adapted | 32.422 | 32.048 | 32.064 | supported: 94, insufficient: 80, contradicted: 82 | supported: 0.3789, insufficient: 0.2692, contradicted: 0.3133 | [[36, 29, 30], [31, 21, 26], [27, 30, 26]] |
| intern-decision-0.8b | wanli | gptq-blockdiag-adapted | 35.938 | 36.327 | 36.066 | supported: 88, insufficient: 86, contradicted: 82 | supported: 0.3158, insufficient: 0.4487, contradicted: 0.3253 | [[30, 30, 35], [23, 35, 20], [35, 21, 27]] |
| intern-decision-0.8b | wanli | spinquant-nohad-adapted | 32.031 | 32.106 | 31.930 | supported: 87, insufficient: 89, contradicted: 80 | supported: 0.3263, insufficient: 0.3718, contradicted: 0.2651 | [[31, 30, 34], [25, 29, 24], [31, 30, 22]] |
| intern-decision-0.8b | wanli | spinquant-had-adapted | 37.500 | 38.138 | 37.539 | supported: 82, insufficient: 93, contradicted: 81 | supported: 0.2947, insufficient: 0.5000, contradicted: 0.3494 | [[28, 32, 35], [22, 39, 17], [32, 22, 29]] |
| intern-decision-0.8b | wanli | s1q-joint | 39.844 | 38.409 | 37.204 | supported: 139, insufficient: 51, contradicted: 66 | supported: 0.6316, insufficient: 0.2436, contradicted: 0.2771 | [[60, 16, 19], [35, 19, 24], [44, 16, 23]] |
| intern-decision-0.8b | wanli | s1q-ac | 32.812 | 32.500 | 32.517 | supported: 93, insufficient: 80, contradicted: 83 | supported: 0.3684, insufficient: 0.2692, contradicted: 0.3373 | [[35, 30, 30], [32, 21, 25], [26, 29, 28]] |
| intern-decision-0.8b | wanli | s1q-margin | 35.156 | 34.885 | 34.920 | supported: 102, insufficient: 68, contradicted: 86 | supported: 0.3895, insufficient: 0.3077, contradicted: 0.3494 | [[37, 28, 30], [27, 24, 27], [38, 16, 29]] |
| intern-decision-0.8b | wanli | s1q | 32.812 | 32.552 | 32.588 | supported: 105, insufficient: 72, contradicted: 79 | supported: 0.3684, insufficient: 0.2949, contradicted: 0.3133 | [[35, 33, 27], [29, 23, 26], [41, 16, 26]] |
| intern-decision-0.8b | wildjailbreak | native | 58.594 | 67.460 | 46.427 | harmful: 70, benign: 58 | harmful: 0.5714, benign: 0.7778 | [[68, 51], [2, 7]] |
| intern-decision-0.8b | wildjailbreak | rtn | 76.562 | 46.312 | 46.429 | harmful: 105, benign: 23 | harmful: 0.8151, benign: 0.1111 | [[97, 22], [8, 1]] |
| intern-decision-0.8b | wildjailbreak | s1q-local | 43.750 | 38.936 | 33.621 | harmful: 59, benign: 69 | harmful: 0.4454, benign: 0.3333 | [[53, 66], [6, 3]] |
| intern-decision-0.8b | wildjailbreak | s1q2-beta05 | 50.000 | 52.568 | 39.174 | harmful: 63, benign: 65 | harmful: 0.4958, benign: 0.5556 | [[59, 60], [4, 5]] |
| intern-decision-0.8b | wildjailbreak | smoothquant-adapted | 67.969 | 57.096 | 48.260 | harmful: 88, benign: 40 | harmful: 0.6975, benign: 0.4444 | [[83, 36], [5, 4]] |
| intern-decision-0.8b | wildjailbreak | awq-adapted | 53.125 | 49.113 | 39.925 | harmful: 69, benign: 59 | harmful: 0.5378, benign: 0.4444 | [[64, 55], [5, 4]] |
| intern-decision-0.8b | wildjailbreak | gptq-blockdiag-adapted | 83.594 | 75.770 | 63.473 | harmful: 104, benign: 24 | harmful: 0.8487, benign: 0.6667 | [[101, 18], [3, 6]] |
| intern-decision-0.8b | wildjailbreak | spinquant-nohad-adapted | 67.188 | 51.541 | 46.154 | harmful: 89, benign: 39 | harmful: 0.6975, benign: 0.3333 | [[83, 36], [6, 3]] |
| intern-decision-0.8b | wildjailbreak | spinquant-had-adapted | 49.219 | 47.012 | 37.720 | harmful: 64, benign: 64 | harmful: 0.4958, benign: 0.4444 | [[59, 60], [5, 4]] |
| intern-decision-0.8b | wildjailbreak | s1q-joint | 43.750 | 59.477 | 36.963 | harmful: 51, benign: 77 | harmful: 0.4118, benign: 0.7778 | [[49, 70], [2, 7]] |
| intern-decision-0.8b | wildjailbreak | s1q-ac | 55.469 | 50.373 | 41.232 | harmful: 72, benign: 56 | harmful: 0.5630, benign: 0.4444 | [[67, 52], [5, 4]] |
| intern-decision-0.8b | wildjailbreak | s1q-margin | 37.500 | 35.574 | 29.959 | harmful: 51, benign: 77 | harmful: 0.3782, benign: 0.3333 | [[45, 74], [6, 3]] |
| intern-decision-0.8b | wildjailbreak | s1q | 50.000 | 42.297 | 37.081 | harmful: 67, benign: 61 | harmful: 0.5126, benign: 0.3333 | [[61, 58], [6, 3]] |
| intern-decision-2b | mnli | native | 58.197 | 57.701 | 58.194 | entailment: 41, neutral: 54, contradiction: 27 | entailment: 0.5366, neutral: 0.6667, contradiction: 0.5278 | [[22, 13, 6], [13, 30, 2], [6, 11, 19]] |
| intern-decision-2b | mnli | rtn | 32.787 | 33.071 | 31.300 | entailment: 67, neutral: 25, contradiction: 30 | entailment: 0.5366, neutral: 0.1778, contradiction: 0.2778 | [[22, 11, 8], [25, 8, 12], [20, 6, 10]] |
| intern-decision-2b | mnli | s1q-local | 37.705 | 37.087 | 30.744 | entailment: 88, neutral: 21, contradiction: 13 | entailment: 0.8293, neutral: 0.2000, contradiction: 0.0833 | [[34, 3, 4], [30, 9, 6], [24, 9, 3]] |
| intern-decision-2b | mnli | s1q2-beta05 | 36.066 | 35.840 | 33.183 | entailment: 74, neutral: 30, contradiction: 18 | entailment: 0.6585, neutral: 0.2222, contradiction: 0.1944 | [[27, 7, 7], [31, 10, 4], [16, 13, 7]] |
| intern-decision-2b | mnli | smoothquant-adapted | 26.230 | 27.010 | 25.081 | entailment: 62, neutral: 12, contradiction: 48 | entailment: 0.3659, neutral: 0.1111, contradiction: 0.3333 | [[15, 5, 21], [25, 5, 15], [22, 2, 12]] |
| intern-decision-2b | mnli | awq-adapted | 28.689 | 28.844 | 25.072 | entailment: 76, neutral: 25, contradiction: 21 | entailment: 0.6098, neutral: 0.0889, contradiction: 0.1667 | [[25, 10, 6], [32, 4, 9], [19, 11, 6]] |
| intern-decision-2b | mnli | gptq-blockdiag-adapted | 37.705 | 37.918 | 36.003 | entailment: 65, neutral: 23, contradiction: 34 | entailment: 0.6098, neutral: 0.2222, contradiction: 0.3056 | [[25, 9, 7], [19, 10, 16], [21, 4, 11]] |
| intern-decision-2b | mnli | spinquant-nohad-adapted | 32.787 | 31.350 | 29.939 | entailment: 42, neutral: 63, contradiction: 17 | entailment: 0.2683, neutral: 0.5333, contradiction: 0.1389 | [[11, 24, 6], [15, 24, 6], [16, 15, 5]] |
| intern-decision-2b | mnli | spinquant-had-adapted | 34.426 | 34.038 | 33.391 | entailment: 58, neutral: 42, contradiction: 22 | entailment: 0.4878, neutral: 0.3111, contradiction: 0.2222 | [[20, 15, 6], [23, 14, 8], [15, 13, 8]] |
| intern-decision-2b | mnli | s1q-joint | 33.607 | 32.389 | 31.389 | entailment: 36, neutral: 62, contradiction: 24 | entailment: 0.2439, neutral: 0.5333, contradiction: 0.1944 | [[10, 20, 11], [15, 24, 6], [11, 18, 7]] |
| intern-decision-2b | mnli | s1q-ac | 37.705 | 37.227 | 37.290 | entailment: 39, neutral: 49, contradiction: 34 | entailment: 0.4390, neutral: 0.4000, contradiction: 0.2778 | [[18, 14, 9], [12, 18, 15], [9, 17, 10]] |
| intern-decision-2b | mnli | s1q-margin | 31.967 | 32.023 | 27.322 | entailment: 89, neutral: 15, contradiction: 18 | entailment: 0.6829, neutral: 0.1111, contradiction: 0.1667 | [[28, 8, 5], [33, 5, 7], [28, 2, 6]] |
| intern-decision-2b | mnli | s1q | 31.967 | 32.588 | 29.735 | entailment: 74, neutral: 18, contradiction: 30 | entailment: 0.5610, neutral: 0.1111, contradiction: 0.3056 | [[23, 7, 11], [32, 5, 8], [19, 6, 11]] |
| intern-decision-2b | tweet-offensive | native | 78.125 | 77.885 | 73.801 | not_offensive: 83, offensive: 45 | not_offensive: 0.7835, offensive: 0.7742 | [[76, 21], [7, 24]] |
| intern-decision-2b | tweet-offensive | rtn | 57.812 | 53.508 | 51.556 | not_offensive: 77, offensive: 51 | not_offensive: 0.6186, offensive: 0.4516 | [[60, 37], [17, 14]] |
| intern-decision-2b | tweet-offensive | s1q-local | 54.688 | 44.862 | 44.877 | not_offensive: 85, offensive: 43 | not_offensive: 0.6392, offensive: 0.2581 | [[62, 35], [23, 8]] |
| intern-decision-2b | tweet-offensive | s1q2-beta05 | 57.812 | 50.216 | 49.474 | not_offensive: 83, offensive: 45 | not_offensive: 0.6495, offensive: 0.3548 | [[63, 34], [20, 11]] |
| intern-decision-2b | tweet-offensive | smoothquant-adapted | 60.938 | 45.693 | 45.578 | not_offensive: 99, offensive: 29 | not_offensive: 0.7526, offensive: 0.1613 | [[73, 24], [26, 5]] |
| intern-decision-2b | tweet-offensive | awq-adapted | 57.812 | 50.216 | 49.474 | not_offensive: 83, offensive: 45 | not_offensive: 0.6495, offensive: 0.3548 | [[63, 34], [20, 11]] |
| intern-decision-2b | tweet-offensive | gptq-blockdiag-adapted | 59.375 | 47.955 | 47.935 | not_offensive: 91, offensive: 37 | not_offensive: 0.7010, offensive: 0.2581 | [[68, 29], [23, 8]] |
| intern-decision-2b | tweet-offensive | spinquant-nohad-adapted | 56.250 | 60.160 | 53.707 | not_offensive: 61, offensive: 67 | not_offensive: 0.5258, offensive: 0.6774 | [[51, 46], [10, 21]] |
| intern-decision-2b | tweet-offensive | spinquant-had-adapted | 53.906 | 52.029 | 49.189 | not_offensive: 70, offensive: 58 | not_offensive: 0.5567, offensive: 0.4839 | [[54, 43], [16, 15]] |
| intern-decision-2b | tweet-offensive | s1q-joint | 49.219 | 59.910 | 48.689 | not_offensive: 44, offensive: 84 | not_offensive: 0.3918, offensive: 0.8065 | [[38, 59], [6, 25]] |
| intern-decision-2b | tweet-offensive | s1q-ac | 63.281 | 62.604 | 58.610 | not_offensive: 74, offensive: 54 | not_offensive: 0.6392, offensive: 0.6129 | [[62, 35], [12, 19]] |
| intern-decision-2b | tweet-offensive | s1q-margin | 61.719 | 53.891 | 53.050 | not_offensive: 86, offensive: 42 | not_offensive: 0.6907, offensive: 0.3871 | [[67, 30], [19, 12]] |
| intern-decision-2b | tweet-offensive | s1q | 62.500 | 59.894 | 56.939 | not_offensive: 77, offensive: 51 | not_offensive: 0.6495, offensive: 0.5484 | [[63, 34], [14, 17]] |
| intern-decision-2b | wanli | native | 54.297 | 52.742 | 51.650 | supported: 149, insufficient: 71, contradicted: 36 | supported: 0.8211, insufficient: 0.4359, contradicted: 0.3253 | [[78, 12, 5], [40, 34, 4], [31, 25, 27]] |
| intern-decision-2b | wanli | rtn | 33.203 | 33.286 | 33.231 | supported: 84, insufficient: 84, contradicted: 88 | supported: 0.3368, insufficient: 0.3846, contradicted: 0.2771 | [[32, 26, 37], [20, 30, 28], [32, 28, 23]] |
| intern-decision-2b | wanli | s1q-local | 29.688 | 29.592 | 29.564 | supported: 86, insufficient: 81, contradicted: 89 | supported: 0.3053, insufficient: 0.2692, contradicted: 0.3133 | [[29, 32, 34], [28, 21, 29], [29, 28, 26]] |
| intern-decision-2b | wanli | s1q2-beta05 | 35.156 | 35.141 | 35.083 | supported: 88, insufficient: 88, contradicted: 80 | supported: 0.3579, insufficient: 0.3590, contradicted: 0.3373 | [[34, 34, 27], [25, 28, 25], [29, 26, 28]] |
| intern-decision-2b | wanli | smoothquant-adapted | 30.078 | 30.401 | 30.090 | supported: 76, insufficient: 90, contradicted: 90 | supported: 0.2526, insufficient: 0.3462, contradicted: 0.3133 | [[24, 32, 39], [26, 27, 25], [26, 31, 26]] |
| intern-decision-2b | wanli | awq-adapted | 39.844 | 40.064 | 39.880 | supported: 81, insufficient: 85, contradicted: 90 | supported: 0.3684, insufficient: 0.4359, contradicted: 0.3976 | [[35, 25, 35], [22, 34, 22], [24, 26, 33]] |
| intern-decision-2b | wanli | gptq-blockdiag-adapted | 37.500 | 37.755 | 37.480 | supported: 81, insufficient: 92, contradicted: 83 | supported: 0.3474, insufficient: 0.4359, contradicted: 0.3494 | [[33, 29, 33], [23, 34, 21], [25, 29, 29]] |
| intern-decision-2b | wanli | spinquant-nohad-adapted | 32.812 | 32.883 | 32.872 | supported: 75, insufficient: 88, contradicted: 93 | supported: 0.3263, insufficient: 0.3590, contradicted: 0.3012 | [[31, 28, 36], [18, 28, 32], [26, 32, 25]] |
| intern-decision-2b | wanli | spinquant-had-adapted | 34.375 | 33.751 | 33.607 | supported: 105, insufficient: 70, contradicted: 81 | supported: 0.4316, insufficient: 0.2436, contradicted: 0.3373 | [[41, 29, 25], [31, 19, 28], [33, 22, 28]] |
| intern-decision-2b | wanli | s1q-joint | 36.328 | 35.762 | 35.420 | supported: 110, insufficient: 90, contradicted: 56 | supported: 0.4737, insufficient: 0.3462, contradicted: 0.2530 | [[45, 31, 19], [35, 27, 16], [30, 32, 21]] |
| intern-decision-2b | wanli | s1q-ac | 31.641 | 31.423 | 31.447 | supported: 90, insufficient: 94, contradicted: 72 | supported: 0.3579, insufficient: 0.3077, contradicted: 0.2771 | [[34, 32, 29], [34, 24, 20], [22, 38, 23]] |
| intern-decision-2b | wanli | s1q-margin | 37.109 | 36.133 | 34.970 | supported: 131, insufficient: 76, contradicted: 49 | supported: 0.5579, insufficient: 0.3333, contradicted: 0.1928 | [[53, 23, 19], [38, 26, 14], [40, 27, 16]] |
| intern-decision-2b | wanli | s1q | 35.938 | 35.309 | 34.814 | supported: 114, insufficient: 79, contradicted: 63 | supported: 0.4842, insufficient: 0.3462, contradicted: 0.2289 | [[46, 26, 23], [30, 27, 21], [38, 26, 19]] |
| intern-decision-2b | wildjailbreak | native | 72.656 | 75.023 | 55.832 | harmful: 88, benign: 40 | harmful: 0.7227, benign: 0.7778 | [[86, 33], [2, 7]] |
| intern-decision-2b | wildjailbreak | rtn | 62.500 | 59.290 | 46.499 | harmful: 79, benign: 49 | harmful: 0.6303, benign: 0.5556 | [[75, 44], [4, 5]] |
| intern-decision-2b | wildjailbreak | s1q-local | 59.375 | 67.880 | 46.922 | harmful: 71, benign: 57 | harmful: 0.5798, benign: 0.7778 | [[69, 50], [2, 7]] |
| intern-decision-2b | wildjailbreak | s1q2-beta05 | 50.000 | 57.703 | 40.117 | harmful: 61, benign: 67 | harmful: 0.4874, benign: 0.6667 | [[58, 61], [3, 6]] |
| intern-decision-2b | wildjailbreak | smoothquant-adapted | 79.688 | 47.993 | 47.870 | harmful: 109, benign: 19 | harmful: 0.8487, benign: 0.1111 | [[101, 18], [8, 1]] |
| intern-decision-2b | wildjailbreak | awq-adapted | 54.688 | 44.818 | 39.583 | harmful: 73, benign: 55 | harmful: 0.5630, benign: 0.3333 | [[67, 52], [6, 3]] |
| intern-decision-2b | wildjailbreak | gptq-blockdiag-adapted | 62.500 | 54.155 | 45.143 | harmful: 81, benign: 47 | harmful: 0.6387, benign: 0.4444 | [[76, 43], [5, 4]] |
| intern-decision-2b | wildjailbreak | spinquant-nohad-adapted | 57.812 | 36.228 | 38.286 | harmful: 81, benign: 47 | harmful: 0.6134, benign: 0.1111 | [[73, 46], [8, 1]] |
| intern-decision-2b | wildjailbreak | spinquant-had-adapted | 35.156 | 49.720 | 30.539 | harmful: 42, benign: 86 | harmful: 0.3277, benign: 0.6667 | [[39, 80], [3, 6]] |
| intern-decision-2b | wildjailbreak | s1q-joint | 21.875 | 57.983 | 21.395 | harmful: 19, benign: 109 | harmful: 0.1597, benign: 1.0000 | [[19, 100], [0, 9]] |
| intern-decision-2b | wildjailbreak | s1q-ac | 17.969 | 50.747 | 17.723 | harmful: 16, benign: 112 | harmful: 0.1261, benign: 0.8889 | [[15, 104], [1, 8]] |
| intern-decision-2b | wildjailbreak | s1q-margin | 32.812 | 53.595 | 29.436 | harmful: 37, benign: 91 | harmful: 0.2941, benign: 0.7778 | [[35, 84], [2, 7]] |
| intern-decision-2b | wildjailbreak | s1q | 44.531 | 59.897 | 37.475 | harmful: 52, benign: 76 | harmful: 0.4202, benign: 0.7778 | [[50, 69], [2, 7]] |
| intern-decision-4b | mnli | native | 78.689 | 79.061 | 78.878 | entailment: 36, neutral: 48, contradiction: 38 | entailment: 0.6829, neutral: 0.8000, contradiction: 0.8889 | [[28, 10, 3], [6, 36, 3], [2, 2, 32]] |
| intern-decision-4b | mnli | rtn | 30.328 | 31.012 | 30.088 | entailment: 48, neutral: 28, contradiction: 46 | entailment: 0.3415, neutral: 0.2000, contradiction: 0.3889 | [[14, 11, 16], [20, 9, 16], [14, 8, 14]] |
| intern-decision-4b | mnli | s1q-local | 36.885 | 35.212 | 29.990 | entailment: 69, neutral: 44, contradiction: 9 | entailment: 0.6341, neutral: 0.4222, contradiction: 0.0000 | [[26, 11, 4], [21, 19, 5], [22, 14, 0]] |
| intern-decision-4b | mnli | s1q2-beta05 | 37.705 | 37.136 | 34.637 | entailment: 69, neutral: 32, contradiction: 21 | entailment: 0.6585, neutral: 0.2889, contradiction: 0.1667 | [[27, 8, 6], [23, 13, 9], [19, 11, 6]] |
| intern-decision-4b | mnli | smoothquant-adapted | 28.689 | 28.451 | 27.892 | entailment: 58, neutral: 30, contradiction: 34 | entailment: 0.4146, neutral: 0.2444, contradiction: 0.1944 | [[17, 10, 14], [21, 11, 13], [20, 9, 7]] |
| intern-decision-4b | mnli | awq-adapted | 29.508 | 28.306 | 27.079 | entailment: 54, neutral: 47, contradiction: 21 | entailment: 0.3659, neutral: 0.4000, contradiction: 0.0833 | [[15, 15, 11], [20, 18, 7], [19, 14, 3]] |
| intern-decision-4b | mnli | gptq-blockdiag-adapted | 32.787 | 32.547 | 29.883 | entailment: 75, neutral: 29, contradiction: 18 | entailment: 0.6098, neutral: 0.2000, contradiction: 0.1667 | [[25, 11, 5], [29, 9, 7], [21, 9, 6]] |
| intern-decision-4b | mnli | spinquant-nohad-adapted | 40.164 | 38.555 | 36.457 | entailment: 55, neutral: 50, contradiction: 17 | entailment: 0.5122, neutral: 0.5333, contradiction: 0.1111 | [[21, 15, 5], [13, 24, 8], [21, 11, 4]] |
| intern-decision-4b | mnli | spinquant-had-adapted | 38.525 | 37.805 | 35.731 | entailment: 74, neutral: 34, contradiction: 14 | entailment: 0.6341, neutral: 0.3333, contradiction: 0.1667 | [[26, 11, 4], [26, 15, 4], [22, 8, 6]] |
| intern-decision-4b | mnli | s1q-joint | 36.885 | 36.003 | 35.713 | entailment: 40, neutral: 58, contradiction: 24 | entailment: 0.4634, neutral: 0.4222, contradiction: 0.1944 | [[19, 19, 3], [12, 19, 14], [9, 20, 7]] |
| intern-decision-4b | mnli | s1q-ac | 48.361 | 47.805 | 47.666 | entailment: 50, neutral: 51, contradiction: 21 | entailment: 0.6341, neutral: 0.4667, contradiction: 0.3333 | [[26, 13, 2], [17, 21, 7], [7, 17, 12]] |
| intern-decision-4b | mnli | s1q-margin | 31.148 | 30.578 | 29.954 | entailment: 29, neutral: 58, contradiction: 35 | entailment: 0.1951, neutral: 0.4444, contradiction: 0.2778 | [[8, 24, 9], [9, 20, 16], [12, 14, 10]] |
| intern-decision-4b | mnli | s1q | 40.984 | 40.673 | 41.058 | entailment: 44, neutral: 52, contradiction: 26 | entailment: 0.4146, neutral: 0.4444, contradiction: 0.3611 | [[17, 19, 5], [17, 20, 8], [10, 13, 13]] |
| intern-decision-4b | tweet-offensive | native | 81.250 | 75.557 | 75.000 | not_offensive: 95, offensive: 33 | not_offensive: 0.8660, offensive: 0.6452 | [[84, 13], [11, 20]] |
| intern-decision-4b | tweet-offensive | rtn | 57.812 | 52.411 | 50.909 | not_offensive: 79, offensive: 49 | not_offensive: 0.6289, offensive: 0.4194 | [[61, 36], [18, 13]] |
| intern-decision-4b | tweet-offensive | s1q-local | 56.250 | 50.283 | 49.091 | not_offensive: 79, offensive: 49 | not_offensive: 0.6186, offensive: 0.3871 | [[60, 37], [19, 12]] |
| intern-decision-4b | tweet-offensive | s1q2-beta05 | 51.562 | 48.287 | 46.320 | not_offensive: 71, offensive: 57 | not_offensive: 0.5464, offensive: 0.4194 | [[53, 44], [18, 13]] |
| intern-decision-4b | tweet-offensive | smoothquant-adapted | 64.062 | 54.340 | 53.942 | not_offensive: 91, offensive: 37 | not_offensive: 0.7320, offensive: 0.3548 | [[71, 26], [20, 11]] |
| intern-decision-4b | tweet-offensive | awq-adapted | 46.875 | 47.389 | 43.787 | not_offensive: 61, offensive: 67 | not_offensive: 0.4639, offensive: 0.4839 | [[45, 52], [16, 15]] |
| intern-decision-4b | tweet-offensive | gptq-blockdiag-adapted | 59.375 | 51.247 | 50.579 | not_offensive: 85, offensive: 43 | not_offensive: 0.6701, offensive: 0.3548 | [[65, 32], [20, 11]] |
| intern-decision-4b | tweet-offensive | spinquant-nohad-adapted | 50.000 | 50.549 | 46.667 | not_offensive: 63, offensive: 65 | not_offensive: 0.4948, offensive: 0.5161 | [[48, 49], [15, 16]] |
| intern-decision-4b | tweet-offensive | spinquant-had-adapted | 55.469 | 49.767 | 48.529 | not_offensive: 78, offensive: 50 | not_offensive: 0.6082, offensive: 0.3871 | [[59, 38], [19, 12]] |
| intern-decision-4b | tweet-offensive | s1q-joint | 40.625 | 58.630 | 40.494 | not_offensive: 25, offensive: 103 | not_offensive: 0.2371, offensive: 0.9355 | [[23, 74], [2, 29]] |
| intern-decision-4b | tweet-offensive | s1q-ac | 57.031 | 65.065 | 55.598 | not_offensive: 54, offensive: 74 | not_offensive: 0.4948, offensive: 0.8065 | [[48, 49], [6, 25]] |
| intern-decision-4b | tweet-offensive | s1q-margin | 55.469 | 67.326 | 54.849 | not_offensive: 46, offensive: 82 | not_offensive: 0.4433, offensive: 0.9032 | [[43, 54], [3, 28]] |
| intern-decision-4b | tweet-offensive | s1q | 67.969 | 64.599 | 61.924 | not_offensive: 82, offensive: 46 | not_offensive: 0.7113, offensive: 0.5806 | [[69, 28], [13, 18]] |
| intern-decision-4b | wanli | native | 67.188 | 66.301 | 66.495 | supported: 110, insufficient: 75, contradicted: 71 | supported: 0.7895, insufficient: 0.5128, contradicted: 0.6867 | [[75, 18, 2], [26, 40, 12], [9, 17, 57]] |
| intern-decision-4b | wanli | rtn | 33.984 | 33.629 | 33.641 | supported: 97, insufficient: 79, contradicted: 80 | supported: 0.3895, insufficient: 0.2821, contradicted: 0.3373 | [[37, 30, 28], [32, 22, 24], [28, 27, 28]] |
| intern-decision-4b | wanli | s1q-local | 34.766 | 34.306 | 34.270 | supported: 103, insufficient: 70, contradicted: 83 | supported: 0.4211, insufficient: 0.2949, contradicted: 0.3133 | [[40, 23, 32], [30, 23, 25], [33, 24, 26]] |
| intern-decision-4b | wanli | s1q2-beta05 | 31.250 | 31.224 | 31.301 | supported: 101, insufficient: 80, contradicted: 75 | supported: 0.3158, insufficient: 0.3077, contradicted: 0.3133 | [[30, 35, 30], [35, 24, 19], [36, 21, 26]] |
| intern-decision-4b | wanli | smoothquant-adapted | 30.859 | 31.282 | 30.835 | supported: 80, insufficient: 93, contradicted: 83 | supported: 0.2632, insufficient: 0.4103, contradicted: 0.2651 | [[25, 31, 39], [24, 32, 22], [31, 30, 22]] |
| intern-decision-4b | wanli | awq-adapted | 30.078 | 30.148 | 29.868 | supported: 100, insufficient: 87, contradicted: 69 | supported: 0.3158, insufficient: 0.3718, contradicted: 0.2169 | [[30, 35, 30], [28, 29, 21], [42, 23, 18]] |
| intern-decision-4b | wanli | gptq-blockdiag-adapted | 38.672 | 38.961 | 38.685 | supported: 82, insufficient: 89, contradicted: 85 | supported: 0.3579, insufficient: 0.4615, contradicted: 0.3494 | [[34, 29, 32], [18, 36, 24], [30, 24, 29]] |
| intern-decision-4b | wanli | spinquant-nohad-adapted | 34.766 | 34.688 | 34.662 | supported: 88, insufficient: 83, contradicted: 85 | supported: 0.3684, insufficient: 0.3590, contradicted: 0.3133 | [[35, 29, 31], [22, 28, 28], [31, 26, 26]] |
| intern-decision-4b | wanli | spinquant-had-adapted | 30.078 | 30.326 | 30.061 | supported: 89, insufficient: 88, contradicted: 79 | supported: 0.2842, insufficient: 0.3846, contradicted: 0.2410 | [[27, 31, 37], [26, 30, 22], [36, 27, 20]] |
| intern-decision-4b | wanli | s1q-joint | 39.453 | 37.778 | 36.008 | supported: 152, insufficient: 46, contradicted: 58 | supported: 0.6632, insufficient: 0.2051, contradicted: 0.2651 | [[63, 14, 18], [44, 16, 18], [45, 16, 22]] |
| intern-decision-4b | wanli | s1q-ac | 41.406 | 40.219 | 39.486 | supported: 130, insufficient: 50, contradicted: 76 | supported: 0.5895, insufficient: 0.2436, contradicted: 0.3735 | [[56, 15, 24], [38, 19, 21], [36, 16, 31]] |
| intern-decision-4b | wanli | s1q-margin | 43.359 | 42.790 | 42.538 | supported: 121, insufficient: 81, contradicted: 54 | supported: 0.5474, insufficient: 0.4231, contradicted: 0.3133 | [[52, 29, 14], [31, 33, 14], [38, 19, 26]] |
| intern-decision-4b | wanli | s1q | 41.797 | 41.207 | 41.312 | supported: 109, insufficient: 87, contradicted: 60 | supported: 0.5158, insufficient: 0.3590, contradicted: 0.3614 | [[49, 34, 12], [32, 28, 18], [28, 25, 30]] |
| intern-decision-4b | wildjailbreak | native | 86.719 | 87.722 | 70.431 | harmful: 104, benign: 24 | harmful: 0.8655, benign: 0.8889 | [[103, 16], [1, 8]] |
| intern-decision-4b | wildjailbreak | rtn | 60.938 | 53.315 | 44.270 | harmful: 79, benign: 49 | harmful: 0.6218, benign: 0.4444 | [[74, 45], [5, 4]] |
| intern-decision-4b | wildjailbreak | s1q-local | 56.250 | 66.200 | 44.946 | harmful: 67, benign: 61 | harmful: 0.5462, benign: 0.7778 | [[65, 54], [2, 7]] |
| intern-decision-4b | wildjailbreak | s1q2-beta05 | 49.219 | 31.606 | 34.297 | harmful: 70, benign: 58 | harmful: 0.5210, benign: 0.1111 | [[62, 57], [8, 1]] |
| intern-decision-4b | wildjailbreak | smoothquant-adapted | 61.719 | 58.870 | 46.038 | harmful: 78, benign: 50 | harmful: 0.6218, benign: 0.5556 | [[74, 45], [4, 5]] |
| intern-decision-4b | wildjailbreak | awq-adapted | 32.031 | 48.039 | 28.354 | harmful: 38, benign: 90 | harmful: 0.2941, benign: 0.6667 | [[35, 84], [3, 6]] |
| intern-decision-4b | wildjailbreak | gptq-blockdiag-adapted | 62.500 | 43.884 | 42.081 | harmful: 85, benign: 43 | harmful: 0.6555, benign: 0.2222 | [[78, 41], [7, 2]] |
| intern-decision-4b | wildjailbreak | spinquant-nohad-adapted | 43.750 | 38.936 | 33.621 | harmful: 59, benign: 69 | harmful: 0.4454, benign: 0.3333 | [[53, 66], [6, 3]] |
| intern-decision-4b | wildjailbreak | spinquant-had-adapted | 53.906 | 39.262 | 37.890 | harmful: 74, benign: 54 | harmful: 0.5630, benign: 0.2222 | [[67, 52], [7, 2]] |
| intern-decision-4b | wildjailbreak | s1q-joint | 29.688 | 62.185 | 27.928 | harmful: 29, benign: 99 | harmful: 0.2437, benign: 1.0000 | [[29, 90], [0, 9]] |
| intern-decision-4b | wildjailbreak | s1q-ac | 46.875 | 66.293 | 39.756 | harmful: 53, benign: 75 | harmful: 0.4370, benign: 0.8889 | [[52, 67], [1, 8]] |
| intern-decision-4b | wildjailbreak | s1q-margin | 22.656 | 53.268 | 21.850 | harmful: 22, benign: 106 | harmful: 0.1765, benign: 0.8889 | [[21, 98], [1, 8]] |
| intern-decision-4b | wildjailbreak | s1q | 35.156 | 54.855 | 31.116 | harmful: 40, benign: 88 | harmful: 0.3193, benign: 0.7778 | [[38, 81], [2, 7]] |
| startlux-decision-0.8b | mnli | native | 88.525 | 89.083 | 88.652 | entailment: 41, neutral: 41, contradiction: 40 | entailment: 0.8780, neutral: 0.8222, contradiction: 0.9722 | [[36, 4, 1], [4, 37, 4], [1, 0, 35]] |
| startlux-decision-0.8b | mnli | rtn | 32.787 | 32.669 | 32.012 | entailment: 59, neutral: 32, contradiction: 31 | entailment: 0.4634, neutral: 0.2667, contradiction: 0.2500 | [[19, 11, 11], [22, 12, 11], [18, 9, 9]] |
| startlux-decision-0.8b | mnli | s1q-local | 42.623 | 45.754 | 37.960 | entailment: 21, neutral: 12, contradiction: 89 | entailment: 0.3171, neutral: 0.1111, contradiction: 0.9444 | [[13, 6, 22], [7, 5, 33], [1, 1, 34]] |
| startlux-decision-0.8b | mnli | s1q2-beta05 | 45.902 | 48.306 | 44.024 | entailment: 22, neutral: 16, contradiction: 84 | entailment: 0.3659, neutral: 0.2222, contradiction: 0.8611 | [[15, 3, 23], [5, 10, 30], [2, 3, 31]] |
| startlux-decision-0.8b | mnli | smoothquant-adapted | 34.426 | 33.564 | 33.194 | entailment: 51, neutral: 54, contradiction: 17 | entailment: 0.3902, neutral: 0.4222, contradiction: 0.1944 | [[16, 19, 6], [22, 19, 4], [13, 16, 7]] |
| startlux-decision-0.8b | mnli | awq-adapted | 38.525 | 40.578 | 35.801 | entailment: 15, neutral: 28, contradiction: 79 | entailment: 0.1951, neutral: 0.2444, contradiction: 0.7778 | [[8, 11, 22], [5, 11, 29], [2, 6, 28]] |
| startlux-decision-0.8b | mnli | gptq-blockdiag-adapted | 31.148 | 30.723 | 30.533 | entailment: 30, neutral: 52, contradiction: 40 | entailment: 0.2439, neutral: 0.4000, contradiction: 0.2778 | [[10, 16, 15], [12, 18, 15], [8, 18, 10]] |
| startlux-decision-0.8b | mnli | spinquant-nohad-adapted | 38.525 | 38.419 | 38.382 | entailment: 32, neutral: 49, contradiction: 41 | entailment: 0.3415, neutral: 0.4222, contradiction: 0.3889 | [[14, 15, 12], [11, 19, 15], [7, 15, 14]] |
| startlux-decision-0.8b | mnli | spinquant-had-adapted | 30.328 | 33.889 | 17.972 | entailment: 1, neutral: 4, contradiction: 117 | entailment: 0.0000, neutral: 0.0444, contradiction: 0.9722 | [[0, 1, 40], [1, 2, 42], [0, 1, 35]] |
| startlux-decision-0.8b | mnli | s1q-joint | 40.984 | 42.534 | 40.125 | entailment: 21, neutral: 26, contradiction: 75 | entailment: 0.2927, neutral: 0.2889, contradiction: 0.6944 | [[12, 5, 24], [6, 13, 26], [3, 8, 25]] |
| startlux-decision-0.8b | mnli | s1q-ac | 40.164 | 41.608 | 39.163 | entailment: 27, neutral: 29, contradiction: 66 | entailment: 0.2927, neutral: 0.2889, contradiction: 0.6667 | [[12, 9, 20], [10, 13, 22], [5, 7, 24]] |
| startlux-decision-0.8b | mnli | s1q-margin | 43.443 | 44.765 | 40.695 | entailment: 11, neutral: 45, contradiction: 66 | entailment: 0.1707, neutral: 0.4222, contradiction: 0.7500 | [[7, 19, 15], [2, 19, 24], [2, 7, 27]] |
| startlux-decision-0.8b | mnli | s1q | 50.000 | 50.375 | 49.972 | entailment: 32, neutral: 43, contradiction: 47 | entailment: 0.4390, neutral: 0.4889, contradiction: 0.5833 | [[18, 13, 10], [7, 22, 16], [7, 8, 21]] |
| startlux-decision-0.8b | tweet-offensive | native | 83.594 | 72.714 | 75.016 | not_offensive: 106, offensive: 22 | not_offensive: 0.9381, offensive: 0.5161 | [[91, 6], [15, 16]] |
| startlux-decision-0.8b | tweet-offensive | rtn | 51.562 | 54.872 | 49.128 | not_offensive: 59, offensive: 69 | not_offensive: 0.4845, offensive: 0.6129 | [[47, 50], [12, 19]] |
| startlux-decision-0.8b | tweet-offensive | s1q-local | 77.344 | 55.421 | 54.190 | not_offensive: 122, offensive: 6 | not_offensive: 0.9794, offensive: 0.1290 | [[95, 2], [27, 4]] |
| startlux-decision-0.8b | tweet-offensive | s1q2-beta05 | 76.562 | 51.613 | 46.429 | not_offensive: 127, offensive: 1 | not_offensive: 1.0000, offensive: 0.0323 | [[97, 0], [30, 1]] |
| startlux-decision-0.8b | tweet-offensive | smoothquant-adapted | 54.688 | 50.349 | 48.616 | not_offensive: 75, offensive: 53 | not_offensive: 0.5876, offensive: 0.4194 | [[57, 40], [18, 13]] |
| startlux-decision-0.8b | tweet-offensive | awq-adapted | 75.000 | 52.777 | 50.555 | not_offensive: 121, offensive: 7 | not_offensive: 0.9588, offensive: 0.0968 | [[93, 4], [28, 3]] |
| startlux-decision-0.8b | tweet-offensive | gptq-blockdiag-adapted | 53.906 | 54.223 | 50.181 | not_offensive: 66, offensive: 62 | not_offensive: 0.5361, offensive: 0.5484 | [[52, 45], [14, 17]] |
| startlux-decision-0.8b | tweet-offensive | spinquant-nohad-adapted | 62.500 | 51.114 | 51.005 | not_offensive: 93, offensive: 35 | not_offensive: 0.7320, offensive: 0.2903 | [[71, 26], [22, 9]] |
| startlux-decision-0.8b | tweet-offensive | spinquant-had-adapted | 72.656 | 50.133 | 47.064 | not_offensive: 120, offensive: 8 | not_offensive: 0.9381, offensive: 0.0645 | [[91, 6], [29, 2]] |
| startlux-decision-0.8b | tweet-offensive | s1q-joint | 75.781 | 53.292 | 51.030 | not_offensive: 122, offensive: 6 | not_offensive: 0.9691, offensive: 0.0968 | [[94, 3], [28, 3]] |
| startlux-decision-0.8b | tweet-offensive | s1q-ac | 75.781 | 52.195 | 48.701 | not_offensive: 124, offensive: 4 | not_offensive: 0.9794, offensive: 0.0645 | [[95, 2], [29, 2]] |
| startlux-decision-0.8b | tweet-offensive | s1q-margin | 78.125 | 57.034 | 56.736 | not_offensive: 121, offensive: 7 | not_offensive: 0.9794, offensive: 0.1613 | [[95, 2], [26, 5]] |
| startlux-decision-0.8b | tweet-offensive | s1q | 77.344 | 54.323 | 52.010 | not_offensive: 124, offensive: 4 | not_offensive: 0.9897, offensive: 0.0968 | [[96, 1], [28, 3]] |
| startlux-decision-0.8b | wanli | native | 73.438 | 71.913 | 71.232 | supported: 128, insufficient: 42, contradicted: 86 | supported: 0.9263, insufficient: 0.4359, contradicted: 0.7952 | [[88, 3, 4], [28, 34, 16], [12, 5, 66]] |
| startlux-decision-0.8b | wanli | rtn | 40.234 | 40.619 | 40.227 | supported: 76, insufficient: 93, contradicted: 87 | supported: 0.3579, insufficient: 0.4872, contradicted: 0.3735 | [[34, 28, 33], [17, 38, 23], [25, 27, 31]] |
| startlux-decision-0.8b | wanli | s1q-local | 41.406 | 41.949 | 41.418 | supported: 63, insufficient: 106, contradicted: 87 | supported: 0.3158, insufficient: 0.4487, contradicted: 0.4940 | [[30, 41, 24], [21, 35, 22], [12, 30, 41]] |
| startlux-decision-0.8b | wanli | s1q2-beta05 | 43.359 | 44.266 | 43.386 | supported: 56, insufficient: 131, contradicted: 69 | supported: 0.3053, insufficient: 0.5769, contradicted: 0.4458 | [[29, 51, 15], [16, 45, 17], [11, 35, 37]] |
| startlux-decision-0.8b | wanli | smoothquant-adapted | 31.250 | 31.046 | 31.148 | supported: 87, insufficient: 89, contradicted: 80 | supported: 0.3368, insufficient: 0.2692, contradicted: 0.3253 | [[32, 39, 24], [28, 21, 29], [27, 29, 27]] |
| startlux-decision-0.8b | wanli | awq-adapted | 41.016 | 41.497 | 41.031 | supported: 58, insufficient: 105, contradicted: 93 | supported: 0.3263, insufficient: 0.4487, contradicted: 0.4699 | [[31, 38, 26], [15, 35, 28], [12, 32, 39]] |
| startlux-decision-0.8b | wanli | gptq-blockdiag-adapted | 35.156 | 35.038 | 35.060 | supported: 82, insufficient: 83, contradicted: 91 | supported: 0.3684, insufficient: 0.3333, contradicted: 0.3494 | [[35, 27, 33], [23, 26, 29], [24, 30, 29]] |
| startlux-decision-0.8b | wanli | spinquant-nohad-adapted | 34.375 | 34.516 | 34.365 | supported: 72, insufficient: 100, contradicted: 84 | supported: 0.3368, insufficient: 0.3974, contradicted: 0.3012 | [[32, 36, 27], [15, 31, 32], [25, 33, 25]] |
| startlux-decision-0.8b | wanli | spinquant-had-adapted | 37.891 | 37.520 | 37.816 | supported: 87, insufficient: 95, contradicted: 74 | supported: 0.4316, insufficient: 0.3205, contradicted: 0.3735 | [[41, 36, 18], [28, 25, 25], [18, 34, 31]] |
| startlux-decision-0.8b | wanli | s1q-joint | 45.312 | 44.844 | 44.208 | supported: 85, insufficient: 69, contradicted: 102 | supported: 0.4737, insufficient: 0.2692, contradicted: 0.6024 | [[45, 31, 19], [24, 21, 33], [16, 17, 50]] |
| startlux-decision-0.8b | wanli | s1q-ac | 44.922 | 44.775 | 44.616 | supported: 80, insufficient: 82, contradicted: 94 | supported: 0.4421, insufficient: 0.3590, contradicted: 0.5422 | [[42, 34, 19], [20, 28, 30], [18, 20, 45]] |
| startlux-decision-0.8b | wanli | s1q-margin | 45.703 | 46.041 | 46.461 | supported: 77, insufficient: 135, contradicted: 44 | supported: 0.4316, insufficient: 0.5641, contradicted: 0.3855 | [[41, 51, 3], [25, 44, 9], [11, 40, 32]] |
| startlux-decision-0.8b | wanli | s1q | 40.234 | 40.976 | 40.246 | supported: 66, insufficient: 126, contradicted: 64 | supported: 0.3053, insufficient: 0.5385, contradicted: 0.3855 | [[29, 51, 15], [19, 42, 17], [18, 33, 32]] |
| startlux-decision-0.8b | wildjailbreak | native | 86.719 | 82.586 | 68.803 | harmful: 106, benign: 22 | harmful: 0.8739, benign: 0.7778 | [[104, 15], [2, 7]] |
| startlux-decision-0.8b | wildjailbreak | rtn | 53.906 | 39.262 | 37.890 | harmful: 74, benign: 54 | harmful: 0.5630, benign: 0.2222 | [[67, 52], [7, 2]] |
| startlux-decision-0.8b | wildjailbreak | s1q-local | 28.125 | 61.345 | 26.675 | harmful: 27, benign: 101 | harmful: 0.2269, benign: 1.0000 | [[27, 92], [0, 9]] |
| startlux-decision-0.8b | wildjailbreak | s1q2-beta05 | 28.125 | 61.345 | 26.675 | harmful: 27, benign: 101 | harmful: 0.2269, benign: 1.0000 | [[27, 92], [0, 9]] |
| startlux-decision-0.8b | wildjailbreak | smoothquant-adapted | 41.406 | 53.081 | 34.707 | harmful: 50, benign: 78 | harmful: 0.3950, benign: 0.6667 | [[47, 72], [3, 6]] |
| startlux-decision-0.8b | wildjailbreak | awq-adapted | 16.406 | 55.042 | 16.360 | harmful: 12, benign: 116 | harmful: 0.1008, benign: 1.0000 | [[12, 107], [0, 9]] |
| startlux-decision-0.8b | wildjailbreak | gptq-blockdiag-adapted | 53.906 | 59.804 | 42.505 | harmful: 66, benign: 62 | harmful: 0.5294, benign: 0.6667 | [[63, 56], [3, 6]] |
| startlux-decision-0.8b | wildjailbreak | spinquant-nohad-adapted | 18.750 | 56.303 | 18.571 | harmful: 15, benign: 113 | harmful: 0.1261, benign: 1.0000 | [[15, 104], [0, 9]] |
| startlux-decision-0.8b | wildjailbreak | spinquant-had-adapted | 33.594 | 64.286 | 30.960 | harmful: 34, benign: 94 | harmful: 0.2857, benign: 1.0000 | [[34, 85], [0, 9]] |
| startlux-decision-0.8b | wildjailbreak | s1q-joint | 14.844 | 49.066 | 14.797 | harmful: 12, benign: 116 | harmful: 0.0924, benign: 0.8889 | [[11, 108], [1, 8]] |
| startlux-decision-0.8b | wildjailbreak | s1q-ac | 15.625 | 54.622 | 15.604 | harmful: 11, benign: 117 | harmful: 0.0924, benign: 1.0000 | [[11, 108], [0, 9]] |
| startlux-decision-0.8b | wildjailbreak | s1q-margin | 21.875 | 52.848 | 21.182 | harmful: 21, benign: 107 | harmful: 0.1681, benign: 0.8889 | [[20, 99], [1, 8]] |
| startlux-decision-0.8b | wildjailbreak | s1q | 17.969 | 50.747 | 17.723 | harmful: 16, benign: 112 | harmful: 0.1261, benign: 0.8889 | [[15, 104], [1, 8]] |
| startlux-decision-2b | mnli | native | 89.344 | 89.968 | 89.484 | entailment: 45, neutral: 38, contradiction: 39 | entailment: 0.9268, neutral: 0.8000, contradiction: 0.9722 | [[38, 2, 1], [6, 36, 3], [1, 0, 35]] |
| startlux-decision-2b | mnli | rtn | 37.705 | 38.329 | 36.519 | entailment: 65, neutral: 21, contradiction: 36 | entailment: 0.5610, neutral: 0.2000, contradiction: 0.3889 | [[23, 7, 11], [25, 9, 11], [17, 5, 14]] |
| startlux-decision-2b | mnli | s1q-local | 31.967 | 32.322 | 27.695 | entailment: 77, neutral: 13, contradiction: 32 | entailment: 0.6585, neutral: 0.0889, contradiction: 0.2222 | [[27, 4, 10], [27, 4, 14], [23, 5, 8]] |
| startlux-decision-2b | mnli | s1q2-beta05 | 37.705 | 37.742 | 37.350 | entailment: 57, neutral: 28, contradiction: 37 | entailment: 0.4878, neutral: 0.3111, contradiction: 0.3333 | [[20, 10, 11], [17, 14, 14], [20, 4, 12]] |
| startlux-decision-2b | mnli | smoothquant-adapted | 29.508 | 29.214 | 25.501 | entailment: 80, neutral: 22, contradiction: 20 | entailment: 0.6098, neutral: 0.1556, contradiction: 0.1111 | [[25, 8, 8], [30, 7, 8], [25, 7, 4]] |
| startlux-decision-2b | mnli | awq-adapted | 37.705 | 38.433 | 34.857 | entailment: 75, neutral: 14, contradiction: 33 | entailment: 0.6585, neutral: 0.1333, contradiction: 0.3611 | [[27, 5, 9], [28, 6, 11], [20, 3, 13]] |
| startlux-decision-2b | mnli | gptq-blockdiag-adapted | 29.508 | 29.941 | 29.627 | entailment: 45, neutral: 25, contradiction: 52 | entailment: 0.2927, neutral: 0.2444, contradiction: 0.3611 | [[12, 9, 20], [15, 11, 19], [18, 5, 13]] |
| startlux-decision-2b | mnli | spinquant-nohad-adapted | 37.705 | 38.071 | 37.006 | entailment: 52, neutral: 32, contradiction: 38 | entailment: 0.5366, neutral: 0.2444, contradiction: 0.3611 | [[22, 13, 6], [15, 11, 19], [15, 8, 13]] |
| startlux-decision-2b | mnli | spinquant-had-adapted | 42.623 | 42.308 | 42.262 | entailment: 35, neutral: 54, contradiction: 33 | entailment: 0.3415, neutral: 0.5111, contradiction: 0.4167 | [[14, 19, 8], [12, 23, 10], [9, 12, 15]] |
| startlux-decision-2b | mnli | s1q-joint | 52.459 | 54.553 | 51.014 | entailment: 36, neutral: 23, contradiction: 63 | entailment: 0.5366, neutral: 0.2667, contradiction: 0.8333 | [[22, 6, 13], [13, 12, 20], [1, 5, 30]] |
| startlux-decision-2b | mnli | s1q-ac | 44.262 | 45.537 | 42.809 | entailment: 19, neutral: 34, contradiction: 69 | entailment: 0.2439, neutral: 0.4000, contradiction: 0.7222 | [[10, 9, 22], [6, 18, 21], [3, 7, 26]] |
| startlux-decision-2b | mnli | s1q-margin | 44.262 | 44.715 | 44.028 | entailment: 28, neutral: 45, contradiction: 49 | entailment: 0.3415, neutral: 0.4444, contradiction: 0.5556 | [[14, 15, 12], [8, 20, 17], [6, 10, 20]] |
| startlux-decision-2b | mnli | s1q | 43.443 | 43.379 | 43.266 | entailment: 37, neutral: 45, contradiction: 40 | entailment: 0.3902, neutral: 0.4667, contradiction: 0.4444 | [[16, 15, 10], [10, 21, 14], [11, 9, 16]] |
| startlux-decision-2b | tweet-offensive | native | 83.594 | 74.909 | 76.303 | not_offensive: 102, offensive: 26 | not_offensive: 0.9175, offensive: 0.5806 | [[89, 8], [13, 18]] |
| startlux-decision-2b | tweet-offensive | rtn | 33.594 | 53.991 | 32.669 | not_offensive: 16, offensive: 112 | not_offensive: 0.1443, offensive: 0.9355 | [[14, 83], [2, 29]] |
| startlux-decision-2b | tweet-offensive | s1q-local | 41.406 | 49.268 | 40.796 | not_offensive: 44, offensive: 84 | not_offensive: 0.3402, offensive: 0.6452 | [[33, 64], [11, 20]] |
| startlux-decision-2b | tweet-offensive | s1q2-beta05 | 42.969 | 53.592 | 42.685 | not_offensive: 40, offensive: 88 | not_offensive: 0.3299, offensive: 0.7419 | [[32, 65], [8, 23]] |
| startlux-decision-2b | tweet-offensive | smoothquant-adapted | 32.812 | 51.280 | 32.217 | not_offensive: 19, offensive: 109 | not_offensive: 0.1546, offensive: 0.8710 | [[15, 82], [4, 27]] |
| startlux-decision-2b | tweet-offensive | awq-adapted | 35.156 | 51.729 | 34.962 | not_offensive: 24, offensive: 104 | not_offensive: 0.1959, offensive: 0.8387 | [[19, 78], [5, 26]] |
| startlux-decision-2b | tweet-offensive | gptq-blockdiag-adapted | 46.875 | 50.682 | 44.939 | not_offensive: 55, offensive: 73 | not_offensive: 0.4330, offensive: 0.5806 | [[42, 55], [13, 18]] |
| startlux-decision-2b | tweet-offensive | spinquant-nohad-adapted | 39.844 | 51.530 | 39.752 | not_offensive: 36, offensive: 92 | not_offensive: 0.2887, offensive: 0.7419 | [[28, 69], [8, 23]] |
| startlux-decision-2b | tweet-offensive | spinquant-had-adapted | 75.000 | 56.069 | 56.089 | not_offensive: 115, offensive: 13 | not_offensive: 0.9278, offensive: 0.1935 | [[90, 7], [25, 6]] |
| startlux-decision-2b | tweet-offensive | s1q-joint | 67.188 | 66.279 | 62.321 | not_offensive: 77, offensive: 51 | not_offensive: 0.6804, offensive: 0.6452 | [[66, 31], [11, 20]] |
| startlux-decision-2b | tweet-offensive | s1q-ac | 71.875 | 64.982 | 63.955 | not_offensive: 91, offensive: 37 | not_offensive: 0.7835, offensive: 0.5161 | [[76, 21], [15, 16]] |
| startlux-decision-2b | tweet-offensive | s1q-margin | 74.219 | 72.015 | 68.884 | not_offensive: 84, offensive: 44 | not_offensive: 0.7629, offensive: 0.6774 | [[74, 23], [10, 21]] |
| startlux-decision-2b | tweet-offensive | s1q | 60.156 | 60.542 | 56.079 | not_offensive: 70, offensive: 58 | not_offensive: 0.5979, offensive: 0.6129 | [[58, 39], [12, 19]] |
| startlux-decision-2b | wanli | native | 68.750 | 67.782 | 67.936 | supported: 114, insufficient: 75, contradicted: 67 | supported: 0.8316, insufficient: 0.5513, contradicted: 0.6506 | [[79, 13, 3], [25, 43, 10], [10, 19, 54]] |
| startlux-decision-2b | wanli | rtn | 33.594 | 33.457 | 33.499 | supported: 93, insufficient: 73, contradicted: 90 | supported: 0.3579, insufficient: 0.3205, contradicted: 0.3253 | [[34, 21, 40], [30, 25, 23], [29, 27, 27]] |
| startlux-decision-2b | wanli | s1q-local | 33.984 | 34.472 | 33.628 | supported: 80, insufficient: 111, contradicted: 65 | supported: 0.2947, insufficient: 0.4744, contradicted: 0.2651 | [[28, 38, 29], [27, 37, 14], [25, 36, 22]] |
| startlux-decision-2b | wanli | s1q2-beta05 | 36.328 | 36.676 | 36.342 | supported: 82, insufficient: 80, contradicted: 94 | supported: 0.2947, insufficient: 0.3718, contradicted: 0.4337 | [[28, 30, 37], [28, 29, 21], [26, 21, 36]] |
| startlux-decision-2b | wanli | smoothquant-adapted | 35.938 | 36.072 | 35.925 | supported: 86, insufficient: 87, contradicted: 83 | supported: 0.3474, insufficient: 0.3974, contradicted: 0.3373 | [[33, 31, 31], [23, 31, 24], [30, 25, 28]] |
| startlux-decision-2b | wanli | awq-adapted | 30.078 | 30.581 | 29.908 | supported: 75, insufficient: 104, contradicted: 77 | supported: 0.2421, insufficient: 0.4103, contradicted: 0.2651 | [[23, 39, 33], [24, 32, 22], [28, 33, 22]] |
| startlux-decision-2b | wanli | gptq-blockdiag-adapted | 33.203 | 33.131 | 33.109 | supported: 81, insufficient: 80, contradicted: 95 | supported: 0.3368, insufficient: 0.3077, contradicted: 0.3494 | [[32, 28, 35], [23, 24, 31], [26, 28, 29]] |
| startlux-decision-2b | wanli | spinquant-nohad-adapted | 35.547 | 35.568 | 35.539 | supported: 88, insufficient: 80, contradicted: 88 | supported: 0.3579, insufficient: 0.3718, contradicted: 0.3373 | [[34, 28, 33], [22, 29, 27], [32, 23, 28]] |
| startlux-decision-2b | wanli | spinquant-had-adapted | 46.875 | 46.608 | 47.171 | supported: 97, insufficient: 103, contradicted: 56 | supported: 0.5158, insufficient: 0.4487, contradicted: 0.4337 | [[49, 37, 9], [32, 35, 11], [16, 31, 36]] |
| startlux-decision-2b | wanli | s1q-joint | 45.703 | 45.837 | 45.575 | supported: 86, insufficient: 111, contradicted: 59 | supported: 0.4632, insufficient: 0.5385, contradicted: 0.3735 | [[44, 33, 18], [26, 42, 10], [16, 36, 31]] |
| startlux-decision-2b | wanli | s1q-ac | 44.141 | 44.001 | 43.908 | supported: 84, insufficient: 102, contradicted: 70 | supported: 0.4842, insufficient: 0.4744, contradicted: 0.3614 | [[46, 30, 19], [20, 37, 21], [18, 35, 30]] |
| startlux-decision-2b | wanli | s1q-margin | 39.062 | 38.269 | 37.982 | supported: 107, insufficient: 97, contradicted: 52 | supported: 0.5368, insufficient: 0.3462, contradicted: 0.2651 | [[51, 33, 11], [32, 27, 19], [24, 37, 22]] |
| startlux-decision-2b | wanli | s1q | 42.969 | 42.259 | 42.276 | supported: 107, insufficient: 80, contradicted: 69 | supported: 0.5368, insufficient: 0.3333, contradicted: 0.3976 | [[51, 28, 16], [32, 26, 20], [24, 26, 33]] |
| startlux-decision-2b | wildjailbreak | native | 92.188 | 85.528 | 77.011 | harmful: 113, benign: 15 | harmful: 0.9328, benign: 0.7778 | [[111, 8], [2, 7]] |
| startlux-decision-2b | wildjailbreak | rtn | 75.781 | 51.027 | 48.701 | harmful: 102, benign: 26 | harmful: 0.7983, benign: 0.2222 | [[95, 24], [7, 2]] |
| startlux-decision-2b | wildjailbreak | s1q-local | 58.594 | 67.460 | 46.427 | harmful: 70, benign: 58 | harmful: 0.5714, benign: 0.7778 | [[68, 51], [2, 7]] |
| startlux-decision-2b | wildjailbreak | s1q2-beta05 | 57.812 | 56.769 | 43.750 | harmful: 73, benign: 55 | harmful: 0.5798, benign: 0.5556 | [[69, 50], [4, 5]] |
| startlux-decision-2b | wildjailbreak | smoothquant-adapted | 83.594 | 44.958 | 45.532 | harmful: 116, benign: 12 | harmful: 0.8992, benign: 0.0000 | [[107, 12], [9, 0]] |
| startlux-decision-2b | wildjailbreak | awq-adapted | 58.594 | 62.325 | 45.356 | harmful: 72, benign: 56 | harmful: 0.5798, benign: 0.6667 | [[69, 50], [3, 6]] |
| startlux-decision-2b | wildjailbreak | gptq-blockdiag-adapted | 65.625 | 55.836 | 46.908 | harmful: 85, benign: 43 | harmful: 0.6723, benign: 0.4444 | [[80, 39], [5, 4]] |
| startlux-decision-2b | wildjailbreak | spinquant-nohad-adapted | 54.688 | 44.818 | 39.583 | harmful: 73, benign: 55 | harmful: 0.5630, benign: 0.3333 | [[67, 52], [6, 3]] |
| startlux-decision-2b | wildjailbreak | spinquant-had-adapted | 22.656 | 58.403 | 22.081 | harmful: 20, benign: 108 | harmful: 0.1681, benign: 1.0000 | [[20, 99], [0, 9]] |
| startlux-decision-2b | wildjailbreak | s1q-joint | 47.656 | 71.849 | 40.998 | harmful: 52, benign: 76 | harmful: 0.4370, benign: 1.0000 | [[52, 67], [0, 9]] |
| startlux-decision-2b | wildjailbreak | s1q-ac | 25.781 | 49.813 | 24.109 | harmful: 28, benign: 100 | harmful: 0.2185, benign: 0.7778 | [[26, 93], [2, 7]] |
| startlux-decision-2b | wildjailbreak | s1q-margin | 34.375 | 64.706 | 31.551 | harmful: 35, benign: 93 | harmful: 0.2941, benign: 1.0000 | [[35, 84], [0, 9]] |
| startlux-decision-2b | wildjailbreak | s1q | 37.500 | 56.116 | 32.755 | harmful: 43, benign: 85 | harmful: 0.3445, benign: 0.7778 | [[41, 78], [2, 7]] |
| startlux-decision-4b | mnli | native | 89.344 | 89.783 | 89.454 | entailment: 44, neutral: 40, contradiction: 38 | entailment: 0.9268, neutral: 0.8222, contradiction: 0.9444 | [[38, 2, 1], [5, 37, 3], [1, 1, 34]] |
| startlux-decision-4b | mnli | rtn | 31.967 | 31.518 | 30.272 | entailment: 62, neutral: 31, contradiction: 29 | entailment: 0.5122, neutral: 0.2667, contradiction: 0.1667 | [[21, 10, 10], [20, 12, 13], [21, 9, 6]] |
| startlux-decision-4b | mnli | s1q-local | 34.426 | 33.871 | 33.528 | entailment: 30, neutral: 50, contradiction: 42 | entailment: 0.2439, neutral: 0.4667, contradiction: 0.3056 | [[10, 15, 16], [9, 21, 15], [11, 14, 11]] |
| startlux-decision-4b | mnli | s1q2-beta05 | 33.607 | 33.243 | 32.722 | entailment: 25, neutral: 59, contradiction: 38 | entailment: 0.2195, neutral: 0.4444, contradiction: 0.3333 | [[9, 20, 12], [11, 20, 14], [5, 19, 12]] |
| startlux-decision-4b | mnli | smoothquant-adapted | 29.508 | 29.079 | 28.333 | entailment: 66, neutral: 30, contradiction: 26 | entailment: 0.4390, neutral: 0.2667, contradiction: 0.1667 | [[18, 8, 15], [28, 12, 5], [20, 10, 6]] |
| startlux-decision-4b | mnli | awq-adapted | 45.902 | 45.086 | 44.741 | entailment: 26, neutral: 63, contradiction: 33 | entailment: 0.3415, neutral: 0.6222, contradiction: 0.3889 | [[14, 19, 8], [6, 28, 11], [6, 16, 14]] |
| startlux-decision-4b | mnli | gptq-blockdiag-adapted | 30.328 | 30.325 | 26.915 | entailment: 74, neutral: 22, contradiction: 26 | entailment: 0.6098, neutral: 0.1333, contradiction: 0.1667 | [[25, 8, 8], [27, 6, 12], [22, 8, 6]] |
| startlux-decision-4b | mnli | spinquant-nohad-adapted | 35.246 | 34.553 | 33.418 | entailment: 53, neutral: 43, contradiction: 26 | entailment: 0.5366, neutral: 0.3333, contradiction: 0.1667 | [[22, 11, 8], [18, 15, 12], [13, 17, 6]] |
| startlux-decision-4b | mnli | spinquant-had-adapted | 31.148 | 30.528 | 30.315 | entailment: 45, neutral: 46, contradiction: 31 | entailment: 0.3659, neutral: 0.3556, contradiction: 0.1944 | [[15, 16, 10], [15, 16, 14], [15, 14, 7]] |
| startlux-decision-4b | mnli | s1q-joint | 49.180 | 50.745 | 48.543 | entailment: 37, neutral: 31, contradiction: 54 | entailment: 0.4390, neutral: 0.3333, contradiction: 0.7500 | [[18, 11, 12], [15, 15, 15], [4, 5, 27]] |
| startlux-decision-4b | mnli | s1q-ac | 45.902 | 46.116 | 46.327 | entailment: 29, neutral: 49, contradiction: 44 | entailment: 0.4390, neutral: 0.4444, contradiction: 0.5000 | [[18, 17, 6], [5, 20, 20], [6, 12, 18]] |
| startlux-decision-4b | mnli | s1q-margin | 60.656 | 61.766 | 60.425 | entailment: 41, neutral: 31, contradiction: 50 | entailment: 0.6585, neutral: 0.4444, contradiction: 0.7500 | [[27, 6, 8], [10, 20, 15], [4, 5, 27]] |
| startlux-decision-4b | mnli | s1q | 61.475 | 62.909 | 60.918 | entailment: 41, neutral: 29, contradiction: 52 | entailment: 0.7317, neutral: 0.3778, contradiction: 0.7778 | [[30, 7, 4], [8, 17, 20], [3, 5, 28]] |
| startlux-decision-4b | tweet-offensive | native | 82.031 | 69.488 | 71.841 | not_offensive: 108, offensive: 20 | not_offensive: 0.9381, offensive: 0.4516 | [[91, 6], [17, 14]] |
| startlux-decision-4b | tweet-offensive | rtn | 46.875 | 51.779 | 45.258 | not_offensive: 53, offensive: 75 | not_offensive: 0.4227, offensive: 0.6129 | [[41, 56], [12, 19]] |
| startlux-decision-4b | tweet-offensive | s1q-local | 67.188 | 47.622 | 46.154 | not_offensive: 111, offensive: 17 | not_offensive: 0.8557, offensive: 0.0968 | [[83, 14], [28, 3]] |
| startlux-decision-4b | tweet-offensive | s1q2-beta05 | 70.312 | 50.782 | 49.648 | not_offensive: 113, offensive: 15 | not_offensive: 0.8866, offensive: 0.1290 | [[86, 11], [27, 4]] |
| startlux-decision-4b | tweet-offensive | smoothquant-adapted | 35.938 | 45.660 | 35.796 | not_offensive: 37, offensive: 91 | not_offensive: 0.2680, offensive: 0.6452 | [[26, 71], [11, 20]] |
| startlux-decision-4b | tweet-offensive | awq-adapted | 72.656 | 52.328 | 51.086 | not_offensive: 116, offensive: 12 | not_offensive: 0.9175, offensive: 0.1290 | [[89, 8], [27, 4]] |
| startlux-decision-4b | tweet-offensive | gptq-blockdiag-adapted | 46.875 | 55.071 | 46.032 | not_offensive: 47, offensive: 81 | not_offensive: 0.3918, offensive: 0.7097 | [[38, 59], [9, 22]] |
| startlux-decision-4b | tweet-offensive | spinquant-nohad-adapted | 52.344 | 50.998 | 47.999 | not_offensive: 68, offensive: 60 | not_offensive: 0.5361, offensive: 0.4839 | [[52, 45], [16, 15]] |
| startlux-decision-4b | tweet-offensive | spinquant-had-adapted | 56.250 | 54.672 | 51.515 | not_offensive: 71, offensive: 57 | not_offensive: 0.5773, offensive: 0.5161 | [[56, 41], [15, 16]] |
| startlux-decision-4b | tweet-offensive | s1q-joint | 79.688 | 60.259 | 61.481 | not_offensive: 119, offensive: 9 | not_offensive: 0.9794, offensive: 0.2258 | [[95, 2], [24, 7]] |
| startlux-decision-4b | tweet-offensive | s1q-ac | 79.688 | 60.259 | 61.481 | not_offensive: 119, offensive: 9 | not_offensive: 0.9794, offensive: 0.2258 | [[95, 2], [24, 7]] |
| startlux-decision-4b | tweet-offensive | s1q-margin | 78.906 | 59.744 | 60.794 | not_offensive: 118, offensive: 10 | not_offensive: 0.9691, offensive: 0.2258 | [[94, 3], [24, 7]] |
| startlux-decision-4b | tweet-offensive | s1q | 81.250 | 64.583 | 67.067 | not_offensive: 115, offensive: 13 | not_offensive: 0.9691, offensive: 0.3226 | [[94, 3], [21, 10]] |
| startlux-decision-4b | wanli | native | 74.609 | 73.169 | 72.583 | supported: 124, insufficient: 48, contradicted: 84 | supported: 0.9263, insufficient: 0.4615, contradicted: 0.8072 | [[88, 4, 3], [28, 36, 14], [8, 8, 67]] |
| startlux-decision-4b | wanli | rtn | 32.422 | 32.355 | 32.084 | supported: 95, insufficient: 91, contradicted: 70 | supported: 0.3579, insufficient: 0.3718, contradicted: 0.2410 | [[34, 33, 28], [27, 29, 22], [34, 29, 20]] |
| startlux-decision-4b | wanli | s1q-local | 34.766 | 34.894 | 34.084 | supported: 88, insufficient: 105, contradicted: 63 | supported: 0.3684, insufficient: 0.4615, contradicted: 0.2169 | [[35, 35, 25], [22, 36, 20], [31, 34, 18]] |
| startlux-decision-4b | wanli | s1q2-beta05 | 30.859 | 30.926 | 30.729 | supported: 97, insufficient: 93, contradicted: 66 | supported: 0.3158, insufficient: 0.3590, contradicted: 0.2530 | [[30, 41, 24], [29, 28, 21], [38, 24, 21]] |
| startlux-decision-4b | wanli | smoothquant-adapted | 35.547 | 35.619 | 35.549 | supported: 84, insufficient: 82, contradicted: 90 | supported: 0.3474, insufficient: 0.3718, contradicted: 0.3494 | [[33, 27, 35], [23, 29, 26], [28, 26, 29]] |
| startlux-decision-4b | wanli | awq-adapted | 37.109 | 37.150 | 37.162 | supported: 89, insufficient: 107, contradicted: 60 | supported: 0.3789, insufficient: 0.4103, contradicted: 0.3253 | [[36, 44, 15], [28, 32, 18], [25, 31, 27]] |
| startlux-decision-4b | wanli | gptq-blockdiag-adapted | 33.984 | 34.342 | 34.049 | supported: 83, insufficient: 83, contradicted: 90 | supported: 0.2737, insufficient: 0.3590, contradicted: 0.3976 | [[26, 29, 40], [33, 28, 17], [24, 26, 33]] |
| startlux-decision-4b | wanli | spinquant-nohad-adapted | 32.812 | 32.527 | 32.431 | supported: 91, insufficient: 95, contradicted: 70 | supported: 0.3895, insufficient: 0.3333, contradicted: 0.2530 | [[37, 33, 25], [28, 26, 24], [26, 36, 21]] |
| startlux-decision-4b | wanli | spinquant-had-adapted | 40.625 | 40.587 | 40.570 | supported: 79, insufficient: 87, contradicted: 90 | supported: 0.4105, insufficient: 0.3974, contradicted: 0.4096 | [[39, 28, 28], [19, 31, 28], [21, 28, 34]] |
| startlux-decision-4b | wanli | s1q-joint | 51.172 | 50.800 | 50.740 | supported: 95, insufficient: 108, contradicted: 53 | supported: 0.6000, insufficient: 0.5385, contradicted: 0.3855 | [[57, 29, 9], [24, 42, 12], [14, 37, 32]] |
| startlux-decision-4b | wanli | s1q-ac | 50.391 | 49.864 | 49.114 | supported: 93, insufficient: 54, contradicted: 109 | supported: 0.5368, insufficient: 0.3205, contradicted: 0.6386 | [[51, 17, 27], [24, 25, 29], [18, 12, 53]] |
| startlux-decision-4b | wanli | s1q-margin | 55.078 | 54.305 | 54.278 | supported: 114, insufficient: 65, contradicted: 77 | supported: 0.6526, insufficient: 0.4103, contradicted: 0.5663 | [[62, 19, 14], [30, 32, 16], [22, 14, 47]] |
| startlux-decision-4b | wanli | s1q | 55.078 | 53.718 | 52.732 | supported: 117, insufficient: 53, contradicted: 86 | supported: 0.7263, insufficient: 0.2949, contradicted: 0.5904 | [[69, 15, 11], [29, 23, 26], [19, 15, 49]] |
| startlux-decision-4b | wildjailbreak | native | 97.656 | 88.469 | 90.549 | harmful: 120, benign: 8 | harmful: 0.9916, benign: 0.7778 | [[118, 1], [2, 7]] |
| startlux-decision-4b | wildjailbreak | rtn | 59.375 | 47.339 | 42.041 | harmful: 79, benign: 49 | harmful: 0.6134, benign: 0.3333 | [[73, 46], [6, 3]] |
| startlux-decision-4b | wildjailbreak | s1q-local | 25.000 | 54.528 | 23.810 | harmful: 25, benign: 103 | harmful: 0.2017, benign: 0.8889 | [[24, 95], [1, 8]] |
| startlux-decision-4b | wildjailbreak | s1q2-beta05 | 23.438 | 53.688 | 22.511 | harmful: 23, benign: 105 | harmful: 0.1849, benign: 0.8889 | [[22, 97], [1, 8]] |
| startlux-decision-4b | wildjailbreak | smoothquant-adapted | 71.094 | 48.506 | 46.273 | harmful: 96, benign: 32 | harmful: 0.7479, benign: 0.2222 | [[89, 30], [7, 2]] |
| startlux-decision-4b | wildjailbreak | awq-adapted | 14.844 | 54.202 | 14.839 | harmful: 10, benign: 118 | harmful: 0.0840, benign: 1.0000 | [[10, 109], [0, 9]] |
| startlux-decision-4b | wildjailbreak | gptq-blockdiag-adapted | 60.156 | 37.488 | 39.325 | harmful: 84, benign: 44 | harmful: 0.6387, benign: 0.1111 | [[76, 43], [8, 1]] |
| startlux-decision-4b | wildjailbreak | spinquant-nohad-adapted | 55.469 | 45.238 | 39.995 | harmful: 74, benign: 54 | harmful: 0.5714, benign: 0.3333 | [[68, 51], [6, 3]] |
| startlux-decision-4b | wildjailbreak | spinquant-had-adapted | 53.906 | 49.533 | 40.362 | harmful: 70, benign: 58 | harmful: 0.5462, benign: 0.4444 | [[65, 54], [5, 4]] |
| startlux-decision-4b | wildjailbreak | s1q-joint | 25.000 | 59.664 | 24.092 | harmful: 23, benign: 105 | harmful: 0.1933, benign: 1.0000 | [[23, 96], [0, 9]] |
| startlux-decision-4b | wildjailbreak | s1q-ac | 41.406 | 68.487 | 36.671 | harmful: 44, benign: 84 | harmful: 0.3697, benign: 1.0000 | [[44, 75], [0, 9]] |
| startlux-decision-4b | wildjailbreak | s1q-margin | 40.625 | 62.932 | 35.525 | harmful: 45, benign: 83 | harmful: 0.3697, benign: 0.8889 | [[44, 75], [1, 8]] |
| startlux-decision-4b | wildjailbreak | s1q | 36.719 | 65.966 | 33.295 | harmful: 38, benign: 90 | harmful: 0.3193, benign: 1.0000 | [[38, 81], [0, 9]] |

Macro-F1 averages the fixed classes with zero F1 for an empty gold/prediction class. Balanced accuracy averages gold-present class recalls; undefined recalls remain explicit. All metrics in CSV/JSON retain unrounded inputs.
