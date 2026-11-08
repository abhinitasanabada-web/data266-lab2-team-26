|                                                                 | base (baseline)   | run1 (final)    | run2            |
|:----------------------------------------------------------------|:------------------|:----------------|:----------------|
| Images not blocked by the NSFW filter (of 40)                   | 39/40             | 33/40           | 32/40           |
| KID in-domain ↓ (unblocked in-domain gens vs test, mean ± std)  | 0.1117 ± 0.0181   | 0.0087 ± 0.0044 | 0.0053 ± 0.0041 |
| CLIP score ↑ (all prompts, blocked images excluded)             | 25.28             | 25.32           | 25.32           |
| CLIP score (all 40 images, black images included)               | 25.07             | 23.84           | 23.53           |
| CLIP score ↑ (in_domain)                                        | 23.9              | 23.4            | 23.33           |
| CLIP score ↑ (out_of_domain)                                    | 24.73             | 26.15           | 25.94           |
| CLIP score ↑ (compositional)                                    | 28.54             | 27.38           | 28.01           |
| LPIPS diversity ↑ (seed 0 vs 1, all prompts)                    | 0.751             | 0.751           | 0.751           |
| LPIPS diversity (in_domain)                                     | 0.745             | 0.761           | 0.762           |
| LPIPS diversity (out_of_domain)                                 | 0.689             | 0.721           | 0.721           |
| LPIPS diversity (compositional)                                 | 0.812             | 0.772           | 0.774           |
| Human eval: win rate vs base, domain style                      | —                 | 0.84            | —               |
| Human eval: win rate vs base, prompt match                      | —                 | 0.45            | —               |
| Selected checkpoint × adapter strength (Section 3.6)            | —                 | step 1000 × 1.0 | step 1000 × 1.0 |
| Trainable parameters                                            | 0 (no training)   | 3,188,736       | 3,288,576       |
| GPU-hours per training run                                      | 0.0               | 1.009           | 1.225           |
| Peak GPU memory, training (GiB reserved)                        | —                 | 3.22            | 3.31            |
| Median generation time (s/image)                                | 5.23              | 5.51            | 6.26            |
| NSFW-flagged images (of 40)                                     | 1                 | 7               | 8               |
| Memorization: max NN CLIP sim to train (10 unblocked in-domain) | 0.7496            | 0.8618          | —               |